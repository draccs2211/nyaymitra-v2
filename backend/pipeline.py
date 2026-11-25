from langgraph.graph import END, START, StateGraph
from backend.state import NyayMitraState, initial_state
from backend.agents.guardian import guardian_agent
from backend.agents.orchestrator import orchestrator_agent
from backend.agents.research import research_agent
from backend.agents.drafting import drafting_agent
from backend.agents.analysis import analysis_agent
from backend.agents.verification import verification_agent
from backend.agents.referral import referral_agent
from backend.agents.schemas import (
    ApprovalDecision, ApprovalRecord, HumanApprovalOutput, to_state
)
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command, interrupt
from datetime import datetime, timezone
from typing import Literal, Optional
import asyncio
import uuid


def _wants(state: NyayMitraState, action: str) -> bool:
    return action in (state.get("requested_actions") or ["research"])


def route_after_orchestrator(state: NyayMitraState) -> Literal["research", "referral_only"]:
    # Research + analysis are skipped only when referral is the sole request
    actions = set(state.get("requested_actions") or ["research"])
    if actions == {"referral"}:
        return "referral_only"
    return "research"


def route_after_analysis(state: NyayMitraState) -> Literal["drafting", "referral", "verification"]:
    if _wants(state, "drafting"):
        return "drafting"
    if _wants(state, "referral"):
        return "referral"
    return "verification"


def route_after_drafting(state: NyayMitraState) -> Literal["referral", "verification"]:
    if _wants(state, "referral"):
        return "referral"
    return "verification"


def route_after_verification(state: NyayMitraState) -> Literal["end", "human_approval"]:
    if state.get("human_approval_needed", False):
        return "human_approval"
    return "end"


def human_approval_node(state: NyayMitraState) -> dict:
    """
    Pause the run until a reviewer decides (LangGraph interrupt).
    Everything before interrupt() re-runs on resume, so keep it side-effect free.
    """
    raw = interrupt({
        "type": "approval_request",
        "final_response": state.get("final_response", ""),
        "draft_document": state.get("draft_document", ""),
        "document_type": state.get("document_type", ""),
        "draft_missing_fields": state.get("draft_missing_fields", []),
        "draft_unverified_citations": state.get("draft_unverified_citations", []),
        "confidence_score": state.get("confidence_score", 0.0),
        "hallucination_detected": state.get("hallucination_detected", False),
        "verified_sources": state.get("verified_sources", []),
    })
    decision = ApprovalDecision.model_validate(raw)
    edited = decision.edited_response is not None
    return to_state(HumanApprovalOutput(
        approval_status=decision.decision,
        approval=ApprovalRecord(
            reviewer_id=decision.reviewer_id,
            notes=decision.notes,
            decided_at=datetime.now(timezone.utc).isoformat(),
            edited=edited,
        ),
        final_response=decision.edited_response if edited else None,
    ))


def dispatch_node(state: NyayMitraState) -> dict:
    """Join point after the parallel guardian + orchestrator; routing happens on its edges."""
    return {}


def get_checkpointer():
    """
    Checkpointer holding paused runs between interrupt and approval.
    In-memory: paused runs are lost on restart and not shared across workers.
    Swap for a Postgres saver (langgraph-checkpoint-postgres) for production.
    """
    return InMemorySaver()


def build_graph(checkpointer=None):
    workflow = StateGraph(NyayMitraState)

    # Add all nodes
    workflow.add_node("guardian", guardian_agent)
    workflow.add_node("orchestrator", orchestrator_agent)
    workflow.add_node("research", research_agent)
    workflow.add_node("analysis", analysis_agent)
    workflow.add_node("drafting", drafting_agent)
    workflow.add_node("verification", verification_agent)
    workflow.add_node("referral", referral_agent)
    workflow.add_node("human_approval", human_approval_node)
    workflow.add_node("dispatch", dispatch_node)

    # Guardian (urgency) and orchestrator (classification) are independent LLM
    # calls: run them in parallel, join at dispatch. Emergencies continue too.
    workflow.add_edge(START, "guardian")
    workflow.add_edge(START, "orchestrator")
    workflow.add_edge(["guardian", "orchestrator"], "dispatch")

    # Dispatch → research or referral only
    workflow.add_conditional_edges(
        "dispatch",
        route_after_orchestrator,
        {
            "research": "research",
            "referral_only": "referral"
        }
    )

    # Research → analysis (always)
    workflow.add_edge("research", "analysis")

    # Analysis → drafting / referral / verification
    workflow.add_conditional_edges(
        "analysis",
        route_after_analysis,
        {
            "drafting": "drafting",
            "referral": "referral",
            "verification": "verification"
        }
    )

    # Drafting → referral / verification
    workflow.add_conditional_edges(
        "drafting",
        route_after_drafting,
        {
            "referral": "referral",
            "verification": "verification"
        }
    )

    # Referral → verification
    workflow.add_edge("referral", "verification")

    # Verification → end or human approval
    workflow.add_conditional_edges(
        "verification",
        route_after_verification,
        {
            "end": END,
            "human_approval": "human_approval"
        }
    )

    # Human approval → end
    workflow.add_edge("human_approval", END)

    return workflow.compile(checkpointer=checkpointer)


# Compile the graph
nyaymitra_graph = build_graph(checkpointer=get_checkpointer())


def _config(run_id: str) -> dict:
    return {"configurable": {"thread_id": run_id}}


async def _snapshot_result(run_id: str) -> dict:
    snap = await asyncio.to_thread(nyaymitra_graph.get_state, _config(run_id))
    result = dict(snap.values)
    result["run_id"] = run_id
    if "human_approval" in (snap.next or ()):
        result["approval_status"] = "pending"
    return result


async def run_pipeline(
    query: str,
    session_id: str,
    user_id: str,
    conversation_history: list = None,
    run_id: Optional[str] = None,
    jurisdiction: str = "IN",
) -> dict:
    """Run the graph. If approval is needed, returns with approval_status='pending'."""
    run_id = run_id or str(uuid.uuid4())
    state = initial_state(
        user_query=query,
        session_id=session_id,
        user_id=user_id,
        conversation_history=conversation_history or [],
        jurisdiction=jurisdiction,
    )
    # Sync invoke in a worker thread: agents are sync (would block the event
    # loop) and interrupt() needs contextvars that ainvoke loses on Python 3.10
    await asyncio.to_thread(nyaymitra_graph.invoke, state, _config(run_id))
    return await _snapshot_result(run_id)


async def get_run(run_id: str) -> Optional[dict]:
    """Current state of a run, or None if unknown."""
    snap = await asyncio.to_thread(nyaymitra_graph.get_state, _config(run_id))
    if not snap.values:
        return None
    return await _snapshot_result(run_id)


# Human-readable stage for progress UIs, keyed by the node about to run.
STAGES = {
    "guardian": "screening", "orchestrator": "classifying", "dispatch": "classifying",
    "research": "researching", "analysis": "analysing", "drafting": "drafting",
    "referral": "finding_legal_aid", "verification": "verifying", "human_approval": "awaiting_approval",
}


async def run_progress(run_id: str) -> Optional[dict]:
    """
    Lightweight progress for polling while a run executes. None if unknown.
    status: running | awaiting_approval | completed
    """
    snap = await asyncio.to_thread(nyaymitra_graph.get_state, _config(run_id))
    if not snap.values:
        return None
    nxt = list(snap.next or ())
    if not nxt:
        status, stage = "completed", "completed"
    elif nxt == ["human_approval"] and snap.tasks and any(t.interrupts for t in snap.tasks):
        status, stage = "awaiting_approval", "awaiting_approval"
    else:
        status, stage = "running", STAGES.get(nxt[0], nxt[0])
    return {"run_id": run_id, "status": status, "stage": stage, "next": nxt,
            "user_id": snap.values.get("user_id")}


async def resume_pipeline(run_id: str, decision: dict) -> dict:
    """
    Resume a paused run with a reviewer decision.
    Raises KeyError if run is unknown, ValueError if it isn't awaiting approval
    or the decision is invalid (validated before resuming, so a bad decision
    doesn't consume the interrupt).
    """
    current = await get_run(run_id)
    if current is None:
        raise KeyError(run_id)
    if current.get("approval_status") != "pending":
        raise ValueError(f"run is not awaiting approval (status: {current.get('approval_status')})")
    validated = ApprovalDecision.model_validate(decision)
    await asyncio.to_thread(
        nyaymitra_graph.invoke, Command(resume=validated.model_dump()), _config(run_id)
    )
    return await _snapshot_result(run_id)
