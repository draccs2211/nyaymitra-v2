from langgraph.graph import StateGraph, END
from backend.state import NyayMitraState, initial_state
from backend.agents.guardian import guardian_agent
from backend.agents.orchestrator import orchestrator_agent
from backend.agents.research import research_agent
from backend.agents.drafting import drafting_agent
from backend.agents.analysis import analysis_agent
from backend.agents.verification import verification_agent
from backend.agents.referral import referral_agent
from backend.agents.schemas import HumanApprovalOutput, to_state
from typing import Literal


def check_emergency(state: NyayMitraState) -> Literal["emergency", "normal"]:
    if state.get("is_emergency", False):
        return "emergency"
    return "normal"


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
    # In production this would pause and wait for human input
    # For now, we pass through with a note
    response = state.get("final_response", "")
    note = "\n\n🔍 [This response has been flagged for review due to lower confidence. Please verify before use.]"
    return to_state(HumanApprovalOutput(final_response=response + note))


def build_graph():
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

    # Entry point
    workflow.set_entry_point("guardian")

    # Guardian → emergency or normal
    workflow.add_conditional_edges(
        "guardian",
        check_emergency,
        {
            "emergency": END,
            "normal": "orchestrator"
        }
    )

    # Orchestrator → research or referral only
    workflow.add_conditional_edges(
        "orchestrator",
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

    return workflow.compile()


# Compile the graph
nyaymitra_graph = build_graph()


async def run_pipeline(
    query: str,
    session_id: str,
    user_id: str,
    conversation_history: list = None
) -> dict:
    state = initial_state(
        user_query=query,
        session_id=session_id,
        user_id=user_id,
        conversation_history=conversation_history or [],
    )

    result = await nyaymitra_graph.ainvoke(state)
    return result
