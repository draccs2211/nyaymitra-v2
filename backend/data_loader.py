"""
Script to load legal data into ChromaDB.
Run this once before starting the server.
Usage: python -m backend.data_loader
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import hashlib
import re

# Official short titles (only acts named in the corpus text itself)
IPC = "The Indian Penal Code, 1860"
CRPC = "The Code of Criminal Procedure, 1973"
UP_RENT_ACT = "The Uttar Pradesh Urban Buildings (Regulation of Letting, Rent and Eviction) Act, 1972"
CPA = "The Consumer Protection Act, 2019"
RTI = "The Right to Information Act, 2005"
WAGES = "The Payment of Wages Act, 1936"
IDA = "The Industrial Disputes Act, 1947"
POSH = "The Sexual Harassment of Women at Workplace (Prevention, Prohibition and Redressal) Act, 2013"
HMA = "The Hindu Marriage Act, 1955"
PWDVA = "The Protection of Women from Domestic Violence Act, 2005"

CURATED_SOURCE = "NyayMitra curated summary (unverified — check official text)"


def _m(domain: str, doc_type: str, **fields) -> dict:
    """Build corpus metadata; full schema/validation in backend.rag.metadata."""
    return {"domain": domain, "doc_type": doc_type, "source_name": CURATED_SOURCE, **fields}

LEGAL_DATA = [
    # IPC / BNS Sections
    {
        "content": "Section 302 IPC / Section 101 BNS - Murder: Whoever commits murder shall be punished with death or imprisonment for life and shall also be liable to fine. Murder is defined as culpable homicide committed with intention to cause death.",
        "metadata": _m("criminal", "ipc_section", act=IPC, section="302")
    },
    {
        "content": "Section 420 IPC / Section 318 BNS - Cheating: Whoever cheats and thereby dishonestly induces the person deceived to deliver any property or to make or alter a valuable security shall be punished with imprisonment up to 7 years and fine.",
        "metadata": _m("criminal", "ipc_section", act=IPC, section="420")
    },
    {
        "content": "Section 498A IPC / Section 85 BNS - Cruelty by husband or relatives: Whoever being the husband or relative of husband subjects a woman to cruelty shall be punished with imprisonment up to 3 years and fine. Cruelty includes harassment for dowry.",
        "metadata": _m("criminal", "ipc_section", act=IPC, section="498A")
    },
    {
        "content": "Section 406 IPC / Section 316 BNS - Criminal breach of trust: Whoever commits criminal breach of trust shall be punished with imprisonment up to 3 years or fine or both. Security deposit not returned by landlord can fall under this section.",
        "metadata": _m("criminal", "ipc_section", act=IPC, section="406")
    },
    {
        "content": "Section 376 IPC / Section 63 BNS - Rape: Punishment for rape is minimum 7 years imprisonment which may extend to life imprisonment and fine. Victim can file FIR at any police station under Zero FIR provision.",
        "metadata": _m("criminal", "ipc_section", act=IPC, section="376")
    },

    # Tenant Rights
    {
        "content": "UP Urban Buildings (Regulation of Letting, Rent and Eviction) Act 1972 - Tenant Rights: Landlord cannot evict tenant without court order from Rent Control Court. Security deposit must be returned within 30 days of vacating the premises. Failure to return deposit is punishable.",
        "metadata": _m("property", "act", act=UP_RENT_ACT, jurisdiction="IN-UP")
    },
    {
        "content": "Security deposit return: Under UP Urban Buildings Act, if landlord fails to return security deposit within 30 days, tenant can file complaint in Rent Control Court. Tenant can also send legal notice under Section 406 IPC for criminal breach of trust.",
        "metadata": _m("property", "legal_guidance", act=UP_RENT_ACT, topic="security_deposit", jurisdiction="IN-UP")
    },
    {
        "content": "Eviction of tenant in UP: Landlord can only evict tenant on specific grounds - non-payment of rent, subletting without permission, damaging property, landlord's genuine need. Eviction requires court order from Rent Control Court. Self-help eviction is illegal.",
        "metadata": _m("property", "legal_guidance", act=UP_RENT_ACT, topic="eviction", jurisdiction="IN-UP")
    },

    # Consumer Law
    {
        "content": "Consumer Protection Act 2019 - Jurisdiction: District Consumer Commission handles complaints up to Rs 50 lakh. State Consumer Commission handles Rs 50 lakh to Rs 2 crore. National Consumer Commission handles above Rs 2 crore. Online complaints can be filed at edaakhil.nic.in",
        "metadata": _m("consumer", "act", act=CPA)
    },
    {
        "content": "Consumer complaint process: File complaint at appropriate Consumer Forum based on claim amount. Attach purchase receipt, warranty card, communication with company. Forum gives opportunity to opposite party to reply. Award must be complied within 30 days.",
        "metadata": _m("consumer", "legal_guidance", act=CPA, topic="consumer_complaint")
    },
    {
        "content": "Defective product consumer rights: Under Consumer Protection Act 2019, consumer can claim replacement, refund, compensation for defective product. Complaint must be filed within 2 years of cause of action. Builder, e-commerce companies also covered under the Act.",
        "metadata": _m("consumer", "legal_guidance", act=CPA, topic="defective_product")
    },

    # RTI
    {
        "content": "Right to Information Act 2005 - Process: File application to Public Information Officer (PIO) with Rs 10 fee. BPL category is exempt from fee. PIO must provide information within 30 days. For life and liberty matters, response in 48 hours. Online filing at rtionline.gov.in",
        "metadata": _m("rti", "act", act=RTI)
    },
    {
        "content": "RTI Appeal process: If PIO does not respond within 30 days or gives incomplete information, file First Appeal with First Appellate Authority within 30 days. Second Appeal to Central/State Information Commission within 90 days. Can claim compensation for delay.",
        "metadata": _m("rti", "legal_guidance", act=RTI, topic="rti_appeal")
    },

    # Labor Law
    {
        "content": "Payment of Wages Act 1936 - Salary not paid: Employer must pay wages on time. Unauthorized deductions are illegal. Worker can file complaint before Payment of Wages Authority (Labour Commissioner). Employer liable to pay wages plus compensation.",
        "metadata": _m("labor", "act", act=WAGES)
    },
    {
        "content": "Wrongful termination: Employee can challenge wrongful termination under Industrial Disputes Act. File complaint with Labour Commissioner or Industrial Tribunal. Employee entitled to notice period or pay in lieu. Reinstatement or compensation can be ordered.",
        "metadata": _m("labor", "legal_guidance", act=IDA, topic="wrongful_termination")
    },
    {
        "content": "Workplace harassment: Sexual harassment at workplace is covered under POSH Act 2013. File complaint with Internal Complaints Committee (ICC). If no ICC, file with Local Complaints Committee. Criminal complaint under IPC Section 354 also applicable.",
        "metadata": _m("labor", "legal_guidance", act=POSH, topic="workplace_harassment")
    },

    # Criminal Procedure
    {
        "content": "FIR - First Information Report: Under Section 154 CrPC / Section 173 BNSS, police must register FIR for cognizable offences. If police refuses, file complaint to SP or Judicial Magistrate. Zero FIR can be filed at any police station regardless of jurisdiction.",
        "metadata": _m("criminal", "legal_guidance", act=CRPC, section="154", topic="fir")
    },
    {
        "content": "Bail types in India: Regular Bail under Section 437/439 CrPC after arrest. Anticipatory Bail under Section 438 CrPC before arrest. Interim bail is temporary. Statutory bail when chargesheet not filed within 60/90 days. Default bail is fundamental right.",
        "metadata": _m("criminal", "legal_guidance", act=CRPC, topic="bail")
    },

    # Family Law
    {
        "content": "Divorce under Hindu Marriage Act 1955: Grounds for divorce include adultery, cruelty, desertion for 2 years, conversion, mental disorder, communicable disease. Mutual consent divorce possible after 1 year of marriage with 6 month cooling off period.",
        "metadata": _m("family", "legal_guidance", act=HMA, topic="divorce")
    },
    {
        "content": "Child custody in India: Courts decide custody based on best interest of child. Mother gets custody of children below 5 years usually. Visitation rights given to non-custodial parent. Custody order can be modified based on changed circumstances.",
        "metadata": _m("family", "legal_guidance", topic="custody")
    },
    {
        "content": "Maintenance under Section 125 CrPC: Wife, children, parents can claim maintenance from husband/son. Magistrate can order monthly allowance. Interim maintenance can be ordered during proceedings. Failure to pay maintenance is punishable.",
        "metadata": _m("family", "legal_guidance", act=CRPC, section="125", topic="maintenance")
    },

    # Domestic Violence
    {
        "content": "Protection of Women from Domestic Violence Act 2005: Covers physical, sexual, verbal, emotional and economic abuse. Woman can get Protection Order, Residence Order, Monetary Relief, Custody Order. Contact Protection Officers or NGOs. Emergency helpline: 181.",
        "metadata": _m("criminal", "act", act=PWDVA)
    },

    # Property
    {
        "content": "Property dispute resolution in UP: Land disputes go to Revenue Court (Tehsildar/SDM). Property title disputes go to Civil Court. Encroachment complaints to SDM/District Magistrate. Registration disputes to Sub-Registrar. Mutation in revenue records through Tehsildar.",
        "metadata": _m("property", "legal_guidance", topic="property_dispute", jurisdiction="IN-UP")
    },

    # Legal Aid
    {
        "content": "Free legal aid in India: NALSA provides free legal services to persons with income below Rs 3 lakh, SC/ST, women, children, disabled persons, victims of trafficking, industrial workmen. Contact District Legal Services Authority (DLSA) or call 15100.",
        "metadata": _m("general", "legal_guidance", topic="legal_aid")
    },
    {
        "content": "Lok Adalat: Alternative dispute resolution where disputes are settled amicably. No court fees, award is final with no appeal. Suitable for motor accident claims, matrimonial disputes, labour disputes. Held monthly at District Courts. Contact DLSA for schedule.",
        "metadata": _m("general", "legal_guidance", topic="lok_adalat")
    },

    # Emergency
    {
        "content": "Emergency legal helplines India: Police: 100, Women helpline: 1091, Child helpline: 1098, Senior citizen: 14567, Legal aid: 15100, Domestic violence: 181, Consumer helpline: 1800-11-4000, Anti-corruption: 1064",
        "metadata": _m("general", "helplines", topic="emergency")
    },
]


def make_canonical_id(doc: dict) -> str:
    """
    Deterministic ID: '<doc_type>:<sha256(normalized content)[:16]>'.
    Same content -> same ID on every run, so loads are idempotent.
    Metadata is excluded so metadata fixes update in place via upsert.
    """
    doc_type = doc.get("metadata", {}).get("doc_type", "doc")
    normalized = re.sub(r"\s+", " ", doc["content"]).strip().lower()
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]
    return f"{doc_type}:{digest}"


def build_documents() -> list:
    return [{**doc, "id": make_canonical_id(doc)} for doc in LEGAL_DATA]


def load_data(prune: bool = True):
    """
    Idempotent load. With prune=True, removes docs not in the current corpus
    (e.g. legacy random-UUID entries or docs whose content was edited).
    """
    from backend.rag.hybrid_search import hybrid_rag  # lazy: loads embedding model

    docs = build_documents()
    print("Loading legal corpus into ChromaDB...")
    print(f"Total documents: {len(docs)}")

    try:
        hybrid_rag.add_documents(docs)
        if prune:
            removed = hybrid_rag.delete_ids_not_in([d["id"] for d in docs])
            print(f"🧹 Removed {removed} stale docs")
        stats = hybrid_rag.get_stats()
        print(f"✅ Stats: {stats}")
        if not stats["ids_in_sync"]:
            raise RuntimeError("ChromaDB and BM25 IDs are out of sync")
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        raise


if __name__ == "__main__":
    load_data()
    print("\n✅ Data loading complete! You can now start the server.")
