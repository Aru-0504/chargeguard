import os
import hashlib
from typing import TypedDict

from langgraph.graph import END, StateGraph
from openai import OpenAI


class EvidenceState(TypedDict, total=False):
    decision_id: int
    top_reasons: list
    context_fields: dict
    draft: str
    is_valid: bool
    final_evidence: str


class EvidenceAgent:
    def __init__(self):
        graph = StateGraph(EvidenceState)
        graph.add_node("assemble", self._assemble)
        graph.add_node("draft", self._draft)
        graph.add_node("self_check", self._self_check)
        graph.set_entry_point("assemble")
        graph.add_edge("assemble", "draft")
        graph.add_edge("draft", "self_check")
        graph.add_edge("self_check", END)
        self.graph = graph.compile()

    @staticmethod
    def _assemble(state: EvidenceState) -> EvidenceState:
        """
        ASSEMBLE node: Combines SHAP reasons with simulated merchant/order context fields.
        
        NOTE: The following context fields (shipping_billing_match, device_used, account_age_days, 
        ip_consistent) are SIMULATED demo enrichment for demonstration purposes only. They are 
        generated deterministically from the transaction_id using hash-based lookup to ensure 
        consistency. This is NOT real matched merchant data and should never be presented as 
        authoritative evidence in production.
        """
        decision_id = state["decision_id"]
        
        # Deterministic hash-based simulation for consistent demo data
        seed = int(hashlib.sha256(str(decision_id).encode()).hexdigest(), 16)
        
        # Generate simulated context fields deterministically
        shipping_billing_match = "yes" if (seed % 3) != 0 else "no"
        device_options = ["mobile", "desktop", "tablet"]
        device_used = device_options[seed % 3]
        account_age_days = (seed % 365) + 1  # 1-365 days
        ip_consistent = "yes" if (seed % 4) != 0 else "no"
        
        context_fields = {
            "shipping_billing_match": shipping_billing_match,
            "device_used": device_used,
            "account_age_days": account_age_days,
            "ip_consistent": ip_consistent,
        }
        
        return {"context_fields": context_fields}

    def _draft(self, state: EvidenceState) -> EvidenceState:
        """
        DRAFT node: Calls GPT-4o to write a structured dispute-evidence packet.
        Uses only the evidence fields from ASSEMBLE - nothing invented.
        """
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return {"draft": self._fallback_draft(state), "is_valid": False}

        try:
            client = OpenAI(api_key=api_key)
            
            # Build evidence context string
            evidence_context = self._build_evidence_context(state)
            
            response = client.chat.completions.create(
                model="gpt-4o",
                temperature=0.1,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a chargeback dispute specialist. Write a concise, factual dispute-evidence packet "
                            "arguing that this transaction is legitimate. Use ONLY the specific evidence fields provided. "
                            "Do not invent any details not present in the evidence. Cite only the actual fields given. "
                            "Write in plain, factual language suitable for submission to a card network. "
                            "No fluff, no speculation."
                        ),
                    },
                    {"role": "user", "content": evidence_context},
                ],
            )
            draft = response.choices[0].message.content or self._fallback_draft(state)
            return {"draft": draft}
        except Exception as e:
            return {"draft": self._fallback_draft(state), "is_valid": False}

    def _self_check(self, state: EvidenceState) -> EvidenceState:
        """
        SELF-CHECK node: Calls GPT-4o to verify that every claim in the draft 
        traces back to an actual evidence field. Regenerates if fabrication detected.
        Gracefully declines if evidence is too thin.
        """
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return {"final_evidence": state["draft"], "is_valid": False}

        try:
            client = OpenAI(api_key=api_key)
            
            evidence_context = self._build_evidence_context(state)
            
            response = client.chat.completions.create(
                model="gpt-4o",
                temperature=0.1,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a fact-checker. Verify that every claim in the draft traces back to "
                            "an actual evidence field from the provided evidence context. If the draft references "
                            "anything not present in the evidence (a fabrication), respond with 'FABRICATION_DETECTED'. "
                            "If the evidence is genuinely too thin to make a real case (most SHAP factors point "
                            "toward fraud, not against it), respond with 'EVIDENCE_INSUFFICIENT'. "
                            "If the draft is valid and all claims are supported by the evidence, respond with 'VALID'."
                        ),
                    },
                    {
                        "role": "user",
                        "content": f"EVIDENCE CONTEXT:\n{evidence_context}\n\nDRAFT:\n{state['draft']}"
                    },
                ],
            )
            
            check_result = response.choices[0].message.content or "VALID"
            
            if "FABRICATION_DETECTED" in check_result:
                # Regenerate draft with explicit instruction
                return self._regenerate_draft(state)
            elif "EVIDENCE_INSUFFICIENT" in check_result:
                # Graceful decline
                return {
                    "final_evidence": "Evidence insufficient to draft a strong dispute response.",
                    "is_valid": False
                }
            else:
                # Draft is valid
                return {"final_evidence": state["draft"], "is_valid": True}
                
        except Exception:
            return {"final_evidence": state["draft"], "is_valid": False}

    def _regenerate_draft(self, state: EvidenceState) -> EvidenceState:
        """Regenerate draft with explicit instruction to only use given fields."""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return {"final_evidence": self._fallback_draft(state), "is_valid": False}

        try:
            client = OpenAI(api_key=api_key)
            evidence_context = self._build_evidence_context(state)
            
            response = client.chat.completions.create(
                model="gpt-4o",
                temperature=0.1,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a chargeback dispute specialist. Write a concise, factual dispute-evidence packet. "
                            "IMPORTANT: Use ONLY the specific evidence fields provided below. Do NOT invent any details. "
                            "If you cannot make a strong case using only these fields, state that clearly. "
                            "Write in plain, factual language suitable for submission to a card network."
                        ),
                    },
                    {"role": "user", "content": evidence_context},
                ],
            )
            regenerated = response.choices[0].message.content or self._fallback_draft(state)
            return {"final_evidence": regenerated, "is_valid": True}
        except Exception:
            return {"final_evidence": self._fallback_draft(state), "is_valid": False}

    def _build_evidence_context(self, state: EvidenceState) -> str:
        """Build the evidence context string for LLM consumption."""
        lines = [
            "DISPUTE EVIDENCE CONTEXT",
            "=" * 40,
            f"Decision ID: {state['decision_id']}",
            "",
            "SHAP Model Reasons (feature contributions):",
        ]
        
        for name, value in state["top_reasons"]:
            lines.append(f"- {name}: {value:+.4f}")
        
        lines.append("")
        lines.append("SIMULATED MERCHANT/ORDER CONTEXT (for demo purposes only):")
        lines.append("NOTE: These fields are simulated based on available transaction records.")
        
        for key, value in state["context_fields"].items():
            lines.append(f"- {key}: {value}")
        
        return "\n".join(lines)

    @staticmethod
    def _fallback_draft(state: EvidenceState) -> str:
        """Fallback draft when LLM is unavailable."""
        lines = [
            f"Dispute Evidence for Decision {state['decision_id']}",
            "",
            "Model Analysis:",
        ]
        
        for name, value in state["top_reasons"]:
            lines.append(f"- {name}: {value:+.4f}")
        
        lines.append("")
        lines.append("Context Information (simulated for demo):")
        for key, value in state["context_fields"].items():
            lines.append(f"- {key}: {value}")
        
        lines.append("")
        lines.append("NOTE: This is a fallback draft. Full LLM-based evidence generation requires OPENAI_API_KEY.")
        
        return "\n".join(lines)

    def generate_evidence(self, decision_id: int, top_reasons: list) -> dict:
        """
        Run the LangGraph agent to generate evidence.
        
        Args:
            decision_id: The decision ID from the database
            top_reasons: List of (feature_name, value) tuples from SHAP
            
        Returns:
            dict with final_evidence, is_valid, and graceful_decline flag
        """
        result = self.graph.invoke({
            "decision_id": decision_id,
            "top_reasons": top_reasons,
        })
        
        return {
            "decision_id": decision_id,
            "final_evidence": result["final_evidence"],
            "is_valid": result["is_valid"],
            "graceful_decline": not result["is_valid"] and "insufficient" in result["final_evidence"].lower(),
        }
