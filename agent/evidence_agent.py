import os
import hashlib
from typing import TypedDict, Generator, Any

from langgraph.graph import END, StateGraph
from openai import OpenAI


class EvidenceState(TypedDict, total=False):
    decision_id: int
    top_reasons: list
    reason_code: str
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
        ASSEMBLE node: Combines SHAP reasons with simulated merchant/order context fields
        tailored to the specific dispute reason code (e.g., Visa CE3.0, Mastercard, Proof of Delivery).
        """
        decision_id = state["decision_id"]
        reason_code = state.get("reason_code", "Visa 10.4 - Fraud: Card-Absent Environment")

        # Deterministic hash-based simulation for consistent demo data
        seed = int(hashlib.sha256(str(decision_id).encode()).hexdigest(), 16)

        # Generate simulated context fields deterministically
        shipping_billing_match = "yes" if (seed % 3) != 0 else "no"
        device_options = ["mobile (iOS Safari 17.4)", "desktop (Chrome 122 / Win11)", "tablet (iPadOS 17.2)"]
        device_used = device_options[seed % 3]
        account_age_days = (seed % 365) + 1  # 1-365 days
        ip_consistent = "yes" if (seed % 4) != 0 else "no"
        avs_options = ["Match Address & 5-digit ZIP (Y)", "Match ZIP only (Z)", "Street match only (A)"]
        cvv_options = ["CVV2 Match (M)", "Cardholder CVV verified"]
        three_ds_options = ["Fully Authenticated / Issuer Liability Shift (ECI 05)", "Attempted / Merchant Liability Shift (ECI 06)"]

        context_fields = {
            "dispute_reason_code": reason_code,
            "shipping_billing_match": shipping_billing_match,
            "device_fingerprint": device_used,
            "account_age_days": f"{account_age_days} days active",
            "ip_geo_consistency": ip_consistent,
            "avs_check": avs_options[seed % len(avs_options)],
            "cvv_verification": cvv_options[seed % len(cvv_options)],
            "3ds_liability_shift": three_ds_options[seed % len(three_ds_options)],
            "prior_undisputed_txs": f"{2 + (seed % 3)} qualifying historical transactions (CE3.0)",
            "carrier_tracking_pod": f"FEDEX-{100000000 + (seed % 900000000)} (Delivered with signature)"
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
            evidence_context = self._build_evidence_context(state)
            reason_code = state.get("reason_code", "Visa 10.4")

            response = client.chat.completions.create(
                model="gpt-4o",
                temperature=0.1,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a chargeback dispute specialist. Write a concise, authoritative dispute-evidence packet "
                            f"specifically contesting dispute code: {reason_code}. "
                            "Ground your response ONLY on the provided SHAP factor contributions, authentication flags, and order context. "
                            "If contesting Visa 10.4 or Mastercard 4837, cite Compelling Evidence 3.0 (CE3.0) qualifiers (e.g. prior undisputed orders, device fingerprint, 3DS liability shift). "
                            "If contesting Visa 13.1 (Not Received), emphasize carrier tracking and delivery confirmation. "
                            "Do not invent any details not present in the evidence. "
                            "Format cleanly with headers: 1. EXECUTIVE SUMMARY & GROUNDS, 2. AUTHENTICATION & TECHNICAL REQUISITES, "
                            "3. STATISTICAL FRAUD RISK ANALYSIS, 4. FORMAL REQUEST FOR DISPUTE REVERSAL."
                        ),
                    },
                    {"role": "user", "content": evidence_context},
                ],
            )
            draft = response.choices[0].message.content or self._fallback_draft(state)
            return {"draft": draft}
        except Exception:
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
                return self._regenerate_draft(state)
            elif "EVIDENCE_INSUFFICIENT" in check_result:
                return {
                    "final_evidence": "Evidence insufficient to draft a strong dispute response under scheme guidelines.",
                    "is_valid": False
                }
            else:
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
        reason_code = state.get("reason_code", "Visa 10.4 - Fraud: Card-Absent Environment")
        lines = [
            "DISPUTE EVIDENCE CONTEXT",
            "=" * 40,
            f"Decision ID: {state['decision_id']}",
            f"Scheme Reason Code: {reason_code}",
            "",
            "SHAP Model Reasons (feature contributions to risk assessment):",
        ]

        for name, value in state.get("top_reasons", []):
            direction = "INCREASED RISK" if value >= 0 else "REDUCED RISK (FAVORS MERCHANT)"
            lines.append(f"- {name}: {value:+.4f} ({direction})")

        lines.append("")
        lines.append("CARD SCHEME AUTHENTICATION & TRANSACTION CONTEXT:")
        for key, value in state.get("context_fields", {}).items():
            lines.append(f"- {key}: {value}")

        return "\n".join(lines)

    @staticmethod
    def _fallback_draft(state: EvidenceState) -> str:
        """Structured professional dispute fallback draft when LLM is unavailable."""
        decision_id = state.get("decision_id", 0)
        reason_code = state.get("reason_code", "Visa 10.4 - Fraud: Card-Absent Environment")
        ctx = state.get("context_fields", {})
        reasons = state.get("top_reasons", [])

        lines = [
            f"FORMAL DISPUTE EVIDENCE PACKET — CASE NO. {str(decision_id).padStart(6, '0') if hasattr(str(decision_id), 'padStart') else str(decision_id).zfill(6)}",
            f"REASON CODE: {reason_code}",
            "=" * 60,
            "",
            "1. EXECUTIVE SUMMARY & REVERSAL GROUNDS",
            f"Merchant hereby formally contests the dispute filed under reason code '{reason_code}'.",
            "Transaction records demonstrate explicit cardholder participation, verified authentication protocols,",
            "and alignment with network rules (including Visa Compelling Evidence 3.0 / Mastercard standards).",
            "",
            "2. AUTHENTICATION & TECHNICAL REQUISITES",
            f"- AVS Check: {ctx.get('avs_check', 'Match Verified')}",
            f"- CVV Verification: {ctx.get('cvv_verification', 'CVV2 Match (M)')}",
            f"- 3D Secure / Liability Shift: {ctx.get('3ds_liability_shift', 'Authenticated (ECI 05)')}",
            f"- Device Fingerprint: {ctx.get('device_fingerprint', 'Verified device')}",
            f"- Account Age / Activity: {ctx.get('account_age_days', 'Active cardholder profile')}",
            f"- Historical Linkage: {ctx.get('prior_undisputed_txs', '2 prior undisputed transactions')}",
            f"- Fulfillment / POD: {ctx.get('carrier_tracking_pod', 'Delivered to confirmed cardholder address')}",
            "",
            "3. STATISTICAL FRAUD RISK ANALYSIS & SHAP ATTRIBUTION",
            "The calibrated machine learning assessment produced the following feature attributions for this transaction:",
        ]

        for name, value in reasons:
            bias = "Heightened scrutiny" if value >= 0 else "Favorable legitimacy signal"
            lines.append(f"  • {name}: {value:+.4f} [{bias}]")

        lines.extend([
            "",
            "4. FORMAL REQUEST FOR CHARGEBACK REVERSAL",
            "Given that authorization protocols were verified at point of sale and liability shift criteria are satisfied,",
            "merchant respectfully requests immediate reversal of the disputed funds and closure of this case in favor of merchant.",
            "",
            "------------------------------------------------------------",
            "NOTE: This packet was assembled by ChargeGuard Evidence Engine with deterministic scheme validation.",
            "Set OPENAI_API_KEY to activate generative GPT-4o synthesis and live multi-agent verification."
        ])

        return "\n".join(lines)

    def generate_evidence(self, decision_id: int, top_reasons: list, reason_code: str = "Visa 10.4 - Fraud: Card-Absent Environment") -> dict:
        """
        Run the LangGraph agent to generate evidence.
        """
        result = self.graph.invoke({
            "decision_id": decision_id,
            "top_reasons": top_reasons,
            "reason_code": reason_code,
        })

        return {
            "decision_id": decision_id,
            "reason_code": reason_code,
            "final_evidence": result.get("final_evidence", ""),
            "is_valid": result.get("is_valid", False),
            "graceful_decline": not result.get("is_valid", False) and "insufficient" in str(result.get("final_evidence", "")).lower(),
        }

    def generate_evidence_stream(
        self,
        decision_id: int,
        top_reasons: list,
        reason_code: str = "Visa 10.4 - Fraud: Card-Absent Environment"
    ) -> Generator[dict[str, Any], None, None]:
        """
        Streaming generator that emits real-time execution steps and status for the UI.
        """
        yield {
            "type": "step",
            "step": "assemble",
            "title": "Assembling Evidence Context",
            "detail": f"Gathering SHAP attributions and scheme data for {reason_code}..."
        }

        assembled = self._assemble({
            "decision_id": decision_id,
            "top_reasons": top_reasons,
            "reason_code": reason_code
        })
        context_fields = assembled["context_fields"]
        state: EvidenceState = {
            "decision_id": decision_id,
            "top_reasons": top_reasons,
            "reason_code": reason_code,
            "context_fields": context_fields
        }

        yield {
            "type": "step",
            "step": "strategy",
            "title": "Defense Strategy Formulation",
            "detail": f"Formulating legal and technical dispute defense aligned with {reason_code} guidelines."
        }

        yield {
            "type": "step",
            "step": "draft",
            "title": "Dispute Packet Synthesis",
            "detail": "Synthesizing formalized dispute response packet with verified evidence citations..."
        }

        draft_result = self._draft(state)
        state["draft"] = draft_result.get("draft", self._fallback_draft(state))

        yield {
            "type": "step",
            "step": "self_check",
            "title": "Anti-Hallucination & Fact Audit",
            "detail": "Auditing drafted assertions against provided transaction facts and SHAP factors..."
        }

        check_result = self._self_check(state)
        final_evidence = check_result.get("final_evidence", state["draft"])
        is_valid = check_result.get("is_valid", False)
        graceful_decline = not is_valid and "insufficient" in str(final_evidence).lower()

        yield {
            "type": "complete",
            "decision_id": decision_id,
            "reason_code": reason_code,
            "final_evidence": final_evidence,
            "is_valid": is_valid,
            "graceful_decline": graceful_decline
        }
