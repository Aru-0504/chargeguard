"use client";

import { useEffect, useState } from "react";
import Image from "next/image";
import { ArrowUpRight, Save } from "lucide-react";
import { jsPDF } from "jspdf";
import { api, DecisionDetail, AuditEntry, TransactionRow, ScoreRequest } from "@/lib/api";

const SAMPLES: Record<string, ScoreRequest> = {
  risky: {
    card_number: "400217******1353",
    amount: 172.5,
    tx_count_24h: 5,
    minutes_since_last_tx: 1.5,
    amount_vs_card_avg: 0.4,
    transaction_time: new Date().toISOString(),
  },
  safe: {
    card_number: "500912******0044",
    amount: 42.0,
    tx_count_24h: 0,
    minutes_since_last_tx: -1,
    amount_vs_card_avg: 0.02,
    transaction_time: new Date().toISOString(),
  },
};

function Field({
  label, children,
}: { label: string; children: React.ReactNode }) {
  return (
    <label className="block">
      <span className="text-xs uppercase tracking-widest" style={{ color: "var(--ink-muted)" }}>
        {label}
      </span>
      {children}
    </label>
  );
}

function Stamp({ decision }: { decision: string }) {
  const isFight = decision === "fight";
  return (
    <div className="stamp" style={{ color: isFight ? "var(--stamp-red)" : "var(--stamp-green)" }}>
      {isFight ? "FLAGGED" : "CLEARED"}
    </div>
  );
}

function MiniStamp({ decision }: { decision: string }) {
  const isFight = decision === "fight";
  return (
    <span
      className="text-xs font-bold px-1.5 border"
      style={{
        color: isFight ? "var(--stamp-red)" : "var(--stamp-green)",
        borderColor: isFight ? "var(--stamp-red)" : "var(--stamp-green)",
      }}
    >
      {isFight ? "F" : "C"}
    </span>
  );
}

function ReasonRow({ name, value, max }: { name: string; value: number; max: number }) {
  const cols = 24;
  const filled = Math.max(0, Math.round((Math.abs(value) / max) * cols));
  const bar = "█".repeat(filled) + "░".repeat(cols - filled);
  return (
    <div className="flex items-baseline gap-3 text-sm">
      <span className="w-44 truncate" style={{ color: "var(--ink-muted)" }}>{name}</span>
      <span style={{ color: value >= 0 ? "var(--amber)" : "var(--ink-dim)", letterSpacing: "-1px" }}>
        {bar}
      </span>
      <span className="w-14 text-right" style={{ color: "var(--ink-muted)" }}>{value.toFixed(3)}</span>
    </div>
  );
}

function explainReason(name: string, value: number) {
  const direction = value >= 0 ? "increased" : "reduced";
  const descriptions: Record<string, string> = {
    Amount: "transaction amount compared with the learned fraud patterns",
    tx_count_24h: "number of transactions seen in the prior 24 hours",
    minutes_since_last_tx: "time since the card's previous transaction",
    amount_vs_card_avg: "amount compared with this card's typical spend",
    is_odd_hour: "activity during an unusual hour",
  };
  return `${descriptions[name] || name} ${direction} the fraud score`;
}

function CaseLog({ entries }: { entries: AuditEntry[] }) {
  if (entries.length === 0) {
    return <p className="text-[13px]" style={{ color: "var(--ink-dim)" }}>no log entries.</p>;
  }
  return (
    <div className="space-y-1.5">
      {entries.map((e, i) => (
        <div key={i} className="flex items-baseline text-sm">
          <span style={{ color: "var(--ink-dim)" }}>{new Date(e.at).toLocaleTimeString()}</span>
          <span className="dotted-leader" />
          <span style={{ color: "var(--amber)" }}>{e.event}</span>
          <span className="ml-3 truncate" style={{ color: "var(--ink-muted)" }}>{e.detail}</span>
        </div>
      ))}
    </div>
  );
}

export default function Dashboard() {
  const [txns, setTxns] = useState<TransactionRow[]>([]);
  const [form, setForm] = useState<ScoreRequest>(SAMPLES.risky);
  const [selectedSample, setSelectedSample] = useState<"risky" | "safe">("risky");
  const [status, setStatus] = useState<"idle" | "scoring" | "result">("idle");
  const [decision, setDecision] = useState<DecisionDetail | null>(null);
  const [audit, setAudit] = useState<AuditEntry[]>([]);
  const [evidenceDraft, setEvidenceDraft] = useState("");
  const [evidenceStatus, setEvidenceStatus] = useState<"idle" | "generating" | "ready">("idle");
  const [error, setError] = useState<string | null>(null);
  const [globalImportance, setGlobalImportance] = useState<[string, number][] | null>(null);

  const refreshTxns = () => api.transactions(10).then(setTxns).catch(() => {});

  useEffect(() => {
    refreshTxns();
    api.globalImportance().then(data => setGlobalImportance(data.feature_importance)).catch(() => {});
  }, []);

  async function loadDecision(decisionId: number) {
    setStatus("scoring");
    setError(null);
    try {
      const [d, a] = await Promise.all([api.decision(decisionId), api.audit(decisionId)]);
      setDecision(d);
      setAudit(a);
      setEvidenceDraft(d.evidence_packet || "");
      setStatus("result");
    } catch (e) {
      setError(String(e));
      setStatus("idle");
    }
  }

  async function handleScore(e: React.FormEvent) {
    e.preventDefault();
    setStatus("scoring");
    setError(null);
    try {
      const res = await api.score({ ...form, transaction_time: new Date().toISOString() });
      await loadDecision(res.decision_id);
      refreshTxns();
    } catch (err) {
      setError(String(err));
      setStatus("idle");
    }
  }

  async function handleSubmitEvidence() {
    if (!decision) return;
    try {
      await api.submitEvidence(decision.decision_id, evidenceDraft);
      downloadEvidencePdf();
      await loadDecision(decision.decision_id);
    } catch (e) {
      setError(String(e));
    }
  }

  function downloadEvidencePdf() {
    if (!decision || !evidenceDraft.trim()) return;

    const doc = new jsPDF();
    const pageWidth = doc.internal.pageSize.getWidth();
    const pageHeight = doc.internal.pageSize.getHeight();
    const margin = 18;
    const textWidth = pageWidth - margin * 2;
    let cursorY = 22;

    const addText = (text: string, size = 10, bold = false, gap = 6) => {
      doc.setFont("helvetica", bold ? "bold" : "normal");
      doc.setFontSize(size);
      const safeText = text.replace(/[^\x00-\x7F]/g, (character) => character === "₹" ? "INR " : "?");
      const lines = doc.splitTextToSize(safeText, textWidth);
      if (cursorY + lines.length * 5 + gap > pageHeight - margin) {
        doc.addPage();
        cursorY = margin;
      }
      doc.text(lines, margin, cursorY);
      cursorY += lines.length * 5 + gap;
    };

    addText("CHARGEGUARD", 18, true, 7);
    addText("EVIDENCE PACKET - DRAFT", 11, true, 10);
    addText(`Case ${String(decision.decision_id).padStart(6, "0")}`, 10, true);
    addText(`Transaction: ${decision.transaction_id}`);
    if (decision.transaction) {
      addText(`Card: ${decision.transaction.card_number}`);
      addText(`Amount: INR ${decision.transaction.amount.toFixed(2)}`);
    }
    addText(`Recommendation: ${decision.decision === "fight" ? "Review and fight this dispute" : "Auto-refund this dispute"}`);
    addText(`Fraud score: ${(decision.score * 100).toFixed(1)}%`);
    addText(`Review threshold: ${(decision.threshold_used * 100).toFixed(1)}%`, 10, false, 10);

    addText("REVIEWED EVIDENCE", 11, true, 7);
    addText(evidenceDraft, 10, false, 10);
    addText(`Exported: ${new Date().toLocaleString()}`, 9, false, 0);
    doc.save(`chargeguard-case-${decision.decision_id}-evidence.pdf`);
  }

  async function handleGenerateEvidence() {
    if (!decision) return;
    setEvidenceStatus("generating");
    setError(null);
    try {
      const result = await api.generateEvidence(decision.decision_id);
      setEvidenceDraft(result.final_evidence);
      setEvidenceStatus("ready");
      await loadDecision(decision.decision_id);
    } catch (e) {
      setError(String(e));
      setEvidenceStatus("idle");
    }
  }

  const maxAbsReason = decision
    ? Math.max(...decision.top_reasons.map(([, v]) => Math.abs(v)), 0.001)
    : 1;

  return (
    <main className="dashboard-shell max-w-7xl mx-auto px-5 py-6 sm:px-8 sm:py-8">
      {/* Masthead */}
      <div className="flex items-end justify-between pb-6 rule-bottom">
        <div>
          <Image className="brand-logo" src="/brand/logo.png" alt="ChargeGuard" width={560} height={160} priority />
          <p className="text-base mt-1" style={{ color: "var(--ink-muted)" }}>
            risk case file — chargeback dispute record
          </p>
        </div>
        <span className="status-pill text-xs" style={{ color: "var(--mint)" }}>
          <span className="status-dot" /> model v1 · live
        </span>
      </div>

      <div className="dashboard-grid">
        <div className="dashboard-main">
      {/* Intake form */}
      <section className="section-card intake-card">
        <div className="section-heading">
          <div>
            <p className="eyebrow">NEW REVIEW</p>
            <h2>New case intake</h2>
          </div>
            <span className="section-index">INPUT</span>
        </div>
        <div className="flex gap-3 mb-7 text-sm">
          <button
            type="button"
            onClick={() => { setForm(SAMPLES.risky); setSelectedSample("risky"); }}
            className={`sample-button ${selectedSample === "risky" ? "is-selected" : ""}`}
          >
            risky sample
          </button>
          <button
            type="button"
            onClick={() => { setForm(SAMPLES.safe); setSelectedSample("safe"); }}
            className={`sample-button ${selectedSample === "safe" ? "is-selected" : ""}`}
          >
            safe sample
          </button>
        </div>
        <form onSubmit={handleScore} className="space-y-6">
          <Field label="Card number">
            <input
              className="field-input"
              value={form.card_number}
              onChange={(e) => setForm({ ...form, card_number: e.target.value })}
            />
          </Field>
          <div className="grid grid-cols-2 gap-x-8 gap-y-5">
            <Field label="Amount (₹)">
              <input
                type="number" step="0.01"
                className="field-input"
                value={form.amount}
                onChange={(e) => setForm({ ...form, amount: parseFloat(e.target.value) })}
              />
            </Field>
            <Field label="Tx count, prior 24h">
              <input
                type="number"
                className="field-input"
                value={form.tx_count_24h}
                onChange={(e) => setForm({ ...form, tx_count_24h: parseInt(e.target.value) })}
              />
            </Field>
            <Field label="Minutes since last tx">
              <input
                type="number" step="0.1"
                className="field-input"
                value={form.minutes_since_last_tx}
                onChange={(e) => setForm({ ...form, minutes_since_last_tx: parseFloat(e.target.value) })}
              />
            </Field>
            <Field label="Amount vs card average">
              <input
                type="number" step="0.01"
                className="field-input"
                value={form.amount_vs_card_avg}
                onChange={(e) => setForm({ ...form, amount_vs_card_avg: parseFloat(e.target.value) })}
              />
            </Field>
          </div>
          <button
            type="submit"
            disabled={status === "scoring"}
            className="primary-action px-5 py-3 text-sm uppercase tracking-widest border-2"
            style={{ borderColor: "var(--amber)", color: "var(--amber)" }}
          >
            {status === "scoring" ? "processing…" : "file case for scoring"}
          </button>
          {error && <p className="text-[12px]" style={{ color: "var(--stamp-red)" }}>{error}</p>}
        </form>
      </section>

      {/* Findings */}
      {status === "result" && decision && (
        <section className="relative section-card findings-card">
          <Stamp decision={decision.decision} />
          <div className="section-heading">
            <div><p className="eyebrow">DECISION OUTPUT</p><h2>Case findings</h2></div>
          </div>
          <p className="text-sm mb-6" style={{ color: "var(--ink-dim)" }}>
            case no. {String(decision.decision_id).padStart(6, "0")} · score {decision.score.toFixed(4)} · threshold {decision.threshold_used}
          </p>

          <div className={`decision-summary ${decision.decision === "fight" ? "is-risk" : "is-clear"}`}>
            <div>
              <p className="eyebrow">MODEL RECOMMENDATION</p>
              <h3>{decision.decision === "fight" ? "Review and fight this dispute" : "Auto-refund this dispute"}</h3>
              <p>{decision.decision === "fight" ? "The model score is at or above the configured review threshold." : "The model score is below the configured review threshold."}</p>
            </div>
            <div className="score-readout"><strong>{(decision.score * 100).toFixed(1)}%</strong><span>fraud score</span></div>
            <div className="score-meter"><span style={{ width: `${Math.min(decision.score * 100, 100)}%` }} /><i style={{ left: `${Math.min(decision.threshold_used * 100, 100)}%` }} /></div>
            <div className="meter-labels"><span>0%</span><span>threshold {(decision.threshold_used * 100).toFixed(1)}%</span><span>100%</span></div>
          </div>

          <div className="mb-6">
            <h3 className="subheading mb-3">
              Evidence factors
            </h3>
            <div className="space-y-1.5">
              {decision.top_reasons.map(([name, value]) => <div key={name} className="reason-block"><ReasonRow name={name} value={value} max={maxAbsReason} /><p>{explainReason(name, value)}</p></div>)}
            </div>
          </div>

          {globalImportance && (
            <div className="mb-6">
              <h3 className="subheading mb-3">
                Global feature importance
              </h3>
              <p className="text-sm mb-3" style={{ color: "var(--ink-dim)" }}>
                Overall feature influence across the model's training data
              </p>
              <div className="space-y-1.5">
                {globalImportance.map(([name, value]) => (
                  <div key={name} className="flex items-baseline gap-3 text-sm">
                    <span className="w-44 truncate" style={{ color: "var(--ink-muted)" }}>{name}</span>
                    <div className="flex-1 bg-gray-200 rounded h-2">
                      <div 
                        className="bg-blue-500 h-2 rounded" 
                        style={{ width: `${value}%` }}
                      />
                    </div>
                    <span className="w-14 text-right" style={{ color: "var(--ink-muted)" }}>{value.toFixed(1)}%</span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {decision.decision === "fight" && (
            <div className="mb-6">
              <div className="evidence-heading"><h3 className="subheading">Evidence packet</h3><span className="evidence-status">{evidenceStatus === "ready" || evidenceDraft ? "DRAFT READY" : "NOT GENERATED"}</span></div>
              <p className="supporting-copy">Generate a structured draft from this transaction and its model factors. Review it, then edit or save it against the case.</p>
              <button
                type="button"
                onClick={handleGenerateEvidence}
                disabled={evidenceStatus === "generating"}
                className="secondary-action mb-3 text-sm"
              >
                {evidenceStatus === "generating" ? "generating draft..." : "generate evidence draft"}
              </button>
              <textarea
                className="w-full px-3 py-2 text-sm paper-lines border"
                style={{ borderColor: "var(--rule)", color: "var(--ink)", minHeight: "104px" }}
                value={evidenceDraft}
                onChange={(e) => setEvidenceDraft(e.target.value)}
                placeholder="Generate a draft or write the evidence packet here."
              />
              <button
                type="button"
                onClick={handleSubmitEvidence}
                className="secondary-action mt-3 inline-flex items-center gap-2 text-sm uppercase tracking-widest"
              >
                <Save size={15} aria-hidden="true" />
                save evidence packet
              </button>
            </div>
          )}

          <div>
            <h3 className="subheading mb-3">
              Case log
            </h3>
            <CaseLog entries={audit} />
          </div>
        </section>
      )}

      {status === "idle" && (
        <p className="empty-state" style={{ color: "var(--ink-dim)" }}>
          file a case above, or select one from the index below, to view findings.
        </p>
      )}

      {/* Case index */}
      <section className="section-card index-card">
        <div className="section-heading"><div><p className="eyebrow">RECENT REVIEWS</p><h2>Case queue</h2></div><span className="section-index">QUEUE</span></div>
        <div className="queue-head"><span>DECISION</span><span>CARD</span><span>AMOUNT</span><span>SCORE</span></div>
        <div className="space-y-0">
          {txns.map((t) => (
            <div
              key={t.transaction_id}
              onClick={() => loadDecision(t.decision_id)}
              className="flex items-center gap-4 py-3 rule-bottom cursor-pointer text-sm hover:opacity-80"
            >
              <MiniStamp decision={t.decision} />
              <span className="w-40" style={{ color: "var(--ink-muted)" }}>{t.card_number}</span>
              <span className="w-20" style={{ color: "var(--ink-muted)" }}>₹{t.amount}</span>
              <span className="flex-1" style={{ color: "var(--ink-dim)" }}>score {t.score.toFixed(3)}</span>
              <ArrowUpRight className="queue-arrow" size={16} />
            </div>
          ))}
          {txns.length === 0 && (
            <p className="py-6 text-sm" style={{ color: "var(--ink-dim)" }}>
              no cases on file yet — file one above.
            </p>
          )}
        </div>
      </section>
        </div>
      </div>
    </main>
  );
}
