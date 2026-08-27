const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export type ScoreRequest = {
  card_number: string;
  amount: number;
  tx_count_24h: number;
  minutes_since_last_tx: number;
  amount_vs_card_avg: number;
  transaction_time: string;
};

export type ScoreResponse = {
  transaction_id: number;
  decision_id: number;
  score: number;
  decision: "fight" | "auto_refund";
  top_reasons: [string, number][];
};

export type DecisionDetail = {
  id?: number;
  decision_id: number;
  transaction_id: number;
  score: number;
  decision: string;
  threshold_used: number;
  top_reasons: [string, number][];
  evidence_packet: string | null;
  created_at: string;
  transaction?: {
    id: number;
    card_number: string;
    amount: number;
    created_at: string;
  } | null;
};

export type AuditEntry = { event: string; detail: string; at: string };

export type Metrics = {
  features: string[];
  auc: number;
  precision: number;
  recall: number;
  threshold: number;
  fp_cost_assumption: number;
  fn_cost_assumption: number;
  estimated_total_cost: number;
  test_set_size: number;
};

export type TransactionRow = {
  transaction_id: number;
  card_number: string;
  amount: number;
  decision_id: number;
  score: number;
  decision: string;
  created_at: string;
};

async function req<T>(path: string, options?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    ...options,
    headers: { "Content-Type": "application/json", ...(options?.headers || {}) },
  });
  if (!res.ok) {
    const body = await res.text();
    throw new Error(`${res.status}: ${body}`);
  }
  return res.json();
}

export const api = {
  score: (payload: ScoreRequest) =>
    req<ScoreResponse>("/score", { method: "POST", body: JSON.stringify(payload) }),
  transactions: (limit = 20) => req<TransactionRow[]>(`/transactions?limit=${limit}`),
  decision: (id: number) =>
    req<DecisionDetail>(`/decisions/${id}`).then((detail) => ({
      ...detail,
      decision_id: detail.decision_id ?? detail.id ?? id,
    })),
  submitEvidence: (id: number, evidence_packet: string) =>
    req<{ status: string }>(`/evidence/${id}`, {
      method: "POST",
      body: JSON.stringify({ evidence_packet }),
    }),
  generateEvidence: (id: number) =>
    req<{ final_evidence: string; is_valid: boolean; graceful_decline: boolean; decision_id: number }>(`/agent/generate-evidence/${id}`, {
      method: "POST",
    }),
  audit: (id: number) => req<AuditEntry[]>(`/audit/${id}`),
  metrics: () => req<Metrics>("/metrics"),
  globalImportance: () =>
    req<{ feature_importance: [string, number][]; model_threshold: number; features: string[] }>("/model/global-importance"),
};
