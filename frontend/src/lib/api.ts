const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export type ScoreRequest = {
  card_number: string;
  amount: number;
  tx_count_24h: number;
  minutes_since_last_tx: number;
  amount_vs_card_avg: number;
  transaction_time: string;
  reason_code?: string;
};

export type ScoreResponse = {
  transaction_id: number;
  decision_id: number;
  score: number;
  decision: "fight" | "auto_refund";
  top_reasons: [string, number][];
  reason_code?: string;
};

export type DecisionDetail = {
  id?: number;
  decision_id: number;
  transaction_id: number;
  score: number;
  decision: string;
  threshold_used: number;
  top_reasons: [string, number][];
  reason_code?: string;
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

export type ReasonCodeOption = {
  code: string;
  name: string;
  network: "Visa" | "Mastercard";
  category: string;
  description: string;
  default_win_rate: number;
  ce3_eligible: boolean;
};

export type CounterfactualScenario = {
  amount: number;
  tx_count_24h: number;
  minutes_since_last_tx: number;
  amount_vs_card_avg: number;
  is_odd_hour: boolean;
};

export type CounterfactualRequest = {
  amount: number;
  tx_count_24h: number;
  minutes_since_last_tx: number;
  amount_vs_card_avg: number;
  is_odd_hour: boolean;
};

export type CounterfactualResponse = {
  original: CounterfactualScenario;
  counterfactuals: CounterfactualScenario[];
  explanation: string;
};

export type AgentStepEvent = {
  type: "step" | "complete";
  step?: "assemble" | "strategy" | "draft" | "self_check";
  title?: string;
  detail?: string;
  decision_id?: number;
  reason_code?: string;
  final_evidence?: string;
  is_valid?: boolean;
  graceful_decline?: boolean;
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
  generateEvidence: (id: number, reason_code?: string) =>
    req<{ final_evidence: string; is_valid: boolean; graceful_decline: boolean; decision_id: number; reason_code?: string }>(
      `/agent/generate-evidence/${id}`,
      {
        method: "POST",
        body: JSON.stringify({ reason_code }),
      }
    ),
  generateEvidenceStream: async (
    id: number,
    reason_code: string,
    onStep: (event: AgentStepEvent) => void,
    onComplete: (event: AgentStepEvent) => void,
    onError: (err: Error) => void
  ) => {
    try {
      const response = await fetch(`${API_BASE}/agent/generate-evidence-stream/${id}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ reason_code }),
      });

      if (!response.ok) {
        throw new Error(`Streaming failed: HTTP ${response.status}`);
      }

      if (!response.body) {
        throw new Error("No response body for streaming");
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n\n");
        buffer = lines.pop() || "";

        for (const block of lines) {
          const trimmed = block.trim();
          if (trimmed.startsWith("data:")) {
            try {
              const data: AgentStepEvent = JSON.parse(trimmed.replace(/^data:\s*/, ""));
              if (data.type === "step") {
                onStep(data);
              } else if (data.type === "complete") {
                onComplete(data);
              }
            } catch (err) {
              console.error("Failed to parse SSE data block", err, block);
            }
          }
        }
      }
    } catch (err) {
      onError(err instanceof Error ? err : new Error(String(err)));
    }
  },
  counterfactual: (payload: CounterfactualRequest) =>
    req<CounterfactualResponse>("/model/counterfactual", {
      method: "POST",
      body: JSON.stringify(payload),
    }),
  reasonCodes: () => req<ReasonCodeOption[]>("/agent/reason-codes"),
  audit: (id: number) => req<AuditEntry[]>(`/audit/${id}`),
  metrics: () => req<Metrics>("/metrics"),
  globalImportance: () =>
    req<{ feature_importance: [string, number][]; model_threshold: number; features: string[] }>("/model/global-importance"),
};
