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
  model_version_id?: number | null;
  model_version_name?: string;
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
  model_version_id?: number | null;
  dispute_outcome?: string | null;
  dispute_outcome_at?: string | null;
  outcome_notes?: string | null;
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

export type ModelVersionItem = {
  id: number;
  version_name: string;
  threshold: number;
  auc: number;
  precision: number;
  recall: number;
  training_date: string | null;
  is_active: boolean;
  created_at: string | null;
  model_file: string | null;
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

export type ValidationReport = {
  ce3_eligible?: boolean;
  ce3_score?: number;
  matched_elements?: string[];
  liability_shift_secured?: boolean;
  eci_code?: string;
  pod_verified?: boolean;
  tracking_number?: string;
  fact_audit_passed?: boolean;
  win_probability?: number;
  audit_notes?: string[];
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
  validation_report?: ValidationReport;
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
  modelVersions: () => req<ModelVersionItem[]>("/model/versions"),
  promoteModelVersion: (versionName: string) =>
    req<{ status: string; message: string; active_version: string; metrics: Metrics }>(
      `/model/promote/${versionName}`,
      { method: "POST" }
    ),
  globalImportance: () =>
    req<{ feature_importance: [string, number][]; model_threshold: number; features: string[] }>("/model/global-importance"),
  recordOutcome: (decisionId: number, outcome: "won" | "lost" | "withdrawn" | "pending", notes?: string) =>
    req<{ status: string; decision_id: number; dispute_outcome: string; dispute_outcome_at: string; notes: string | null }>(
      `/decisions/${decisionId}/outcome`,
      {
        method: "POST",
        body: JSON.stringify({ outcome, notes }),
      }
    ),
  outcomesSummary: () =>
    req<{ total_cases: number; won: number; lost: number; withdrawn: number; pending: number; resolved: number; win_rate: number }>(
      "/decisions/outcomes/summary"
    ),
};
