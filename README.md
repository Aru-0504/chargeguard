# ChargeGuard

ChargeGuard is an AI-assisted fraud review and chargeback response workspace. It scores a transaction, explains the score, stores the resulting case, generates a dispute-evidence draft for high-risk cases, and keeps an audit trail of the work.

This repository is a working demo and development foundation. It is not yet a production payment or dispute-processing system.

## What Exists Today

### Transaction review

- A reviewer enters a card identifier, amount, transaction velocity, time since the previous transaction, amount compared with the card average, and transaction time.
- The API derives `is_odd_hour` when the transaction is before 06:00 or at/after 22:00.
- A LightGBM classifier returns a fraud probability.
- The configured threshold is `0.8553` in the checked-in model metrics. Scores at or above it become `fight`; lower scores become `auto_refund`.
- The transaction, decision, threshold, and model explanations are persisted.

### Explainability and case history

- SHAP values are calculated for every scored transaction and returned as ranked feature contributions.
- The dashboard displays the decision, score, threshold, and reason bars.
- Recent cases can be reopened from the case queue.
- Every case records audit events such as `scored` and `evidence_generated`.
- The API exposes global model feature importance and a counterfactual endpoint.

### Evidence workflow

- Evidence generation is available only for `fight` decisions.
- The LangGraph workflow assembles SHAP reasons and deterministic demo context, drafts a response, and runs a self-check.
- With `OPENAI_API_KEY`, the workflow uses GPT-4o for drafting and validation.
- Without the key, it creates a clearly marked fallback draft and reports it as invalid.
- A reviewer can edit and save the evidence packet, then export it as a PDF from the dashboard.

### Model operations

- `backend/retrain_model.py` trains a new LightGBM model, evaluates AUC/precision/recall, searches for a cost-based threshold, writes versioned model and metrics files, and registers a `ModelVersion` row.
- Retrained versions are not promoted automatically. The script asks for explicit promotion and copies the selected files into the serving paths when approved.
- The current checked-in metrics report AUC `0.9035`, precision `0.7053`, and recall `0.5877`.

## Architecture

```text
Next.js dashboard
				|
				| HTTP/JSON (NEXT_PUBLIC_API_URL)
				v
FastAPI application
	|             |                 |
	|             |                 +--> LangGraph + OpenAI evidence workflow
	|             +--> LightGBM + SHAP explanations
	+--> SQLAlchemy database (Neon PostgreSQL or local SQLite fallback)
```

### Repository layout

- `frontend/`: Next.js 16 dashboard, TypeScript API client, Tailwind/PostCSS styling, and PDF export.
- `backend/app/main.py`: FastAPI application, model loading, scoring, evidence, audit, metrics, and explainability routes.
- `backend/app/models.py`: SQLAlchemy tables for transactions, decisions, model versions, and audit logs.
- `backend/app/db.py`: environment loading, database engine, session factory, and table initialization.
- `backend/app/chargeback_model.pkl`: checked-in serving model.
- `backend/app/metrics.json`: serving feature order, threshold, and evaluation metadata.
- `backend/retrain_model.py`: manual training and model-version registration pipeline.
- `backend/app/verify_backend.py`: import, database, and endpoint smoke test.
- `agent/evidence_agent.py`: LangGraph evidence assembly, drafting, validation, and fallback behavior.
- `ml/data/`: raw and processed datasets used by the ML work.
- `ml/notebooks/data.ipynb`: exploratory/data-preparation notebook.
- `docs/`: reserved for architecture and supporting documentation.

## API

The backend runs at `http://localhost:8000` by default and exposes FastAPI documentation at `/docs`.

| Method | Route | Purpose |
| --- | --- | --- |
| `POST` | `/score` | Store and score a transaction; returns the decision, SHAP reasons, and active `model_version_id`. |
| `GET` | `/transactions?limit=20` | List recent scored transactions. |
| `GET` | `/decisions/{decision_id}` | Return a decision, transaction summary, reasons, evidence, and outcome. |
| `POST` | `/decisions/{decision_id}/outcome` | Record real-world dispute outcome (`won`, `lost`, `withdrawn`, `pending`). |
| `GET` | `/decisions/outcomes/summary` | Return dispute win rate and case outcome counts. |
| `POST` | `/evidence/{decision_id}` | Save reviewer-edited evidence for a `fight` case. |
| `POST` | `/agent/generate-evidence/{decision_id}` | Generate and save agent-produced evidence with multi-source validation. |
| `POST` | `/agent/generate-evidence-stream/{decision_id}` | Stream real-time agent reasoning and fact-checking via SSE. |
| `GET` | `/audit/{decision_id}` | Return the case audit events. |
| `GET` | `/metrics` | Return the serving model metrics and assumptions. |
| `GET` | `/model/global-importance` | Return normalized global feature importance. |
| `POST` | `/model/counterfactual` | Request scenarios intended to flip the model decision. |
| `GET` | `/model/versions` | List registered model versions and validation metrics. |
| `POST` | `/model/promote/{version_name}` | Hot-promote a trained model version into active production. |

The score request contains `card_number`, `amount`, `tx_count_24h`, `minutes_since_last_tx`, `amount_vs_card_avg`, and an ISO `transaction_time`.

## Local Setup

### Backend

From the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r backend\requirements.txt
```

Create `backend/.env` with a real database URL. `OPENAI_API_KEY` is optional for the fallback evidence path:

```dotenv
DATABASE_URL=postgresql://user:password@host/dbname?sslmode=require
OPENAI_API_KEY=your_openai_api_key_here
```

Run the API from `backend`:

```powershell
cd backend
uvicorn app.main:app --reload
```

Run the backend smoke test from `backend/app` after the database is configured:

```powershell
python verify_backend.py
```

The smoke test exercises imports, database initialization, scoring, recent transactions, decision retrieval, manual evidence saving, audit retrieval, and metrics retrieval. Its fixture expects the risky sample to produce `fight`.

### Frontend

In a second terminal:

```powershell
cd frontend
npm install
npm run dev
```

Open `http://localhost:3000`. To use another backend URL, set `NEXT_PUBLIC_API_URL` before starting Next.js. Other available commands are `npm run lint`, `npm run build`, and `npm run start`.

## Model And Data Notes

The serving feature order is:

1. `Amount`
2. `tx_count_24h`
3. `minutes_since_last_tx`
4. `amount_vs_card_avg`
5. `is_odd_hour`

The checked-in training/retraining flow supports processed CSVs, raw transaction logs, database transactions with real dispute outcome feedback, and synthetic fallbacks. When transactions have recorded dispute outcomes (`won` or `lost`), the retraining pipeline incorporates these confirmed ground-truth labels directly into model training.

## Important Limitations

- The evidence agent's merchant, order, device, account-age, shipping/billing, and IP adapters can accept real caller/webhook data or fall back to reproducible simulated enrichment when external integration credentials are not set.
- The fallback evidence packet is synthesized programmatically when OpenAI keys are unavailable or rate-limited.
- Authentication, authorization, rate limiting, and PCI DSS payment tokenization vaulting are not implemented.
- Card identifiers are stored as submitted; production handling should use tokenization or stricter redaction and access controls.
- Model versions are tracked with `model_version_id` on scored decisions, and retrained versions can be hot-promoted into production via `/model/promote/{version_name}`.
- Database initialization catches startup failures and can fall back to local SQLite when `DATABASE_URL` is missing. Configure the intended database explicitly before relying on persistence.

## Project History

The repository's implemented milestones are:

1. **Backend foundation** (`78c1ce7`): added the FastAPI classifier, SQLAlchemy persistence, Neon database support, checked-in model and metrics, evidence stub, smoke test, initial dataset/notebook structure, and environment templates. The backend endpoints were verified end to end.
2. **Credential cleanup** (`cb71d11`): removed an exposed credential from the root environment example.
3. **Agentic evidence generation** (`de8a90b`): added the LangGraph evidence workflow, GPT-4o drafting/self-checking, fallback behavior, evidence endpoints, audit logging, and required dependencies.
4. **ModelOps and explainability** (`686b4c4`, current `main`): added the retraining/version-registration pipeline, model metrics and threshold handling, SHAP explanations, global feature importance, counterfactual support, the full Next.js review dashboard, case history, editable evidence, and PDF export.

