# ChargeGuard Architecture Diagram

```mermaid
graph TB
    subgraph "Frontend Layer"
        Dashboard[Next.js Dashboard]
        API_Client[API Client]
    end

    subgraph "Backend Layer"
        FastAPI[FastAPI Server]
        
        subgraph "API Endpoints"
            Score[POST /score]
            Evidence[POST /agent/generate-evidence]
            GlobalImportance[GET /model/global-importance]
            Counterfactual[POST /model/counterfactual]
            Metrics[GET /metrics]
        end
        
        subgraph "ML Components"
            Model[LightGBM Model]
            SHAP[SHAP Explainer]
            EvidenceAgent[LangGraph Evidence Agent]
            DiCE[DiCE Counterfactual]
        end
        
        subgraph "ModelOps"
            Retrain[Retrain Script]
            ModelVersion[Model Versioning]
        end
    end

    subgraph "Database Layer"
        PostgreSQL[(PostgreSQL/SQLite)]
        Tables[Transactions, Decisions, AuditLog, ModelVersions]
    end

    subgraph "External Services"
        OpenAI[OpenAI GPT-4]
    end

    Dashboard --> API_Client
    API_Client --> FastAPI
    FastAPI --> Score
    FastAPI --> Evidence
    FastAPI --> GlobalImportance
    FastAPI --> Counterfactual
    FastAPI --> Metrics
    
    Score --> Model
    Score --> SHAP
    Score --> PostgreSQL
    
    Evidence --> EvidenceAgent
    EvidenceAgent --> OpenAI
    EvidenceAgent --> PostgreSQL
    
    GlobalImportance --> SHAP
    GlobalImportance --> Model
    
    Counterfactual --> DiCE
    Counterfactual --> Model
    
    Metrics --> PostgreSQL
    
    Retrain --> ModelVersion
    Retrain --> Model
    ModelVersion --> PostgreSQL
    
    PostgreSQL --> Tables
    
    style Dashboard fill:#4CAF50,color:#fff
    style FastAPI fill:#2196F3,color:#fff
    style PostgreSQL fill:#FF9800,color:#fff
    style Model fill:#9C27B0,color:#fff
    style EvidenceAgent fill:#E91E63,color:#fff
    style OpenAI fill:#00BCD4,color:#fff
```

## Component Descriptions

### Frontend Layer
- **Next.js Dashboard**: React-based UI for monitoring transactions, viewing evidence, and managing disputes
- **API Client**: TypeScript client for backend communication

### Backend Layer
- **FastAPI Server**: Python REST API with async support
- **API Endpoints**:
  - `/score`: Real-time fraud scoring with SHAP explanations
  - `/agent/generate-evidence`: AI-powered evidence generation
  - `/model/global-importance`: Global feature importance visualization
  - `/model/counterfactual`: Counterfactual explanations (what-if analysis)
  - `/metrics`: Model performance metrics

### ML Components
- **LightGBM Model**: Gradient boosting model for fraud detection
- **SHAP Explainer**: Model explainability using SHAP values
- **LangGraph Evidence Agent**: 3-node workflow (ASSEMBLE → DRAFT → SELF-CHECK) using GPT-4
- **DiCE**: Counterfactual explanation generation

### ModelOps
- **Retrain Script**: Manual retraining pipeline with evaluation
- **Model Versioning**: Database-backed model tracking and rollback

### Database Layer
- **PostgreSQL/SQLite**: Transactional database with fallback support
- **Tables**: Transactions, Decisions, AuditLog, ModelVersions

### External Services
- **OpenAI GPT-4**: LLM for evidence generation and validation

## Data Flow

1. **Transaction Scoring**: Dashboard → API → Model + SHAP → Database
2. **Evidence Generation**: Dashboard → API → Evidence Agent → OpenAI → Database
3. **Model Monitoring**: Dashboard → API → SHAP → Global Importance
4. **Model Retraining**: Retrain Script → New Model → Model Version → Database
