from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, Text
from sqlalchemy.orm import declarative_base
from datetime import datetime

Base = declarative_base()

class Transaction(Base):
    __tablename__ = "transactions"
    id = Column(Integer, primary_key=True, index=True)
    card_number = Column(String, index=True)
    amount = Column(Float)
    tx_count_24h = Column(Integer)
    minutes_since_last_tx = Column(Float)
    amount_vs_card_avg = Column(Float)
    is_odd_hour = Column(Boolean)
    created_at = Column(DateTime, default=datetime.utcnow)

class Decision(Base):
    __tablename__ = "decisions"
    id = Column(Integer, primary_key=True, index=True)
    transaction_id = Column(Integer, index=True)
    score = Column(Float)          # model's fraud probability
    decision = Column(String)      # "fight" or "auto_refund"
    threshold_used = Column(Float)
    top_reasons = Column(Text)     # SHAP-driven reasons, JSON string
    evidence_packet = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

class AuditLog(Base):
    __tablename__ = "audit_log"
    id = Column(Integer, primary_key=True, index=True)
    decision_id = Column(Integer, index=True)
    event = Column(String)         # "scored", "evidence_generated", "outcome_recorded"
    detail = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
