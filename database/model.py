from datetime import datetime
from sqlalchemy import BigInteger, Boolean, DateTime, Float, ForeignKey, Integer, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

class Base(DeclarativeBase):
    pass

'''
1. You run: tx = session.get(RawTransaction, 100)
   ↳ SQLAlchemy loads the row. It sees: tx.user_id = 42

2. You type: tx.user
   ↳ Python asks SQLAlchemy: "What is .user?"
   ↳ SQLAlchemy checks its hidden setup and says:
     "Ah, .user means go fetch the UserFeatureStore row where user_id = 42!"

3. SQLAlchemy automatically runs in the background:
   SELECT * FROM user_feature_store WHERE user_id = 42;
'''


class UserFeatureStore(Base):
    __tablename__ = "user_feature_store"

    # Primary key entity identifier
    user_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    tx_count_1h: Mapped[int | None] = mapped_column(Integer, default=0)
    tx_count_24h: Mapped[int | None] = mapped_column(Integer, default=0)
    avg_amount_1h: Mapped[float | None] = mapped_column(Float, default=0.0)
    avg_amount_24h: Mapped[float | None] = mapped_column(Float, default=0.0)
    last_tx_time: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), 
        server_default=func.now(), 
        onupdate=func.now()
    )

    # ORM relationships
    transactions = relationship("RawTransaction", back_populates="user")
    audit_logs = relationship("PredictionAuditLog", back_populates="user")


class RawTransaction(Base):
    __tablename__ = "raw_transactions"

    transaction_id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("user_feature_store.user_id"), nullable=False, index=True
    )
    time: Mapped[int] = mapped_column(BigInteger, nullable=False)
    amount: Mapped[float] = mapped_column(Float, nullable=False)
    card_class: Mapped[bool] = mapped_column("class", Boolean, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    # Dynamic PCA Features (V1 to V28)
    locals().update({
        f"V{i}": mapped_column(Float, nullable=False) for i in range(1, 29)
    })

    # Link back to user_feature_store
    user = relationship("UserFeatureStore", back_populates="transactions")


class PredictionAuditLog(Base):
    __tablename__ = "prediction_audit_logs"

    log_id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("user_feature_store.user_id"), nullable=False, index=True
    )
    transaction_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("raw_transactions.transaction_id"), nullable=False
    )
    prediction: Mapped[bool] = mapped_column(Boolean, nullable=False)
    confidence_score: Mapped[float] = mapped_column(Float, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    # Link back to user_feature_store
    user = relationship("UserFeatureStore", back_populates="audit_logs")