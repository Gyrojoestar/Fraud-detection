import os
import urllib.parse
from datetime import datetime
from typing import List, Optional
from dotenv import load_dotenv
from sqlalchemy import (
    create_engine,
    text,
    BigInteger,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    func,
)
from sqlalchemy.orm import (
    sessionmaker,
    DeclarativeBase,
    Mapped,
    mapped_column,
    relationship,
)
from sqlalchemy.pool import NullPool

# Load environment variables
load_dotenv()

raw_password = os.getenv("DATABASE_PW")
if not raw_password:
    raise ValueError("DATABASE_PW environment variable is missing!")

encoded_password = urllib.parse.quote_plus(raw_password)

DATABASE_URL = f"postgresql://postgres.avmyohbogdlamvppmesm:{encoded_password}@aws-0-ap-northeast-1.pooler.supabase.com:6543/postgres"

engine = create_engine(DATABASE_URL, poolclass=NullPool, echo=False)
SessionLocal = sessionmaker(bind=engine, autoflush=False)


class Base(DeclarativeBase):
    pass


class RawTransaction(Base):
    __tablename__ = "raw_transactions"

    transaction_id: Mapped[int] = mapped_column(
        BigInteger(), primary_key=True, autoincrement=True
    )
    amount: Mapped[float] = mapped_column(Float, nullable=False)
    card_class: Mapped[bool] = mapped_column("class", Boolean, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    # Dynamic PCA Features (v1 to v28)
    locals().update(
        {f"v{i}": mapped_column(Float, nullable=False) for i in range(1, 29)}
    )

    model_pred: Mapped[List["ModelPred"]] = relationship(
        back_populates="raw_trans"
    )


class ModelPred(Base):
    __tablename__ = "model_predictions"

    prediction_id: Mapped[int] = mapped_column(
        BigInteger(), primary_key=True, autoincrement=True
    )
    transaction_id: Mapped[int] = mapped_column(
        BigInteger(),
        ForeignKey("raw_transactions.transaction_id"),
        nullable=False,
    )
    pred_class: Mapped[bool] = mapped_column("class", Boolean, nullable=False)
    confidence_score: Mapped[float] = mapped_column(Float, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    # Point to a single RawTransaction instance, not a List
    raw_trans: Mapped["RawTransaction"] = relationship(
        back_populates="model_pred"
    )