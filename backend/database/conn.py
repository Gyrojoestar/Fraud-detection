import os
import urllib.parse
from dotenv import load_dotenv
from sqlalchemy import (
    create_engine,
    BigInteger,
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    func,
)
from sqlalchemy.orm import (
    sessionmaker,
    declarative_base,
    relationship,
)
from sqlalchemy.pool import NullPool

load_dotenv()

# connect to supabase database using environment variables
raw_password = os.getenv("DATABASE_PW")
if not raw_password:
    raise ValueError("Neither DATABASE_URL nor DATABASE_PW environment variables are set!")
encoded_password = urllib.parse.quote_plus(raw_password)
DATABASE_URL = f"postgresql://postgres.avmyohbogdlamvppmesm:{encoded_password}@aws-0-ap-northeast-1.pooler.supabase.com:6543/postgres"

# create new SQLAAlchemy engine and sessionmaker to connect to the database
engine = create_engine(DATABASE_URL, poolclass=NullPool, echo=False)
SessionLocal = sessionmaker(bind=engine, autoflush=False)


Base = declarative_base()


# define SQLAlchemy ORM models for all database tables
class RawTransaction(Base):
    __tablename__ = "raw_transactions"

    transaction_id = Column(
        BigInteger(), primary_key=True, autoincrement=True
    )
    amount = Column(Float, nullable=False)
    card_class = Column("class", Boolean, nullable=False)
    created_at = Column(
        DateTime(timezone=True), server_default=func.now()
    )

    locals().update(
        {f"v{i}": Column(Float, nullable=False) for i in range(1, 29)}
    )

    model_pred = relationship("ModelPred", back_populates="raw_trans")


class ModelPred(Base):
    __tablename__ = "model_predictions"

    prediction_id = Column(
        BigInteger(), primary_key=True, autoincrement=True
    )
    transaction_id = Column(
        BigInteger(),
        ForeignKey("raw_transactions.transaction_id"),
        nullable=False,
    )
    pred_class = Column("class", Boolean, nullable=False)
    confidence_score = Column(Float, nullable=False)
    created_at = Column(
        DateTime(timezone=True), server_default=func.now()
    )

    raw_trans = relationship("RawTransaction", back_populates="model_pred")