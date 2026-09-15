from sqlalchemy import create_engine, text, select, Integer, BigInteger, Double, Boolean, Float
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, Session
from sqlalchemy.pool import NullPool
from sqlalchemy import select
import urllib.parse
import os
from dotenv import load_dotenv

load_dotenv()
raw_password = os.getenv("DATABASE_PW")
encoded_password = urllib.parse.quote_plus(raw_password)

# Get connection string from supabase
# Format: postgresql+psycopg2://user:password@host:port/dbname
DATABASE_URL=f"postgresql://postgres.avmyohbogdlamvppmesm:{encoded_password}@aws-0-ap-northeast-1.pooler.supabase.com:6543/postgres"
# initialise the sqlalchemy engine
# nullpool to close connection immediately after use, sqlalchemy runs perpetually in the background,
# so we need to close the connection after use to avoid connection limit error
# also echo to print all the sql statements to the console for debugging
engine = create_engine(DATABASE_URL, poolclass=NullPool, echo=True)

# Test the connection + error handling
try:
    with engine.connect() as connection:
        result = connection.execute(text("SELECT version();"))
        print("Connected successfully!")
        print("PostgreSQL Version:", result.fetchone()[0])
except Exception as e:
    print(f"An error occurred: {e}")
    
# define base class for models
class Base(DeclarativeBase):
    pass

class creditCard(Base):
    __tablename__ = 'creditCard'  # Must match the case-sensitive table name in Supabase
    
    # Map the primary key column (case-sensitive)
    Id: Mapped[int] = mapped_column(Integer, primary_key=True)
    
    # Map other columns you need to query or filter by
    Time: Mapped[int] = mapped_column(BigInteger)
    Amount: Mapped[float] = mapped_column(Float)
    Class: Mapped[bool] = mapped_column(Boolean)  # True/False maps to 1/0 in Postgres
    
    # Map V1 through V28 columns dynamically to save code space
    # (Since you have V1 to V28 as double precision variables)
    locals().update({
        f"V{i}": mapped_column(Double, nullable=False) for i in range(1, 29)
    })


session = Session(engine)

stmt = select(creditCard).where(creditCard.Class == True).limit(10)

transaction_list = []
for row in session.scalars(stmt).all():
    v_list = [getattr(row, f"V{i}") for i in range(1, 29)]
    # can use *v_list to unpack the list, model see list as 1 element so not good
    transaction_list.append((row.Amount, row.Time, int(row.Class), *v_list))

print("=================================================================================")
for i in range(3):
    print(f"Transition number {i + 1}: {transaction_list[i]}")

print("=================================================================================\n=================================================================================\n=================================================================================")
print(transaction_list)