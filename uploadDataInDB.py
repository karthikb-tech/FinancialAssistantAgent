# uploadDataInDB.py
import logging
import os
from fastapi import FastAPI, UploadFile, File, HTTPException
import pandas as pd
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, Numeric, Date, Text, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sentence_transformers import SentenceTransformer
from sqlalchemy import ForeignKey
from sqlalchemy.orm import relationship
from collections import defaultdict
from pgvector.sqlalchemy import Vector  # pgvector integration

# --- Logging ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# --- FastAPI ---
app = FastAPI(title="Financial Data API", version="1.1.0")

# --- Database Setup ---
DATABASE_URL = os.getenv("DATABASE_URL")

# --- Embedding Setup ---
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class Transaction(Base):
    __tablename__ = "transactions"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    transaction_date = Column(Date, nullable=False)
    description = Column(Text, nullable=False)
    debit = Column(Numeric, nullable=True)  # money type → Numeric
    credit = Column(Numeric, nullable=True)  # money type → Numeric
    category = Column(Text, nullable=False)

class TransactionEmbedding(Base):
    __tablename__ = "transaction_embeddings"

    id = Column(Integer, primary_key=True, autoincrement=True)
    transaction_id = Column(Integer, ForeignKey("transactions.id"), nullable=True)
    text = Column(String, nullable=False)
    embedding = Column(Vector(384), nullable=False)  # pgvector column
    balance = Column(Float, nullable=True)
    month = Column(String, nullable=True)

    transaction = relationship("Transaction", backref="embeddings")


Base.metadata.create_all(bind=engine)


# --- Helper: Clean DataFrame ---
def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    current_year = datetime.now().year

    # Drop footer rows like "Totals"
    df = df[~df["Date"].str.contains("Totals", na=False)].copy()

    # Parse dates
    df.loc[:, "Date"] = df["Date"].apply(
        lambda x: datetime.strptime(x.strip(), "%b %d").replace(year=current_year)
    )

    # Clean numeric fields
    for col in ["Debit (-)", "Credit (+)", "Balance"]:
        df.loc[:, col] = (
            df[col]
            .astype(str)
            .replace(r"[\$,]", "", regex=True)
            .replace("", "0")
            .astype(float)
        )

    # Rename columns
    df = df.rename(
        columns={
            "Date": "transaction_date",
            "Transaction Description": "description",
            "Category": "category",
            "Debit (-)": "debit",
            "Credit (+)": "credit",
            "Balance": "balance",
        }
    )

    # Replace NaN with None for JSON + DB
    df = df.where(pd.notnull(df), None)

    return df


# --- Validation ---
def validate_row(row) -> bool:
    """Return True if row is valid, False otherwise."""
    logging.info("Row sample: %s", row.to_dict())
    # Must have a date
    if not row["transaction_date"]:
        return False
    # Must have a description
    if not row["description"] or str(row["description"]).strip() == "":
        return False
    # Must have a category
    if not row["category"] or str(row["category"]).strip() == "":
        return False
    # Must have a balance
    if row["balance"] is None:
        return False
    # Debit and credit can be None, but not both missing
    if row["debit"] is None and row["credit"] is None:
        return False

    return True


# --- Preview Endpoint ---
@app.post("/upload-csv/")
async def upload_csv(file: UploadFile = File(...)):
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only .csv files are supported")

    try:
        df = pd.read_csv(file.file)
        df = clean_dataframe(df)

        preview = df.head(10).to_dict(orient="records")
        return {
            "message": f"File '{file.filename}' processed successfully",
            "total_rows": len(df),
            "columns": list(df.columns),
            "preview": preview,
        }

    except Exception as e:
        logging.error("Error processing CSV: %s", e)
        raise HTTPException(status_code=500, detail=f"Error processing file: {e}")

# --- Save Endpoint with Validation ---
@app.post("/save-csv/")
async def save_csv(file: UploadFile = File(...)):
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only .csv files are supported")

    try:
        df = pd.read_csv(file.file)
        df = clean_dataframe(df)

        session = SessionLocal()
        inserted, skipped = 0, 0

        for _, row in df.iterrows():
            if validate_row(row):
                txn = Transaction(
                    transaction_date=row["transaction_date"],
                    description=row["description"],
                    category=row["category"],
                    debit=row["debit"],
                    credit=row["credit"],
                    balance=row["balance"],
                )
                session.add(txn)
                inserted += 1
            else:
                skipped += 1

        session.commit()
        session.close()

        return {
            "message": f"Inserted {inserted} transactions, skipped {skipped} invalid rows",
            "total_rows": len(df),
            "inserted": inserted,
            "skipped": skipped,
        }

    except Exception as e:
        logging.error("Error saving CSV to DB: %s", e)
        raise HTTPException(status_code=500, detail=f"Error saving file: {e}")

# --- Convert Endpoint ---
@app.post("/convert-csv/")
async def convert_csv(file: UploadFile = File(...)):
    """
    Upload a CSV, clean + validate rows, save transactions,
    embed descriptions, and store embeddings in transaction_embeddings (pgvector).
    """
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only .csv files are supported")

    try:
        df = pd.read_csv(file.file)
        df = clean_dataframe(df)
        df.sort_values("transaction_date", inplace=True)

        session = SessionLocal()
        sentences, mapping, balances, months = [], [], [], []
        monthly_balances = defaultdict(float)
        inserted, skipped = 0, 0
        running_balance = 0.0

        # --- Save transactions ---
        for _, row in df.iterrows():
            if validate_row(row):
                txn = Transaction(
                    transaction_date=row["transaction_date"],
                    description=row["description"],
                    category=row["category"],
                    debit=row["debit"],
                    credit=row["credit"]
                )
                session.add(txn)
                session.flush()  # get txn.id before commit

                amount = row["debit"] if row["debit"] else row["credit"]
                txn_type = "Debit" if row["debit"] else "Credit"
                if txn_type == "Debit":
                    running_balance -= amount
                else:
                    running_balance += amount

                month_key = row["transaction_date"].strftime("%Y-%m")
                monthly_balances[month_key] = running_balance

                sentence = (
                    f"{row['description']} for {amount} on {row['transaction_date'].strftime('%Y-%m-%d')}. "
                    f"Category: {row['category']}. Type: {txn_type}"
                )

                sentences.append(sentence)
                mapping.append(txn.id)
                balances.append(running_balance)
                months.append(month_key)
                inserted += 1
            else:
                skipped += 1

        # --- Add monthly balance summaries ---
        for month, balance in sorted(monthly_balances.items()):
            summary = f"Balance for end of {month} is ${balance:,.2f}"
            sentences.append(summary)
            mapping.append(None)  # summaries have no transaction_id
            balances.append(balance)
            months.append(month)

        # --- Generate embeddings with pgvector ---
        embeddings = embedding_model.encode(sentences, convert_to_numpy=True)

        # --- Save embeddings ---
        for tid, text, emb, bal, month in zip(mapping, sentences, embeddings, balances, months):
            emb_row = TransactionEmbedding(
                transaction_id=tid,   # can be None for summaries
                text=text,
                embedding=emb,        # pgvector expects raw numpy array
                balance=bal,
                month=month
            )
            session.add(emb_row)

        session.commit()
        session.close()

        return {
            "message": f"Inserted {inserted} transactions and embeddings, skipped {skipped} invalid rows",
            "monthly_summaries": len(monthly_balances),
            "total_embeddings": len(sentences)
        }

    except Exception as e:
        logging.error("Error saving CSV + embeddings: %s", e)
        raise HTTPException(status_code=500, detail=f"Error saving file: {e}")

