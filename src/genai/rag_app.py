
"""Insurance Claims Fraud Intelligence - GenAI / RAG application."""

from pathlib import Path
import re
import time

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CSV_PATH = ROOT / "data" / "raw" / "claims" / "claims_core.csv"
DB_PATH = ROOT / "outputs" / "genai" / "chroma_db"

COLLECTION_NAME = "insurance_claims"
LLM_MODEL = "phi3:mini"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 5


def load_claims():
    """Load original insurance claim records."""
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {CSV_PATH}")

    df = pd.read_csv(CSV_PATH, dtype={"claim_id": str})
    df.columns = [str(column).strip() for column in df.columns]

    if "claim_id" not in df.columns:
        raise ValueError("The dataset does not contain a claim_id column.")

    df["claim_id"] = df["claim_id"].astype(str).str.strip()
    df["_lookup_id"] = df["claim_id"].str.upper()
    return df.drop_duplicates("_lookup_id", keep="first")


def display_claim(row):
    """Display actual values from the original CSV."""
    print("\n" + "=" * 68)
    print(f"📄 CLAIM DETAILS: {row['claim_id']}")
    print("=" * 68)

    for column, value in row.items():
        if column in {"_lookup_id", "claim_id"} or pd.isna(value):
            continue
        if str(value).strip():
            label = column.replace("_", " ").title()
            print(f"{label:30}: {value}")

    print("\n📌 Source: Original claims_core.csv")
    print("✅ Values shown above come directly from the dataset.")


def find_claim_id(question):
    match = re.search(r"\bCLM\d+\b", question, re.IGNORECASE)
    return match.group(0).upper() if match else None


def load_rag():
    """Connect to the existing vector store; do not rebuild it."""
    import chromadb
    from sentence_transformers import SentenceTransformer

    if not DB_PATH.exists():
        raise FileNotFoundError(f"Vector database not found: {DB_PATH}")

    client = chromadb.PersistentClient(path=str(DB_PATH))
    collection = client.get_collection(name=COLLECTION_NAME)

    if collection.count() == 0:
        raise RuntimeError("The existing vector collection is empty.")

    embedding_model = SentenceTransformer(EMBEDDING_MODEL)
    return collection, embedding_model


def retrieve_claims(question, collection, embedding_model, claims):
    """Retrieve similar records and recover their original CSV values."""
    vector = embedding_model.encode(
        [question], normalize_embeddings=True
    ).tolist()

    result = collection.query(
        query_embeddings=vector,
        n_results=min(TOP_K, collection.count()),
        include=["metadatas", "distances"],
    )

    metadata_list = (result.get("metadatas") or [[]])[0]
    distances = (result.get("distances") or [[]])[0]
    lookup = claims.set_index("_lookup_id", drop=False)

    records = []

    for metadata, distance in zip(metadata_list, distances):
        metadata = metadata or {}
        claim_id = metadata.get("claim_id")

        if not claim_id:
            continue

        key = str(claim_id).strip().upper()
        if key not in lookup.index:
            continue

        row = lookup.loc[key]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]

        fields = {}
        for column, value in row.items():
            if column != "_lookup_id" and not pd.isna(value):
                fields[column] = str(value)

        records.append({
            "claim_id": str(row["claim_id"]),
            "fields": fields,
            "distance": float(distance),
        })

    return records


def generate_answer(question, records):
    """Generate an answer using local Phi-3 Mini."""
    import ollama

    evidence = "\n\n".join(
        f"Claim ID: {record['claim_id']}\n"
        + "\n".join(
            f"{key}: {value}"
            for key, value in record["fields"].items()
        )
        for record in records
    )

    response = ollama.chat(
        model=LLM_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an insurance claims data analyst. Answer using "
                    "only the supplied evidence. Do not invent fields, values, "
                    "dates, amounts, or claim IDs. Keep claims separate. "
                    "Do not infer fraud from similarity. If the evidence is "
                    "insufficient, say so. Use at most five concise bullets."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Question: {question}\n\n"
                    f"Retrieved original claim records:\n{evidence}"
                ),
            },
        ],
        options={"temperature": 0, "num_predict": 250},
    )

    return response["message"]["content"].strip()


def main():
    print("\n" + "=" * 68)
    print("🚗 INSURANCE CLAIMS FRAUD INTELLIGENCE")
    print("🤖 Generative AI | Retrieval-Augmented Generation (RAG)")
    print("=" * 68)

    print("\n🧰 TECHNOLOGIES USED")
    print("• Python and Pandas — dataset loading and exact claim lookup")
    print("• ChromaDB — vector database and semantic retrieval")
    print("• Sentence Transformers — text embeddings")
    print("• Ollama with Phi-3 Mini — local language model")
    print("• Original claims CSV — source of claim details")

    claims = load_claims()
    print(f"\n📊 Dataset records: {len(claims):,}")
    print("✅ Application ready.")
    print("💬 Ask your question, or type 'exit' to close.")

    collection = None
    embedding_model = None

    while True:
        try:
            question = input("\n💬 Ask your question: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n👋 Application closed.")
            break

        if question.lower() in {"exit", "quit"}:
            print("👋 Thank you for using Insurance Claims Fraud Intelligence!")
            break

        if not question:
            continue

        # Exact claim lookup: accurate CSV values, no LLM required.
        claim_id = find_claim_id(question)
        if claim_id:
            matches = claims[claims["_lookup_id"] == claim_id]

            if matches.empty:
                print(f"\n❌ Claim {claim_id} was not found in the dataset.")
            else:
                display_claim(matches.iloc[0])
            continue

        # General question: retrieve from the existing vector database.
        try:
            if collection is None:
                print("\n🔎 Loading retrieval components...")
                collection, embedding_model = load_rag()
                print(f"📚 Available indexed records: {collection.count():,}")

            start = time.perf_counter()
            records = retrieve_claims(
                question, collection, embedding_model, claims
            )
            retrieval_time = time.perf_counter() - start

            if not records:
                print("\n⚠️ No matching source records were found.")
                continue

            start = time.perf_counter()
            answer = generate_answer(question, records)
            generation_time = time.perf_counter() - start

            print("\n" + "=" * 68)
            print("✨ EVIDENCE-BASED RESPONSE")
            print("=" * 68)
            print(answer)

            print("\n📚 RETRIEVED SOURCE REFERENCES")
            for record in records:
                print(
                    f"• {record['claim_id']} "
                    f"(distance: {record['distance']:.4f})"
                )

            print("\n⚙️ EXECUTION METRICS")
            print(f"• Retrieved records: {len(records)}")
            print(f"• Retrieval time: {retrieval_time:.2f} seconds")
            print(f"• LLM response time: {generation_time:.2f} seconds")
            print("ℹ️ Embedding distance is not a fraud probability.")

        except Exception as exc:
            print(f"\n❌ Request failed: {exc}")
            print("Check the existing vector database and Ollama server.")


if __name__ == "__main__":
    main()
