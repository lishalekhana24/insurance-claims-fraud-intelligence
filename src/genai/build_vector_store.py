"""Build a ChromaDB vector store from redacted insurance claim records."""

import json
import sys
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_PATH = PROJECT_ROOT / "outputs" / "genai" / "claims_knowledge_base.jsonl"
DB_PATH = PROJECT_ROOT / "outputs" / "genai" / "chroma_db"

# Lightweight embedding model suitable for a local demonstration.
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION_NAME = "insurance_claims"


def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Knowledge base not found: {INPUT_PATH}. "
            "Run claims_knowledge_base.py first."
        )

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from pii_redaction import redact_pii

    print("Loading embedding model...")
    model = SentenceTransformer(EMBEDDING_MODEL)

    client = chromadb.PersistentClient(path=str(DB_PATH))
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    # Rebuild deterministically so reruns do not duplicate records.
    existing = collection.count()
    if existing:
        client.delete_collection(COLLECTION_NAME)
        collection = client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    batch_ids, batch_docs, batch_meta = [], [], []
    batch_size = 64
    total = 0

    def write_batch():
        nonlocal total
        if not batch_ids:
            return

        embeddings = model.encode(
            batch_docs, normalize_embeddings=True
        ).tolist()

        collection.add(
            ids=batch_ids,
            documents=batch_docs,
            metadatas=batch_meta,
            embeddings=embeddings,
        )
        total += len(batch_ids)
        batch_ids.clear()
        batch_docs.clear()
        batch_meta.clear()

    with INPUT_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            safe_text = redact_pii(record["text"])

            batch_ids.append(record["id"])
            batch_docs.append(safe_text)
            batch_meta.append({
                "claim_id": record["metadata"]["claim_id"],
                "claim_year": record["metadata"]["claim_year"],
                "fraud_label": record["metadata"]["fraud_label"],
                "source": record["metadata"]["source"],
                "text_type": record["metadata"]["text_type"],
            })

            if len(batch_ids) >= batch_size:
                write_batch()
                if total % 1024 == 0:
                    print(f"Indexed {total} records...")

    write_batch()

    print(f"Vector indexing completed: {total} records")
    print(f"ChromaDB location: {DB_PATH}")
    print(f"Collection: {COLLECTION_NAME}")


if __name__ == "__main__":
    main()