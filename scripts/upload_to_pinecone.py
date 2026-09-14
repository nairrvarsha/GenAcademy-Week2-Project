"""Upload with LangChain, reusing existing matching vectors.

.venv/bin/python scripts/upload_to_pinecone.py --check  # read-only connection test
.venv/bin/python scripts/upload_to_pinecone.py          # upload/resume
"""
import argparse
import json
import math
import time
from pinecone import ServerlessSpec
from rag_store import (
    DIMENSIONS, INDEX, MODEL, ROOT, as_document, dataset_hash,
    get_pinecone, load_chunks, make_store, run_cli, setting,
)


def matches(record, document):
    if record is None:
        return False
    return (
        len(record.values) == DIMENSIONS
        and all(math.isfinite(v) for v in record.values)
        and record.metadata.get("text") == document.page_content
        and all(record.metadata.get(k) == v for k, v in document.metadata.items())
    )


def upload_documents(index, store, documents, namespace):
    uploaded = skipped = 0
    for start in range(0, len(documents), 10):
        batch = documents[start:start + 10]
        records = index.fetch(ids=[d.id for d in batch], namespace=namespace).vectors
        pending = [d for d in batch if not matches(records.get(d.id), d)]
        if pending:
            # LangChain embeds page_content with Nebius, then uploads to Pinecone.
            store.add_documents(documents=pending, ids=[d.id for d in pending],
                                batch_size=10, embedding_chunk_size=10, async_req=False)
        uploaded += len(pending)
        skipped += len(batch) - len(pending)
        print(f"Checked {min(start + 10, len(documents))}/{len(documents)}; "
              f"uploaded {uploaded}, reused {skipped}", flush=True)
    return uploaded, skipped


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    pc = get_pinecone()
    names = pc.list_indexes().names()
    print("Pinecone SDK connection successful.", flush=True)
    if args.check:
        print(f"Project has {len(names)} indexes. No changes made.")
        return
    setting("NEBIUS_API_KEY")
    chunks = load_chunks()
    documents = [as_document(c) for c in chunks]
    digest = dataset_hash(chunks)
    namespace = "chunks-" + digest
    if INDEX not in names:
        pc.create_index(name=INDEX, dimension=DIMENSIONS, metric="cosine",
                        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
                        deletion_protection="enabled")
        print(f"Created {INDEX}", flush=True)
    for _ in range(60):
        description = pc.describe_index(INDEX)
        if (description.dimension != DIMENSIONS or description.metric != "cosine"
                or description.get("embed")):
            raise ValueError("Index has incompatible embedding settings.")
        if description.status["ready"]:
            break
        time.sleep(2)
    else:
        raise RuntimeError("Index is not ready yet.")
    index = pc.Index(host=description.host)
    store = make_store(index, namespace)
    uploaded, skipped = upload_documents(index, store, documents, namespace)
    for _ in range(12):
        stats = index.describe_index_stats()
        count = stats.namespaces.get(namespace, {}).get("vector_count", 0)
        if count == len(documents):
            break
        time.sleep(3)
    else:
        raise RuntimeError("Record count not yet verified; rerun to resume.")
    # Reused records were verified before skipping. Check new writes after propagation.
    if uploaded:
        for start in range(0, len(documents), 25):
            batch = documents[start:start + 25]
            for attempt in range(6):
                records = index.fetch(ids=[d.id for d in batch], namespace=namespace).vectors
                if all(matches(records.get(d.id), d) for d in batch):
                    break
                time.sleep(2)
            else:
                raise RuntimeError("Stored-record verification failed.")
    manifest = {"index": INDEX, "host": description.host, "namespace": namespace,
                "model": MODEL, "dimensions": DIMENSIONS, "metric": "cosine",
                "chunk_count": len(chunks), "dataset_hash": digest}
    (ROOT / "data/pinecone_index.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Verified {count} records. Uploaded {uploaded}; reused {skipped}.")


if __name__ == "__main__":
    run_cli(main)
