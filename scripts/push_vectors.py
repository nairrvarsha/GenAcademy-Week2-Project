"""Store vectors already generated in the notebook; never call an embedder."""

import json
import math
import time

from pinecone import ServerlessSpec

from rag_store import DIMENSIONS, INDEX, MODEL, ROOT, as_document, dataset_hash, get_pinecone, load_chunks


def push_vectors(chunks, documents, vectors, embedding_model):
    # Validate notebook execution order before making any remote changes.
    if embedding_model != MODEL:
        raise ValueError("Embedding model changed. Run section 3 again.")
    if not chunks or not (len(chunks) == len(documents) == len(vectors)):
        raise ValueError("Chunk/document/vector counts differ. Complete sections 2 and 3 first.")
    if chunks != load_chunks():
        raise ValueError("Saved chunks differ from notebook chunks. Rerun sections 2 and 3.")
    records = []
    for chunk, document, vector in zip(chunks, documents, vectors):
        expected = as_document(chunk)
        if (document.id != expected.id or document.page_content != expected.page_content
                or document.metadata != expected.metadata):
            raise ValueError("Documents changed after chunking. Run section 3 again.")
        if len(vector) != DIMENSIONS or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in vector):
            raise ValueError("An embedding has an invalid size or non-finite values.")
        records.append({"id": document.id, "values": vector,
                        "metadata": {**document.metadata, "text": document.page_content}})

    pc = get_pinecone()
    if INDEX not in pc.list_indexes().names():
        print(f"Creating Pinecone index {INDEX}...")
        pc.create_index(
            name=INDEX, dimension=DIMENSIONS, metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            deletion_protection="enabled",
        )
    for _ in range(60):
        description = pc.describe_index(INDEX)
        if (description.dimension != DIMENSIONS or description.metric != "cosine"
                or description.get("embed")):
            raise ValueError("The existing Pinecone index is incompatible with these embeddings.")
        if description.status["ready"]:
            break
        time.sleep(2)
    else:
        raise RuntimeError("Pinecone index is not ready yet. Retry section 4 shortly.")
    index = pc.Index(host=description.host)
    digest = dataset_hash(chunks)
    namespace = "chunks-" + digest
    for start in range(0, len(records), 10):
        batch = records[start:start + 10]
        # Pinecone SDK accepts precomputed vectors. LangChain add_documents would
        # embed the content again, so it isn't used in this separate upload step.
        response = index.upsert(vectors=batch, namespace=namespace)
        if response.upserted_count != len(batch):
            raise RuntimeError("Pinecone did not acknowledge every record. Retry section 4.")
        print(f"Uploaded {min(start + 10, len(records))}/{len(records)} vectors")

    for _ in range(12):
        count = index.describe_index_stats().namespaces.get(namespace, {}).get("vector_count", 0)
        if count == len(records):
            break
        time.sleep(3)
    else:
        raise RuntimeError("Upload acknowledged but count not visible yet. Retry section 4.")
    manifest = {"index": INDEX, "host": description.host, "namespace": namespace,
                "model": MODEL, "dimensions": DIMENSIONS, "metric": "cosine",
                "chunk_count": len(chunks), "dataset_hash": digest}
    (ROOT / "data/pinecone_index.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest
