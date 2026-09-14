"""Shared LangChain documents, embeddings, and vector store."""
import hashlib
import json
import os
from pathlib import Path
import sys
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_nebius import NebiusEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

ROOT = Path(__file__).resolve().parents[1]
MODEL = "Qwen/Qwen3-Embedding-8B"
DIMENSIONS = 4096
INDEX = "restaurant-kb-qwen3"
BASE_URL = "https://api.tokenfactory.nebius.com/v1/"


def setting(name):
    load_dotenv(ROOT / ".env", override=False)
    value = os.environ.get(name, "").strip()
    if not value:
        raise ValueError(f"Set {name} in .env and save it.")
    return value


def compact(content):
    # Keep the exact JSON representation already embedded in Pinecone.
    return json.dumps(content, ensure_ascii=False, separators=(",", ":"))


def load_chunks():
    chunks = json.loads((ROOT / "data/chunks.json").read_text())
    ids = [c["chunk_id"] for c in chunks]
    if not chunks or len(ids) != len(set(ids)):
        raise ValueError("Expected nonempty chunks with unique IDs.")
    return chunks


def dataset_hash(chunks):
    return hashlib.sha256((MODEL + compact(chunks)).encode()).hexdigest()[:16]


def as_document(chunk):
    metadata = {k: v for k, v in chunk["metadata"].items() if v is not None}
    metadata["embedding_model"] = MODEL
    return Document(id=chunk["chunk_id"], page_content=compact(chunk["content"]), metadata=metadata)


def get_embeddings():
    return NebiusEmbeddings(
        model=MODEL, api_key=setting("NEBIUS_API_KEY"), base_url=BASE_URL,
        timeout=60, max_retries=2, model_kwargs={"encoding_format": "float"},
    )


def get_pinecone():
    # Administrative operations use the Pinecone SDK.
    return Pinecone(api_key=setting("PINECONE_API_KEY"))


def make_store(index, namespace):
    return PineconeVectorStore(index=index, embedding=get_embeddings(),
                               namespace=namespace, text_key="text")


def existing_store():
    manifest = json.loads((ROOT / "data/pinecone_index.json").read_text())
    if (manifest["model"] != MODEL or manifest["dimensions"] != DIMENSIONS
            or manifest["metric"] != "cosine"):
        raise ValueError("Saved index is incompatible with embedding settings.")
    if manifest["dataset_hash"] != dataset_hash(load_chunks()):
        raise ValueError("Chunks changed; run upload_to_pinecone.py first.")
    index = get_pinecone().Index(host=manifest["host"])
    return make_store(index, manifest["namespace"])


def run_cli(main):
    try:
        main()
    except Exception as error:
        # SDK error bodies may contain request details: do not print them.
        status = getattr(error, "status_code", None) or getattr(error, "status", None)
        suffix = f" (HTTP {status})" if status else ""
        print(f"Failed: {type(error).__name__}{suffix}. Check saved keys, network access, "
              "and index/chunk configuration. Error body omitted.", file=sys.stderr)
        sys.exit(1)
