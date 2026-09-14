"""LangChain retriever combining local BM25 and Pinecone using rank fusion."""

import math
import re
from typing import Any

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import PrivateAttr
from rank_bm25 import BM25Okapi

STOP_WORDS = set("a an the is are was were do does can could i me my you your please "
                 "what which where how for of to in on at and or with about its it".split())


def tokenize(text):
    return [word for word in re.findall(r"[^\W_]+", text.casefold()) if word not in STOP_WORDS]


def validate_filters(filters):
    filters = dict(filters or {})
    if set(filters) - {"locality", "cuisine", "max_cost", "valet"}:
        raise ValueError("Unknown restaurant filter.")
    for key in ("locality", "cuisine"):
        if key in filters and (not isinstance(filters[key], str) or not filters[key].strip()):
            raise ValueError(f"Invalid {key} filter.")
    if "valet" in filters and not isinstance(filters["valet"], bool):
        raise ValueError("Valet filter must be boolean.")
    if "max_cost" in filters:
        value = filters["max_cost"]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError("Budget must be a nonnegative number.")
    return filters


def eligible(document, filters):
    metadata = document.metadata
    # Platform FAQs remain searchable regardless of restaurant filters.
    if metadata.get("chunk_type") == "faq":
        return True
    if "locality" in filters and metadata.get("locality") != filters["locality"]:
        return False
    if "cuisine" in filters and filters["cuisine"] not in metadata.get("cuisines", []):
        return False
    if "valet" in filters and metadata.get("valet") != filters["valet"]:
        return False
    cost = metadata.get("approx_cost_for_two_inr")
    if "max_cost" in filters and (cost is None or cost > filters["max_cost"]):
        return False
    return True


def pinecone_filter(filters):
    conditions = []
    for key, field, operator in (
        ("locality", "locality", "$eq"), ("cuisine", "cuisines", "$in"),
        ("max_cost", "approx_cost_for_two_inr", "$lte"), ("valet", "valet", "$eq"),
    ):
        if key in filters:
            value = [filters[key]] if operator == "$in" else filters[key]
            conditions.append({field: {operator: value}})
    if not conditions:
        return None
    return {"$or": [{"chunk_type": {"$eq": "faq"}}, {"$and": conditions}]}


def fuse_rankings(keyword, semantic, k=5):
    """Equal-weight RRF: add 1/(60+rank) per list; deduplicate by chunk ID."""
    scores, documents, ranks = {}, {}, {}
    for name, ranked in (("keyword", keyword), ("semantic", semantic)):
        seen = set()
        for rank, document in enumerate(ranked, start=1):
            if document.id in seen:
                continue
            seen.add(document.id)
            documents[document.id] = document
            scores[document.id] = scores.get(document.id, 0) + 1 / (60 + rank)
            ranks.setdefault(document.id, {})[name + "_rank"] = rank
    ordered = sorted(scores, key=lambda identifier: (-scores[identifier], identifier))[:k]
    return [
        (documents[identifier].model_copy(update={"metadata": {
            **documents[identifier].metadata,
            "retrieval": {**ranks[identifier], "rrf_score": scores[identifier]},
        }}), scores[identifier])
        for identifier in ordered
    ]


class HybridRetriever(BaseRetriever):
    store: Any
    documents: list[Document]
    k: int = 5
    candidate_k: int = 20
    _bm25: Any = PrivateAttr()

    def model_post_init(self, context):
        if not self.documents or any(not doc.id for doc in self.documents):
            raise ValueError("Hybrid retrieval requires documents with stable IDs.")
        if len({doc.id for doc in self.documents}) != len(self.documents):
            raise ValueError("Duplicate chunk IDs in keyword index.")
        self._bm25 = BM25Okapi([tokenize(doc.page_content) or ["empty"] for doc in self.documents])

    def search(self, query, k=None, filters=None):
        filters = validate_filters(filters)
        k = self.k if k is None else k
        if k < 1:
            raise ValueError("k must be positive.")
        candidate_k = max(k, self.candidate_k)
        scores = self._bm25.get_scores(tokenize(query))
        positions = sorted(range(len(scores)), key=lambda i: (-scores[i], self.documents[i].id))
        keyword = [self.documents[i] for i in positions
                   if scores[i] > 0 and eligible(self.documents[i], filters)][:candidate_k]
        kwargs = {"k": candidate_k}
        filter_expression = pinecone_filter(filters)
        if filter_expression:
            kwargs["filter"] = filter_expression
        dense = self.store.similarity_search_with_score(query, **kwargs)
        semantic = [doc for doc, score in dense if eligible(doc, filters)]
        # A semantic-service error is surfaced; never silently label BM25-only as hybrid.
        return fuse_rankings(keyword, semantic, k=k)

    def _get_relevant_documents(self, query, *, run_manager):
        return [document for document, score in self.search(query)]
