import logging
import re
from typing import Any, Dict, List, Optional

from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.chroma import ChromaVectorStore

import chromadb
from src.agentic_rag.config import (
    HYBRID_KEYWORD_TOP_K,
    HYBRID_RERANK_ENABLED,
    HYBRID_RERANK_MODEL,
    HYBRID_RERANK_TOP_N,
    HYBRID_VECTOR_TOP_K,
    RRF_K,
)
from src.data_processing.config import (
    CHROMADB_DIR,
    EMBEDDING_DIMENSIONS,
    EMBEDDING_MODEL,
    OPENAI_API_KEY,
)

logger = logging.getLogger(__name__)
_RERANKER: Any = None
_RERANKER_LOAD_ATTEMPTED = False

_OSHA_STD_RE = re.compile(
    r"\b(19\d{2}\.\d{1,4}(?:\([a-z]\)(?:\(\d+\))?)?)\b", re.IGNORECASE
)
_COMPACT_STANDARD_RE = re.compile(
    r"\bstandard\s+(19\d{6})\s+([A-Z]\d{2})(?:\s+((?:I{1,3})|IV))?\b",
)
_CITATION_ID_RE = re.compile(
    r"\bcitation(?:\s+(?:number|#|no\.?)?)?\s*(\d{5}[A-Z]?)\b",
    re.IGNORECASE,
)
_OSHA_CITATION_SHORT_RE = re.compile(
    r"\bosha\s+citation\s+(\d{5}[A-Z]?)\b",
    re.IGNORECASE,
)
_AT_ESTABLISHMENT_RE = re.compile(
    r"\bat\s+([A-Za-z0-9&'./,\-]+(?:\s+[A-Za-z0-9&'./,\-]+){0,8}?)"
    r"(?=\s+in\s+[A-Za-z]|\s+under\s+|\s*$|\.)",
    re.IGNORECASE,
)
_ACCIDENT_ID_RE = re.compile(r"\b\d{8,9}\b")
_YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")
_FACILITY_PHRASE_RE = re.compile(
    r"\b([A-Za-z0-9&'./-]+(?:\s+[A-Za-z0-9&'./-]+){0,4}\s+"
    r"(?:refinery|plant|facility|terminal|complex|works|laboratory|lab))\b",
    re.IGNORECASE,
)
_STATE_ABBREVS = {
    "AL",
    "AK",
    "AZ",
    "AR",
    "CA",
    "CO",
    "CT",
    "DE",
    "FL",
    "GA",
    "HI",
    "ID",
    "IL",
    "IN",
    "IA",
    "KS",
    "KY",
    "LA",
    "ME",
    "MD",
    "MA",
    "MI",
    "MN",
    "MS",
    "MO",
    "MT",
    "NE",
    "NV",
    "NH",
    "NJ",
    "NM",
    "NY",
    "NC",
    "ND",
    "OH",
    "OK",
    "OR",
    "PA",
    "RI",
    "SC",
    "SD",
    "TN",
    "TX",
    "UT",
    "VT",
    "VA",
    "WA",
    "WV",
    "WI",
    "WY",
}
try:
    from src.evaluation.utils import _CHEMICAL_KEYWORDS as _EVAL_CHEMICAL_KEYWORDS
except Exception:
    _EVAL_CHEMICAL_KEYWORDS = (
        "ammonia",
        "chlorine",
        "benzene",
        "ethylene oxide",
        "hydrogen sulfide",
    )
_CHEMICAL_KEYWORDS = tuple(str(x).lower() for x in _EVAL_CHEMICAL_KEYWORDS)

# Mapping from lowercase query substrings to canonical CSB incident_name metadata values.
# Used for both metadata filtering (workers.py) and BM25 keyword injection (tools.py).
_CSB_INCIDENT_NAME_MAP: Dict[str, str] = {
    "watson grinding": "Watson_Grinding_and_Manufacturing",
    "ab specialty silicones": "AB_Specialty_Silicones,_LLC",
    "ab specialty": "AB_Specialty_Silicones,_LLC",
    "west fertilizer": "West_Fertilizer_Explosion_and_Fire",
    "imperial sugar": "Imperial_Sugar_Company_Dust_Explosion",
    "dupont la porte": "DuPont_La_Porte",
    "bp texas city": "BP_Texas_City",
    "t2 laboratories": "T2_Laboratories",
    "bayer cropscience": "Bayer_CropScience",
    "tesoro anacortes": "Tesoro_Anacortes",
    "chevron richmond": "Chevron_Richmond",
    "williams geismar": "Williams_Geismar",
    "exxonmobil torrance": "ExxonMobil_Torrance",
    "mgpi processing": "MGPI_Processing",
    "husky superior": "Husky_Superior",
    "kuraray": "Kuraray_America",
}


def _extract_citation_ids(query: str) -> List[str]:
    """Extract OSHA citation IDs (e.g. 01001, 01001B) from violation-style queries."""
    ids: List[str] = []
    seen = set()
    for pattern in (_CITATION_ID_RE, _OSHA_CITATION_SHORT_RE):
        for match in pattern.finditer(query):
            cid = match.group(1).upper()
            if cid not in seen:
                seen.add(cid)
                ids.append(cid)
    return ids


def _extract_enforcement_standard_number(query: str) -> Optional[str]:
    """Normalize compact OSHA standard references used in enforcement queries."""
    match = _COMPACT_STANDARD_RE.search(query)
    if not match:
        return None
    parts = [match.group(1), match.group(2).upper()]
    if match.group(3):
        parts.append(match.group(3).upper())
    return " ".join(parts)


def _extract_establishment_at_clause(query: str) -> Optional[str]:
    """Extract establishment names from 'at COMPANY ...' phrasing in enforcement queries."""
    match = _AT_ESTABLISHMENT_RE.search(query)
    if not match:
        return None
    return re.sub(r"\s+", " ", match.group(1)).strip() or None


def _extract_keyword_terms(query: str) -> List[str]:
    """Extract domain-specific identifiers for keyword-based retrieval."""
    terms: List[str] = []
    q_lower = query.lower()

    for match in _OSHA_STD_RE.finditer(query):
        terms.append(match.group(1))

    compact_std = _extract_enforcement_standard_number(query)
    if compact_std:
        terms.append(compact_std)

    if re.search(r"\b(accident|report)\b", q_lower):
        for match in _ACCIDENT_ID_RE.finditer(query):
            terms.append(match.group(0))

    if re.search(r"\b(violation|citation|inspection|penalty|penalti)\b", q_lower):
        for match in re.finditer(r"\b\d{9,12}\b", query):
            terms.append(match.group(0))
        terms.extend(_extract_citation_ids(query))
        establishment = _extract_establishment_at_clause(query)
        if establishment:
            terms.append(establishment)

    for year in _YEAR_RE.findall(query):
        terms.append(year)

    for chemical in _CHEMICAL_KEYWORDS:
        if chemical in q_lower:
            terms.append(chemical)

    for match in _FACILITY_PHRASE_RE.finditer(query):
        phrase = re.sub(r"\s+", " ", match.group(1)).strip()
        if phrase:
            terms.append(phrase)

    # CSB incident name keywords: expand canonical name into component tokens for BM25
    for substring, canonical_name in _CSB_INCIDENT_NAME_MAP.items():
        if substring in q_lower:
            terms.extend(canonical_name.replace("_", " ").split())
            break

    deduped: List[str] = []
    seen = set()
    for term in terms:
        normalized = term.strip()
        if len(normalized) < 3:
            continue
        key = normalized.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(normalized)

    return deduped


def _build_chroma_where_clause(
    metadata_filters: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Convert extracted metadata filters into a ChromaDB-compatible where clause.

    Chroma expects exactly one top-level operator in ``where``. When multiple
    field filters are present (e.g., ``site_state`` + ``fatality``), wrap them
    in an explicit ``$and`` list.
    """
    if not metadata_filters:
        return None

    if len(metadata_filters) == 1:
        return metadata_filters

    return {"$and": [{k: v} for k, v in metadata_filters.items()]}


_DOC_TYPE_VIOLATION_RE = re.compile(
    r"\b(violation|citation|cited|penalty|penalti|willful|serious|repeat)\b",
    re.IGNORECASE,
)
_DOC_TYPE_GENDUTY_RE = re.compile(
    r"\b(general\s+duty|5\(a\)\(1\)|section\s+5|general\s+duty\s+clause)\b",
    re.IGNORECASE,
)
_DOC_TYPE_ACCIDENT_RE = re.compile(
    r"\b(accident|fatalit(?:y|ies)|death|deaths|killed|hospitali[sz]|event)\b",
    re.IGNORECASE,
)


def _infer_query_doc_type_preference(query: str) -> Optional[str]:
    """Infer the preferred doc_type from query keywords.

    Used to apply a boost factor in RRF scoring so that chunks whose
    ``doc_type`` metadata matches the query intent are ranked higher.

    General duty is checked before violation since it is more specific
    (a "general duty clause violation" should prefer genduty chunks).
    """
    if _DOC_TYPE_GENDUTY_RE.search(query):
        return "enforcement_general_duty"
    if _DOC_TYPE_VIOLATION_RE.search(query):
        return "enforcement_violation"
    if _DOC_TYPE_ACCIDENT_RE.search(query):
        return "enforcement_accident"
    return None


def _reciprocal_rank_fusion(
    result_lists: List[List[Dict]],
    k: int = RRF_K,
    preferred_doc_type: Optional[str] = None,
    doc_type_boost: float = 2.0,
) -> List[Dict]:
    """Merge multiple ranked result lists using Reciprocal Rank Fusion.

    When *preferred_doc_type* is set, chunks whose ``metadata["doc_type"]``
    matches the preference receive a multiplicative boost to their RRF score,
    helping enforcement-specific chunks rise above regulation-style chunks.
    """
    scores: Dict[str, float] = {}
    docs_by_id: Dict[str, Dict] = {}

    for result_list in result_lists:
        for rank, doc in enumerate(result_list):
            doc_id = doc["id"]
            base_score = 1.0 / (k + rank + 1)
            if (
                preferred_doc_type
                and doc.get("metadata", {}).get("doc_type") == preferred_doc_type
            ):
                base_score *= doc_type_boost
            scores[doc_id] = scores.get(doc_id, 0.0) + base_score
            docs_by_id[doc_id] = doc

    ranked_ids = sorted(scores, key=lambda d: scores[d], reverse=True)
    if preferred_doc_type and logger.isEnabledFor(logging.INFO):
        for doc_id in ranked_ids[:10]:
            doc = docs_by_id.get(doc_id, {})
            doc_type = (doc.get("metadata") or {}).get("doc_type", "unknown")
            was_boosted = doc_type == preferred_doc_type
            logger.info(
                "RRF | doc=%s | type=%s | score=%.4f | boosted=%s",
                doc_id[:40],
                doc_type,
                scores.get(doc_id, 0.0),
                was_boosted,
            )
    return [docs_by_id[doc_id] for doc_id in ranked_ids]


def _lexical_overlap_score(query: str, text: str) -> float:
    """Simple fallback score when cross-encoder reranker is unavailable."""
    q_tokens = {t for t in re.findall(r"\b\w+\b", query.lower()) if len(t) > 2}
    t_tokens = {t for t in re.findall(r"\b\w+\b", (text or "").lower()) if len(t) > 2}
    if not q_tokens or not t_tokens:
        return 0.0
    return len(q_tokens.intersection(t_tokens)) / len(q_tokens)


def _get_cross_encoder() -> Any:
    global _RERANKER, _RERANKER_LOAD_ATTEMPTED
    if _RERANKER is not None:
        return _RERANKER
    if _RERANKER_LOAD_ATTEMPTED:
        return None
    _RERANKER_LOAD_ATTEMPTED = True
    try:
        from sentence_transformers import CrossEncoder

        # Force CPU to avoid MPS segfaults under concurrent async inference
        _RERANKER = CrossEncoder(HYBRID_RERANK_MODEL, device="cpu")
    except Exception as exc:
        logger.warning(
            "Cross-encoder reranker unavailable (%s). Falling back to lexical overlap scoring.",
            exc,
        )
        _RERANKER = None
    return _RERANKER


def _rerank_results(query: str, results: List[Dict], top_k: int) -> List[Dict]:
    """Rerank top-N RRF results using a cross-encoder, then truncate to top_k."""
    if not results:
        return []
    if not HYBRID_RERANK_ENABLED:
        return results[:top_k]

    rerank_pool = results[:HYBRID_RERANK_TOP_N]
    remainder = results[HYBRID_RERANK_TOP_N:]

    model = _get_cross_encoder()
    if model is not None:
        try:
            pairs = [(query, r.get("text", "")) for r in rerank_pool]
            scores = model.predict(pairs)
            ranked = sorted(
                zip(rerank_pool, scores),
                key=lambda x: float(x[1]),
                reverse=True,
            )
            reranked = [item for item, _ in ranked]
        except Exception as exc:
            logger.warning(
                "Cross-encoder rerank failed (%s). Using lexical fallback.", exc
            )
            reranked = sorted(
                rerank_pool,
                key=lambda r: _lexical_overlap_score(query, r.get("text", "")),
                reverse=True,
            )
    else:
        reranked = sorted(
            rerank_pool,
            key=lambda r: _lexical_overlap_score(query, r.get("text", "")),
            reverse=True,
        )

    return (reranked + remainder)[:top_k]


def _resolve_source(meta: Dict) -> str:
    """Derive a human-readable source label from chunk metadata."""
    std = meta.get("standard_number", "")
    section = meta.get("section_id", "")
    if std and section:
        return f"{std} \u00a7{section}"
    if std:
        return std
    title = meta.get("title", "")
    if title:
        return title
    report = meta.get("report_name", "") or meta.get("incident_name", "")
    if report:
        return report
    return meta.get("source_path", "unknown")


def _resolve_location(meta: Dict) -> str:
    """Derive a page/section location string from chunk metadata."""
    section = meta.get("section_id", "")
    if section:
        return section
    page_start = meta.get("page_start")
    page_end = meta.get("page_end")
    if page_start is not None:
        if page_end is not None and page_end != page_start:
            return f"pp. {page_start}-{page_end}"
        return f"p. {page_start}"
    return meta.get("section_title", "n/a")


def _deduplicate_results(
    results: List[Dict], max_text_overlap: float = 0.85
) -> List[Dict]:
    """Remove near-duplicate chunks based on text similarity.

    Compares each chunk's text to already-accepted chunks.  If the ratio
    of shared 4-gram shingles exceeds *max_text_overlap*, the duplicate
    is dropped.  This eliminates the CSB boilerplate problem where
    identical disclaimers from different reports dominate the results.
    """
    if not results:
        return results

    def _shingle_set(text: str, n: int = 4) -> set:
        words = text.lower().split()
        if len(words) < n:
            return {text.lower()}
        return {" ".join(words[i : i + n]) for i in range(len(words) - n + 1)}

    accepted: List[Dict] = []
    accepted_shingles: List[set] = []

    for doc in results:
        text = doc.get("text", "")
        if not text:
            accepted.append(doc)
            accepted_shingles.append(set())
            continue

        doc_shingles = _shingle_set(text)
        is_dup = False
        for prev_shingles in accepted_shingles:
            if not prev_shingles or not doc_shingles:
                continue
            overlap = len(doc_shingles & prev_shingles) / max(len(doc_shingles), 1)
            if overlap >= max_text_overlap:
                is_dup = True
                break

        if not is_dup:
            accepted.append(doc)
            accepted_shingles.append(doc_shingles)

    if len(accepted) < len(results):
        logger.info(
            "Dedup removed %d/%d near-duplicate chunks",
            len(results) - len(accepted),
            len(results),
        )
    return accepted


def _format_results_with_sources(results: List[Dict], collection_name: str) -> str:
    """Format retrieved documents with source metadata for citation tracking."""
    if not results:
        return f"No relevant documents found in {collection_name}."

    parts = []
    for i, doc in enumerate(results, 1):
        meta = doc.get("metadata", {})
        source = _resolve_source(meta)
        location = _resolve_location(meta)
        text = doc.get("text", "")

        parts.append(
            f"[Source {i}] Collection: {collection_name} | "
            f"Document: {source} | Location: {location}\n{text}"
        )
    return "\n\n---\n\n".join(parts)


def _extract_citation_metadata(
    results: List[Dict], collection_name: str
) -> List[Dict[str, Any]]:
    """Extract structured citation metadata from results."""
    citations = []
    for doc in results:
        meta = doc.get("metadata", {})
        citation = {
            "collection": collection_name,
            "source": _resolve_source(meta),
            "location": _resolve_location(meta),
            "chunk_id": doc.get("id", "unknown"),
        }
        similarity_score = doc.get("similarity_score")
        if isinstance(similarity_score, (int, float)):
            citation["similarity_score"] = round(float(similarity_score), 6)
        citations.append(citation)
    return citations


class HybridChromaQueryEngine:
    """
    Hybrid query engine combining dense vector search with keyword matching
    over a ChromaDB collection, fused via Reciprocal Rank Fusion.

    Generates query embeddings via an external embedding function and passes
    them as ``query_embeddings`` to ChromaDB, bypassing the collection's
    persisted (default) embedding function.
    """

    def __init__(
        self,
        chroma_collection,
        vector_store_index: VectorStoreIndex,
        collection_name: str,
        embedding_function: OpenAIEmbeddingFunction,
    ):
        self._collection = chroma_collection
        self._index = vector_store_index
        self._collection_name = collection_name
        self._embed = embedding_function

    def query(
        self,
        query_str: str,
        metadata_filters: Optional[Dict] = None,
        top_k: int = HYBRID_VECTOR_TOP_K,
    ) -> tuple[str, List[Dict[str, Any]]]:
        """
        Run hybrid retrieval: dense vector + keyword search with RRF fusion.

        Returns (formatted_text, citation_list).
        """
        where_clause = _build_chroma_where_clause(metadata_filters)

        query_embedding = self._embed([query_str])

        # --- Dense vector retrieval via ChromaDB ---
        vector_kwargs = {
            "query_embeddings": query_embedding,
            "n_results": top_k,
            "include": ["documents", "metadatas", "distances"],
        }
        if where_clause:
            vector_kwargs["where"] = where_clause

        try:
            vector_raw = self._collection.query(**vector_kwargs)
        except Exception as e:
            logger.error(
                "Dense vector search failed on '%s': %s", self._collection_name, e
            )
            vector_raw = {
                "ids": [[]],
                "documents": [[]],
                "metadatas": [[]],
                "distances": [[]],
            }

        vector_results = self._parse_chroma_results(vector_raw)

        # --- Keyword retrieval for domain-specific identifiers ---
        keyword_terms = _extract_keyword_terms(query_str)
        keyword_results = []
        if keyword_terms:
            for term in keyword_terms[:3]:
                kw_kwargs = {
                    "query_embeddings": query_embedding,
                    "n_results": HYBRID_KEYWORD_TOP_K,
                    "where_document": {"$contains": term},
                    "include": ["documents", "metadatas", "distances"],
                }
                if where_clause:
                    kw_kwargs["where"] = where_clause
                try:
                    kw_raw = self._collection.query(**kw_kwargs)
                    keyword_results.extend(self._parse_chroma_results(kw_raw))
                except Exception as e:
                    logger.error(
                        "Keyword search failed on '%s' for term '%s': %s",
                        self._collection_name,
                        term,
                        e,
                    )
                    continue

        # --- Reciprocal Rank Fusion ---
        lists_to_fuse = [vector_results]
        if keyword_results:
            lists_to_fuse.append(keyword_results)

        preferred_doc_type = _infer_query_doc_type_preference(query_str)
        fused = _reciprocal_rank_fusion(
            lists_to_fuse, preferred_doc_type=preferred_doc_type
        )
        fused = _rerank_results(query_str, fused, top_k)
        fused = _deduplicate_results(fused)

        formatted = _format_results_with_sources(fused, self._collection_name)
        citations = _extract_citation_metadata(fused, self._collection_name)
        return formatted, citations

    @staticmethod
    def _parse_chroma_results(raw: dict) -> List[Dict]:
        """Convert ChromaDB query() output into a flat list of doc dicts."""
        results = []
        ids = raw.get("ids", [[]])[0]
        docs = raw.get("documents", [[]])[0]
        metas = raw.get("metadatas", [[]])[0]
        distances = raw.get("distances", [[]])[0]
        for idx, (doc_id, text, meta) in enumerate(zip(ids, docs, metas)):
            distance = distances[idx] if idx < len(distances) else None
            similarity_score = None
            if isinstance(distance, (int, float)):
                # Convert distance to bounded similarity for easier log interpretation.
                similarity_score = 1.0 / (1.0 + float(distance))
            results.append(
                {
                    "id": doc_id,
                    "text": text or "",
                    "metadata": meta or {},
                    "distance": distance,
                    "similarity_score": similarity_score,
                }
            )
        return results


def get_chroma_tools() -> Dict[str, HybridChromaQueryEngine]:
    """
    Initializes ChromaDB client and returns HybridChromaQueryEngine instances
    for each specialized collection.

    Query embeddings are generated externally via the OpenAI embedding function
    (same model + dimensions used during indexing) and passed as
    ``query_embeddings`` to ChromaDB, avoiding conflicts with the collection's
    persisted default embedding function configuration.
    """
    chroma_client = chromadb.PersistentClient(path=str(CHROMADB_DIR))

    openai_ef = OpenAIEmbeddingFunction(
        api_key=OPENAI_API_KEY,
        model_name=EMBEDDING_MODEL,
        dimensions=EMBEDDING_DIMENSIONS,
    )

    tools: Dict[str, HybridChromaQueryEngine] = {}

    collections = [
        ("osha_regulations", "Search OSHA PSM and general industry regulations."),
        (
            "osha_enf",
            "Search historical OSHA enforcement actions, citations, and penalties.",
        ),
        (
            "csb",
            "Search Chemical Safety Board incident reports for factual timelines and causes.",
        ),
        (
            "csb_recommendations",
            "Search CSB safety recommendations and their statuses.",
        ),
    ]

    for collection_name, description in collections:
        try:
            chroma_collection = chroma_client.get_or_create_collection(collection_name)
            vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
            index = VectorStoreIndex.from_vector_store(vector_store)
            engine = HybridChromaQueryEngine(
                chroma_collection,
                index,
                collection_name,
                openai_ef,
            )
            tools[collection_name] = engine
        except Exception as e:
            logger.error("Could not initialize tool for %s: %s", collection_name, e)

    return tools
