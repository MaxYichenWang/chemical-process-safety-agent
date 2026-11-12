from unittest.mock import MagicMock, patch

from src.agentic_rag.tools import (
    HybridChromaQueryEngine,
    _CSB_INCIDENT_NAME_MAP,
    _extract_citation_ids,
    _extract_enforcement_standard_number,
    _extract_establishment_at_clause,
    _extract_keyword_terms,
    _format_results_with_sources,
    _infer_query_doc_type_preference,
    _reciprocal_rank_fusion,
    _rerank_results,
    get_chroma_tools,
)


def test_extract_keyword_terms():
    """Test extraction of OSHA standard numbers from query text."""
    terms = _extract_keyword_terms("What are the requirements of 1910.119 and 1910.147?")
    assert "1910.119" in terms
    assert "1910.147" in terms

    terms = _extract_keyword_terms("Tell me about pressure vessel safety")
    assert terms == []


def test_extract_citation_ids_and_establishment():
    query = (
        "What violation was cited in citation 01001 under standard 19100212 A03 II "
        "at MORRIS BEAN & CO in Yellville, AR?"
    )
    assert _extract_citation_ids(query) == ["01001"]
    assert _extract_enforcement_standard_number(query) == "19100212 A03 II"
    assert _extract_establishment_at_clause(query) == "MORRIS BEAN & CO"


def test_extract_keyword_terms_enforcement_citation():
    query = (
        "What violation was cited in citation 01001B under standard 19260451 A02 "
        "at ABBOTT MASONRY COMPANY in Virginia?"
    )
    terms = _extract_keyword_terms(query)
    assert "01001B" in terms
    assert "19260451 A02" in terms
    assert "ABBOTT MASONRY COMPANY" in terms


def test_extract_keyword_terms_accident_chemical_facility_year():
    terms = _extract_keyword_terms(
        "What happened in OSHA accident report 14217376 at AB Specialty Silicones plant in 2019 with vinyl chloride?"
    )
    assert "14217376" in terms
    assert "2019" in terms
    assert "vinyl chloride" in terms
    assert any("silicones plant" in t.lower() for t in terms)


def test_reciprocal_rank_fusion():
    """Test RRF merging of two result lists."""
    list_a = [
        {"id": "doc1", "text": "A", "metadata": {}},
        {"id": "doc2", "text": "B", "metadata": {}},
    ]
    list_b = [
        {"id": "doc2", "text": "B", "metadata": {}},
        {"id": "doc3", "text": "C", "metadata": {}},
    ]

    fused = _reciprocal_rank_fusion([list_a, list_b])
    ids = [d["id"] for d in fused]
    assert "doc2" in ids
    assert ids.index("doc2") == 0


def test_rerank_results_lexical_fallback(monkeypatch):
    docs = [
        {"id": "d1", "text": "general safety overview"},
        {"id": "d2", "text": "vinyl chloride accident report 14217376"},
    ]
    monkeypatch.setattr("src.agentic_rag.tools.HYBRID_RERANK_ENABLED", True)
    monkeypatch.setattr("src.agentic_rag.tools._RERANKER", None)
    monkeypatch.setattr("src.agentic_rag.tools._RERANKER_LOAD_ATTEMPTED", True)

    reranked = _rerank_results("14217376 vinyl chloride accident", docs, top_k=2)
    assert reranked[0]["id"] == "d2"


def test_format_results_with_sources_empty():
    """Test formatting when no results are found."""
    result = _format_results_with_sources([], "csb")
    assert "No relevant documents found" in result


def test_format_results_with_sources_osha_reg():
    """Test formatting includes OSHA regulation metadata."""
    docs = [{"id": "1", "text": "Mechanical integrity", "metadata": {"standard_number": "1910.119", "section_id": "1910.119(j)"}}]
    result = _format_results_with_sources(docs, "osha_regulations")
    assert "1910.119" in result
    assert "1910.119(j)" in result
    assert "Mechanical integrity" in result


def test_format_results_with_sources_csb():
    """Test formatting includes CSB report metadata."""
    docs = [{"id": "1", "text": "Explosion occurred.", "metadata": {"report_name": "Incident Report", "page_start": 5, "page_end": 7}}]
    result = _format_results_with_sources(docs, "csb")
    assert "Incident Report" in result
    assert "pp. 5-7" in result
    assert "Explosion occurred." in result


@patch("src.agentic_rag.tools.OpenAIEmbeddingFunction")
@patch("src.agentic_rag.tools.chromadb.PersistentClient")
@patch("src.agentic_rag.tools.VectorStoreIndex.from_vector_store")
def test_get_chroma_tools(mock_from_vector_store, mock_client, mock_openai_ef):
    """Test that get_chroma_tools initializes HybridChromaQueryEngine instances."""
    mock_instance = mock_client.return_value
    mock_instance.get_or_create_collection.return_value = MagicMock()

    tools = get_chroma_tools()

    assert len(tools) == 4
    assert "osha_regulations" in tools
    assert "osha_enf" in tools
    assert "csb" in tools
    assert "csb_recommendations" in tools

    for tool in tools.values():
        assert isinstance(tool, HybridChromaQueryEngine)

    assert mock_instance.get_or_create_collection.call_count == 4
    mock_openai_ef.assert_called_once()


def test_hybrid_query_engine_parse_chroma_results():
    """Test parsing of ChromaDB raw query output."""
    raw = {
        "ids": [["id1", "id2"]],
        "documents": [["text1", "text2"]],
        "metadatas": [[{"source": "a.pdf"}, {"source": "b.pdf"}]],
        "distances": [[0.25, 0.5]],
    }
    results = HybridChromaQueryEngine._parse_chroma_results(raw)
    assert len(results) == 2
    assert results[0]["id"] == "id1"
    assert results[0]["text"] == "text1"
    assert results[1]["metadata"]["source"] == "b.pdf"
    assert results[0]["distance"] == 0.25
    assert results[0]["similarity_score"] == 0.8
    assert round(results[1]["similarity_score"], 6) == 0.666667


# ---------------------------------------------------------------------------
# P3.1 — _infer_query_doc_type_preference
# ---------------------------------------------------------------------------


def test_infer_doc_type_violation_keywords():
    assert _infer_query_doc_type_preference("How many willful violations were issued?") == "enforcement_violation"
    assert _infer_query_doc_type_preference("List all citations for serious penalty") == "enforcement_violation"
    assert _infer_query_doc_type_preference("Find repeat citations at the refinery") == "enforcement_violation"


def test_infer_doc_type_general_duty():
    assert _infer_query_doc_type_preference("Was there a general duty clause violation?") == "enforcement_general_duty"
    assert _infer_query_doc_type_preference("Section 5(a)(1) violation at plant") == "enforcement_general_duty"


def test_infer_doc_type_accident():
    assert _infer_query_doc_type_preference("How many fatalities occurred in 2020?") == "enforcement_accident"
    assert _infer_query_doc_type_preference("worker killed at chemical plant") == "enforcement_accident"


def test_infer_doc_type_none():
    assert _infer_query_doc_type_preference("What are the PSM requirements for pressure relief?") is None
    assert _infer_query_doc_type_preference("Tell me about OSHA regulations") is None


# ---------------------------------------------------------------------------
# P3.1 — _reciprocal_rank_fusion with doc_type_boost
# ---------------------------------------------------------------------------


def test_reciprocal_rank_fusion_with_doc_type_boost():
    """Violation-type chunks should be ranked higher when preferred_doc_type is set."""
    regulation_chunks = [
        {"id": f"reg_{i}", "text": "regulation text", "metadata": {"doc_type": "regulation"}}
        for i in range(5)
    ]
    violation_chunk = {"id": "viol_1", "text": "violation data", "metadata": {"doc_type": "enforcement_violation"}}

    # Place violation chunk at rank 4 (behind 4 regulation chunks) in vector results
    list_a = regulation_chunks[:4] + [violation_chunk]
    fused_plain = _reciprocal_rank_fusion([list_a])
    fused_boosted = _reciprocal_rank_fusion([list_a], preferred_doc_type="enforcement_violation", doc_type_boost=3.0)

    # With no boost, violation chunk ranks last
    plain_ids = [d["id"] for d in fused_plain]
    assert plain_ids.index("viol_1") == 4

    # With boost, violation chunk moves to rank 0
    boosted_ids = [d["id"] for d in fused_boosted]
    assert boosted_ids.index("viol_1") == 0


def test_reciprocal_rank_fusion_boost_no_match():
    """Boost should not alter ranking when preferred_doc_type doesn't match any chunk."""
    docs = [
        {"id": "a", "text": "", "metadata": {"doc_type": "regulation"}},
        {"id": "b", "text": "", "metadata": {"doc_type": "regulation"}},
    ]
    plain = _reciprocal_rank_fusion([docs])
    boosted = _reciprocal_rank_fusion([docs], preferred_doc_type="enforcement_violation", doc_type_boost=5.0)
    assert [d["id"] for d in plain] == [d["id"] for d in boosted]


# ---------------------------------------------------------------------------
# P3.2 — CSB incident name keyword extraction
# ---------------------------------------------------------------------------


def test_extract_keyword_terms_csb_incident_name():
    """Canonical incident name tokens should be added when a known incident is mentioned."""
    terms = _extract_keyword_terms("What happened in the Watson Grinding explosion?")
    # Canonical name: Watson_Grinding_and_Manufacturing → tokens: Watson, Grinding, and, Manufacturing
    assert any("Watson" in t for t in terms)
    assert any("Grinding" in t for t in terms)


def test_extract_keyword_terms_csb_ab_specialty():
    terms = _extract_keyword_terms("Tell me about the AB Specialty Silicones incident")
    assert any("AB" in t or "Specialty" in t or "Silicones" in t for t in terms)


def test_csb_incident_name_map_not_empty():
    assert len(_CSB_INCIDENT_NAME_MAP) > 0
    # All values should be non-empty strings
    for key, val in _CSB_INCIDENT_NAME_MAP.items():
        assert isinstance(key, str) and key
        assert isinstance(val, str) and val
