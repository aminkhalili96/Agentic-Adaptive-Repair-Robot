"""
Test script for the RAG Knowledge Base.

Verifies:
1. Knowledge base loads correctly
2. Vector search returns relevant results
3. consult_manual tool returns SOP data
"""

import pytest
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agent.knowledge_base import (
    consult_manual,
    search_sop,
    get_vector_store,
    SOP_DOCUMENTS
)


class TestKnowledgeBase:
    """Test suite for the RAG knowledge base."""
    
    def test_sop_documents_loaded(self):
        """Verify SOP documents are properly defined."""
        assert len(SOP_DOCUMENTS) >= 3, "Should have at least 3 SOP documents"
        
        # Check required fields
        for doc in SOP_DOCUMENTS:
            assert "id" in doc, "Each doc should have an id"
            assert "content" in doc, "Each doc should have content"
            assert "keywords" in doc, "Each doc should have keywords"
    
    def test_vector_store_initialization(self):
        """Test that vector store initializes correctly."""
        store = get_vector_store()
        
        assert store is not None
        assert len(store.documents) > 0
        assert store.vectors is not None
        assert len(store.vocabulary) > 0
    
    def test_steel_query_returns_grinder(self):
        """Test that 'Steel' query returns correct SOP with Grinder."""
        result = consult_manual("How do we fix Steel?")
        
        assert "Steel" in result
        assert "Grinder" in result
        assert "3000 RPM" in result
        assert "High" in result
    
    def test_aluminum_query_returns_sanding(self):
        """Test that 'Aluminum' query returns correct SOP."""
        result = consult_manual("Aluminum repair settings")
        
        assert "Aluminum" in result
        assert "Sanding Disc" in result
        assert "1200 RPM" in result
        assert "Low" in result
    
    def test_composite_query_returns_polisher(self):
        """Test that 'Composite' query returns correct SOP."""
        result = consult_manual("Composite material repair")
        
        assert "Composite" in result
        assert "Polisher" in result
        assert "800 RPM" in result
    
    def test_rust_query_returns_treatment(self):
        """Test that 'Rust' query returns treatment procedure."""
        result = consult_manual("rust treatment")
        
        assert "Rust" in result
        # Should mention sanding and/or primer
        assert "Sand" in result or "sand" in result or "Sanding" in result
    
    def test_search_sop_returns_structured_results(self):
        """Test raw search API returns structured data."""
        results = search_sop("Steel repair", top_k=1)
        
        assert len(results) > 0
        assert "id" in results[0]
        assert "content" in results[0]
        assert "score" in results[0]
        assert results[0]["score"] > 0
    
    def test_no_match_returns_fallback(self):
        """Test that unrelated query returns appropriate message."""
        result = consult_manual("completely unrelated xyz query 12345")
        
        # Should return something (fallback or no match message)
        assert len(result) > 0
        assert "SOP" in result or "No matching" in result


class TestConsultManualIntegration:
    """Integration tests for the consult_manual tool."""
    
    def test_consult_manual_format(self):
        """Test that consult_manual returns properly formatted response."""
        result = consult_manual("Steel")
        
        # Should have SOP reference header
        assert "SOP" in result
        # Should have confidence score
        assert "Confidence" in result or "confidence" in result.lower()
    
    def test_multiple_queries_consistent(self):
        """Test that repeated queries return consistent results."""
        result1 = consult_manual("Steel repair")
        result2 = consult_manual("Steel repair")
        
        # Results should be identical (deterministic)
        assert result1 == result2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
