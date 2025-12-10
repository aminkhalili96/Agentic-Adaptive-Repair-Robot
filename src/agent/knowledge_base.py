"""
RAG Knowledge Base for Industrial Repair SOPs.

Provides vector-based retrieval of Standard Operating Procedure (SOP)
documents to ground the repair agent's recommendations in verified data.

Usage:
    from src.agent.knowledge_base import consult_manual
    
    result = consult_manual("How to repair Steel?")
    # Returns: "Material: Steel | Tool: Grinder | Speed: 3000 RPM | Pressure: High"
"""

import numpy as np
from typing import List, Tuple, Optional

# ============ INDUSTRIAL SOP KNOWLEDGE BASE ============
# These are dummy rules for demonstration. In production, this would
# be loaded from a database, PDF documents, or knowledge management system.

SOP_DOCUMENTS = [
    {
        "id": "SOP-001",
        "content": "Material: Aluminum | Tool: Sanding Disc 80 | Speed: 1200 RPM | Pressure: Low",
        "material": "aluminum",
        "keywords": ["aluminum", "al", "light metal", "soft metal", "sanding"]
    },
    {
        "id": "SOP-002", 
        "content": "Material: Steel | Tool: Grinder | Speed: 3000 RPM | Pressure: High",
        "material": "steel",
        "keywords": ["steel", "iron", "hard metal", "grinder", "grinding"]
    },
    {
        "id": "SOP-003",
        "content": "Material: Composite | Tool: Polisher | Speed: 800 RPM | Pressure: Medium",
        "material": "composite",
        "keywords": ["composite", "carbon fiber", "fiberglass", "polymer", "polishing"]
    },
    {
        "id": "SOP-004",
        "content": "Defect: Rust | Treatment: Sand to bare metal, apply primer, then topcoat | Tool: Sanding Disc 80 | Speed: 1500 RPM",
        "material": "any",
        "keywords": ["rust", "corrosion", "oxidation", "oxide", "sanding"]
    },
    {
        "id": "SOP-005",
        "content": "Defect: Crack | Treatment: Clean, apply filler, sand smooth | Tool: Filler Applicator | Cure Time: 30 min",
        "material": "any",
        "keywords": ["crack", "fracture", "split", "filler", "repair"]
    },
    {
        "id": "SOP-006",
        "content": "Defect: Dent | Treatment: PDR (Paintless Dent Removal) or fill and sand | Tool: Body Hammer | Pressure: Medium",
        "material": "any",
        "keywords": ["dent", "depression", "impact damage", "hammer", "pdr"]
    },
]


# ============ VECTOR STORE (Simple Cosine Similarity) ============

class SimpleVectorStore:
    """
    Simple vector store using numpy for cosine similarity search.
    
    This is a lightweight alternative to FAISS for small knowledge bases.
    For production with larger datasets, use FAISS or a vector database.
    """
    
    def __init__(self, documents: List[dict]):
        self.documents = documents
        self.vectors = self._build_vectors()
    
    def _tokenize(self, text: str) -> set:
        """Simple word tokenization - extract alphanumeric words."""
        import re
        # Extract words, including handling of special cases like "RPM"
        words = re.findall(r'[a-zA-Z]+', text.lower())
        return set(words)
    
    def _build_vectors(self) -> np.ndarray:
        """Build TF vectors for all documents."""
        # Build vocabulary from all documents
        all_words = set()
        for doc in self.documents:
            all_words.update(self._tokenize(doc["content"]))
            all_words.update([kw.lower() for kw in doc.get("keywords", [])])
            # Also add material as a keyword
            if doc.get("material"):
                all_words.add(doc["material"].lower())
        
        self.vocabulary = sorted(list(all_words))
        self.word_to_idx = {w: i for i, w in enumerate(self.vocabulary)}
        
        # Build document vectors
        vectors = []
        for doc in self.documents:
            vec = np.zeros(len(self.vocabulary))
            words = self._tokenize(doc["content"])
            words.update([kw.lower() for kw in doc.get("keywords", [])])
            if doc.get("material"):
                words.add(doc["material"].lower())
            
            for word in words:
                if word in self.word_to_idx:
                    vec[self.word_to_idx[word]] = 1.0
            
            # Normalize
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            vectors.append(vec)
        
        return np.array(vectors)
    
    def _query_to_vector(self, query: str) -> np.ndarray:
        """Convert query to vector."""
        vec = np.zeros(len(self.vocabulary))
        words = self._tokenize(query)
        
        for word in words:
            if word in self.word_to_idx:
                vec[self.word_to_idx[word]] = 1.0
        
        # Normalize
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        
        return vec
    
    def search(self, query: str, top_k: int = 2) -> List[Tuple[dict, float]]:
        """
        Search for most relevant documents.
        
        Args:
            query: Search query string
            top_k: Number of results to return
            
        Returns:
            List of (document, similarity_score) tuples
        """
        query_vec = self._query_to_vector(query)
        
        # Compute cosine similarities
        similarities = np.dot(self.vectors, query_vec)
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            if similarities[idx] > 0:  # Only include matches
                results.append((self.documents[idx], float(similarities[idx])))
        
        return results


# ============ GLOBAL VECTOR STORE INSTANCE ============
_vector_store: Optional[SimpleVectorStore] = None


def get_vector_store() -> SimpleVectorStore:
    """Get or create the vector store singleton."""
    global _vector_store
    if _vector_store is None:
        _vector_store = SimpleVectorStore(SOP_DOCUMENTS)
    return _vector_store


# ============ MAIN API ============

def consult_manual(query: str) -> str:
    """
    Query the SOP knowledge base for repair specifications.
    
    This is the main function called by the Supervisor Agent when
    it needs to look up verified repair parameters.
    
    Args:
        query: Natural language query, e.g. "How to fix Steel?" or "Rust treatment"
        
    Returns:
        Formatted string with relevant SOP data, ready for LLM consumption
        
    Example:
        >>> consult_manual("Steel repair")
        "📋 **SOP Reference**
        
        According to SOP-002:
        Material: Steel | Tool: Grinder | Speed: 3000 RPM | Pressure: High
        
        (Confidence: 0.85)"
    """
    store = get_vector_store()
    results = store.search(query, top_k=2)
    
    if not results:
        return "📋 No matching SOP found. Using default repair parameters."
    
    # Format response
    lines = ["📋 **SOP Reference**\n"]
    
    for doc, score in results:
        lines.append(f"According to {doc['id']}:")
        lines.append(f"  {doc['content']}")
        lines.append(f"  _(Confidence: {score:.2f})_\n")
    
    return "\n".join(lines)


def search_sop(query: str, top_k: int = 2) -> List[dict]:
    """
    Raw search API for advanced use cases.
    
    Args:
        query: Search query
        top_k: Number of results
        
    Returns:
        List of matching SOP documents with scores
    """
    store = get_vector_store()
    results = store.search(query, top_k)
    
    return [
        {
            "id": doc["id"],
            "content": doc["content"],
            "material": doc.get("material", "any"),
            "score": score
        }
        for doc, score in results
    ]


# ============ TESTING ============
if __name__ == "__main__":
    print("=" * 60)
    print("RAG Knowledge Base Test")
    print("=" * 60)
    
    test_queries = [
        "How do we fix Steel?",
        "Aluminum repair settings",
        "Rust treatment procedure",
        "What tool for composite?",
        "Crack repair",
    ]
    
    for query in test_queries:
        print(f"\n🔍 Query: \"{query}\"")
        print("-" * 40)
        result = consult_manual(query)
        print(result)
    
    print("\n" + "=" * 60)
    print("Test Complete!")
    print("=" * 60)
