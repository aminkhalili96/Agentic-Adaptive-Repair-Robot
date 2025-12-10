# RAG Knowledge Base

The AARR system includes a Retrieval-Augmented Generation (RAG) knowledge base to ground repair recommendations in verified Standard Operating Procedures (SOPs).

---

## Overview

Instead of relying solely on LLM inference for repair parameters, the Supervisor Agent can now **consult the SOP manual** to retrieve verified specifications for:

- **Material-specific settings** (Aluminum, Steel, Composite)
- **Defect-specific treatments** (Rust, Crack, Dent)
- **Tool, speed, and pressure parameters**

```
┌─────────────────┐     Query      ┌─────────────────┐
│  User: "How do  │ ──────────────▶│  Knowledge Base │
│  we fix Steel?" │                │  (Vector Search)│
└─────────────────┘                └────────┬────────┘
                                            │
                                   SOP-002: Steel | Grinder | 3000 RPM
                                            │
                                            ▼
┌─────────────────────────────────────────────────────┐
│  Agent: "According to the SOP, for Steel I am       │
│  setting the robot to 3000 RPM with High pressure   │
│  using a Grinder."                                  │
└─────────────────────────────────────────────────────┘
```

---

## Architecture

### Files

| File | Description |
|------|-------------|
| `src/agent/knowledge_base.py` | SOP documents + vector search |
| `src/test_knowledge_base.py` | Unit tests for RAG system |

### Components

1. **SOP_DOCUMENTS** - List of industrial repair rules
2. **SimpleVectorStore** - Numpy-based cosine similarity search
3. **consult_manual()** - Main API for the Supervisor Agent

---

## SOP Documents

The knowledge base contains the following rules:

| ID | Material/Defect | Tool | Speed | Pressure |
|----|-----------------|------|-------|----------|
| SOP-001 | Aluminum | Sanding Disc 80 | 1200 RPM | Low |
| SOP-002 | Steel | Grinder | 3000 RPM | High |
| SOP-003 | Composite | Polisher | 800 RPM | Medium |
| SOP-004 | Rust | Sanding Disc 80 | 1500 RPM | - |
| SOP-005 | Crack | Filler Applicator | - | - |
| SOP-006 | Dent | Body Hammer | - | Medium |

---

## Usage

### Agent Tool

The Supervisor Agent automatically calls `consult_manual` when:
- Planning repairs for specific materials
- User asks about tool/speed/pressure settings
- Processing defect-specific treatments

```python
# Automatic via OpenAI function calling
# User: "How do we fix Steel?"
# Agent calls: consult_manual(query="Steel repair")
```

### Direct API

```python
from src.agent.knowledge_base import consult_manual

result = consult_manual("Steel repair")
print(result)
# Output:
# 📋 **SOP Reference**
# 
# According to SOP-002:
#   Material: Steel | Tool: Grinder | Speed: 3000 RPM | Pressure: High
#   _(Confidence: 0.85)_
```

---

## Testing

Run the test suite:

```bash
cd /Users/amin/dev/Robotic\ AI
python -m pytest src/test_knowledge_base.py -v
```

Or test directly:

```bash
python src/agent/knowledge_base.py
```

---

## Extending the Knowledge Base

To add new SOPs, edit `SOP_DOCUMENTS` in `knowledge_base.py`:

```python
SOP_DOCUMENTS = [
    # ... existing entries ...
    {
        "id": "SOP-007",
        "content": "Material: Titanium | Tool: Diamond Disc | Speed: 2000 RPM | Pressure: Medium",
        "material": "titanium",
        "keywords": ["titanium", "ti", "aerospace", "diamond"]
    },
]
```

For larger knowledge bases, consider migrating to:
- **FAISS** via `langchain-community`
- **Pinecone** or **Chroma** for production
