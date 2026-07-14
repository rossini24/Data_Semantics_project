# AntibioticKG-RAG: Graph RAG vs Document RAG vs Parametric LLM

Knowledge Graph + Graph RAG pipeline for question answering over a scientific citation network, comparing three QA architectures — parametric LLM, Document RAG, and Graph RAG — on the antibiotic discovery domain.

## Overview

This project investigates how representing a citation network as a **Knowledge Graph (KG)** changes what can be reliably queried and answered compared to raw text retrieval or a model's parametric memory alone.

Using citation data seeded from Stokes et al. (2020), *A Deep Learning Approach to Antibiotic Discovery*, we:

- Build an OWL/RDF Knowledge Graph from OpenAlex citation data
- Apply OWL reasoning and materialize inferred triples (inverse/symmetric properties, subclassing)
- Support three independent QA systems:
  - **Parametric LLM** (no external grounding — baseline)
  - **Document RAG** (dense retrieval over paper abstracts)
  - **Graph RAG** (Text-to-SPARQL over the inferred KG)
- Evaluate all three on a 40-question, hop-graded benchmark

---

## Research Questions

1. To what extent does a Knowledge Graph representation of citation data improve QA accuracy compared to Document RAG and a parametric LLM, on factual and multi-hop questions?

2. How does answer accuracy for each system degrade as the number of relational hops required by the question increases (1-hop, 2-hop, 3-hop)?

---

## Dataset

**Source:** OpenAlex (free public API)

- Seed: Stokes et al. (2020), *Cell*
- Expansion: BFS, up to 2 hops
- Papers: 600
- Unique authors: 4,657
- Direct citation edges: 1,238
- Open access: 445 / 600
- Published after 2022: 357

---

## Architecture

```text
OpenAlex API
      │
      ▼
Data Collection + Abstract Retrieval
      │
      ▼
Knowledge Graph (OWL/RDF, Turtle)
      │
      ▼
OWL Reasoning (owlrl) — inference materialization
      │
      ├──────────────► System 1: Parametric LLM
      │
      ├──────────────► System 2: Document RAG
      │                    (embeddings + top-k retrieval)
      │
      └──────────────► System 3: Graph RAG
                           (NL → SPARQL → KG → verbalization)
```

---

## Knowledge Graph

### Main Classes

- Paper
- Author (`rdfs:subClassOf foaf:Person`)
- Institution
- Topic
- Country

### Institution Subclasses

- AcademicInstitution
- CompanyInstitution
- NonprofitInstitution

### Object Properties

- cites / citedBy (`owl:inverseOf`)
- hasAuthor
- affiliatedWith
- about
- locatedIn
- coAuthorWith (`owl:SymmetricProperty`)

**Graph size:** 103,840 asserted triples → 181,503 after OWL-RL inference materialization.

---

## Hybrid Question Answering

### Document RAG

Used for:

- Narrative synthesis
- Discursive, multi-paper summaries anchored to abstract text

Example:

```text
What strategies do researchers propose to combat antimicrobial
resistance beyond deep learning-based discovery?
```

↓

Synthesized answer grounded in retrieved abstracts.

### Graph RAG

Used for:

- Multi-hop relational chains
- Aggregations and comparisons over structured metadata

Example:

```text
Do review papers have more citations than article papers on average?
```

↓

```sparql
SELECT (AVG(?citations) AS ?avg) ?type WHERE { ... } GROUP BY ?type
```

↓

```text
Review papers average 242.5 citations vs 203.7 for articles.
```

---

## Evaluation

40 questions, HotpotQA-inspired (Bridge / Comparison / Narrative), graded across hop depths 1–3.

| System | Bridge | Comparison | Narrative | Total |
|---|---|---|---|---|
| Parametric LLM | 0% | 0% | 20% | 7.5% |
| Document RAG | 0% | 0% | 60% | 22.5% |
| **Graph RAG** | **73%** | **30%** | 7% | **37.5%** |

SPARQL validity (Graph RAG): 85% — the main bottleneck is query *semantic* correctness, not syntax generation.

No single architecture dominates across all question types: Graph RAG leads on structured relational queries, Document RAG wins on narrative synthesis, and the parametric LLM fails on all dataset-specific structured facts.

---

## Technology Stack

### Semantic Web

- OWL
- RDF / Turtle
- SPARQL
- RDFLib
- Protégé + HermiT (design-time) / owlrl (materialization)

### Python

- sentence-transformers
- NumPy
- Streamlit

### LLMs

- Gemini 2.5 Flash

---

## Installation

Clone the repository:

```bash
git clone https://github.com/rossini24/Data_Semantics_project.git
cd Data_Semantics_project
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate it:

Linux / macOS

```bash
source venv/bin/activate
```

Windows

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install requests rdflib owlrl google-genai sentence-transformers numpy matplotlib python-dotenv streamlit
```

Add a `.env` file with `GOOGLE_API_KEY=your-key-here`, then run scripts `01` through `05` in order (see docstrings for details).

---

## Limitations

- Comparison questions have uneven sample sizes across hop depths (as few as 2 at hop-3).
- Automated RAGAS scoring for narrative questions could not be reliably configured with Gemini; narrative answers are scored manually instead, following the same Faithfulness/Answer Relevance criteria.
- Graph RAG struggles with SPARQL aggregation patterns (UNION, nested GROUP BY) more than relation-chaining queries.
- Results are based on a single seed paper and domain; generalization to other citation networks is untested.

---

## Authors

- **Luca Rossini**
- **Lorenzo Zanotti**

Data Semantics — University of Milano-Bicocca

---

## License

This repository is released for academic and research purposes.
