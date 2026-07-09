"""
03_systems.py
-------------
Implementa i tre sistemi di question answering:
  System 1 — Parametric LLM   (nessun contesto esterno)
  System 2 — Document RAG     (sentence-transformers embeddings su abstract)
  System 3 — Graph RAG        (Text-to-SPARQL su Knowledge Graph RDF)

Requisiti:
  pip install anthropic sentence-transformers rdflib numpy
  Variabile d'ambiente: ANTHROPIC_API_KEY=sk-...

Modello embedding usato: all-MiniLM-L6-v2 (leggero, ~80MB, no GPU richiesta)
Riferimento: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
Come suggerito dal prof, alternativa più potente: BGE-M3 (arxiv:2402.03216)
"""

import os
import json
import numpy as np
from dotenv import load_dotenv
from google import genai
from google.genai import types
from sentence_transformers import SentenceTransformer
from rdflib import Graph

load_dotenv()  # legge il file .env e carica GOOGLE_API_KEY nell'ambiente

# ── CONFIGURAZIONE ────────────────────────────────────────────────────────────
MODEL          = "gemini-2.5-flash"
MAX_TOKENS     = 1024
KG_PATH        = "data/antibiotic_kg_inferred.ttl"
PAPERS_PATH    = "data/papers.json"
ABSTRACTS_PATH = "data/abstracts.json"

# BGE-M3 commentato per velocità di sviluppo — MiniLM attivo come richiesto:
# EMBED_MODEL = "BAAI/bge-m3"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))


# ═══════════════════════════════════════════════════════════════════════════════
# SYSTEM 1 — PARAMETRIC LLM
# Nessun contesto esterno: risponde solo con la memoria parametrica del modello.
# Baseline per misurare quante allucinazioni produce senza dati esterni.
# ═══════════════════════════════════════════════════════════════════════════════

def system1_parametric_llm(question: str) -> str:
    """Risponde alla domanda senza alcun contesto esterno."""
    try:
        response = client.models.generate_content(
            model=MODEL,
            contents=question,
            config=types.GenerateContentConfig(
                system_instruction=(
                    "You are a scientific assistant. Answer the following question "
                    "about scientific papers and authors as accurately as possible. "
                    "If you are not sure, say so explicitly — do not invent facts."
                ),
                max_output_tokens=MAX_TOKENS,
                temperature=0,
                thinking_config=types.ThinkingConfig(thinking_budget=0),
            )
        )
        return response.text
    except Exception as e:
        return f"[ERRORE API] {e}"

# ═══════════════════════════════════════════════════════════════════════════════
# SYSTEM 2 — DOCUMENT RAG con sentence-transformers
#
# Invece di TF-IDF (approccio keyword-based), usiamo embeddings densi
# prodotti da all-MiniLM-L6-v2: ogni testo viene trasformato in un vettore
# che cattura il significato semantico, non solo le parole esatte.
# La similarità tra la domanda e i chunk viene calcolata con cosine similarity.
#
# Riferimento consigliato dal prof: BGE-M3 (arxiv:2402.03216)
# Usiamo all-MiniLM-L6-v2 come baseline leggera (~80MB, no GPU).
# ═══════════════════════════════════════════════════════════════════════════════

class DocumentRAG:
    def __init__(self, papers_path: str = PAPERS_PATH,
                 abstracts_path: str = ABSTRACTS_PATH):
        print(f"[System 2] Caricamento modello embedding: {EMBED_MODEL}...")
        self.model = SentenceTransformer(EMBED_MODEL)

        with open(papers_path, encoding="utf-8") as f:
            papers = json.load(f)

        # Carica gli abstract se disponibili
        abstracts = {}
        if os.path.exists(abstracts_path):
            with open(abstracts_path, encoding="utf-8") as f:
                abstracts = json.load(f)
            print(f"  Abstract disponibili: {sum(1 for v in abstracts.values() if v)}/{len(abstracts)}")
        else:
            print("  [WARN] abstracts.json non trovato — usa solo titoli e concetti")
            print("         Esegui prima 01b_fetch_abstracts.py")

        # Costruiamo i chunk: titolo + abstract (se disponibile) + concetti
        self.chunks = []
        for p in papers:
            if not p.get("title"):
                continue

            text = p["title"]
            abstract = abstracts.get(p["id"], "")
            if abstract:
                text += "\n" + abstract
            if p.get("concepts"):
                text += "\nTopics: " + ", ".join(p["concepts"][:5])
            if p.get("year"):
                text += f" ({p['year']})"

            self.chunks.append({
                "text":      text,
                "paper_id":  p["id"],
                "title":     p["title"],
                "abstract":  abstract,
                "authors":   [a["name"] for a in p.get("authors", [])],
                "year":      p.get("year"),
                "doi":       p.get("doi", "")
            })

        # Calcola gli embeddings di tutti i chunk una sola volta
        print(f"  Calcolo embeddings per {len(self.chunks)} chunk...")
        texts = [c["text"] for c in self.chunks]
        self.embeddings = self.model.encode(texts, show_progress_bar=True,
                                            convert_to_numpy=True)
        print(f"  Indice costruito: shape {self.embeddings.shape}")

    def retrieve(self, question: str, top_k: int = 5) -> list[dict]:
        """Recupera i top_k chunk semanticamente più simili alla domanda."""
        q_embedding = self.model.encode([question], convert_to_numpy=True)
        # cosine similarity manuale (evita dipendenza da sklearn)
        norms  = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        q_norm = np.linalg.norm(q_embedding)
        scores = (self.embeddings @ q_embedding.T).flatten() / (norms.flatten() * q_norm + 1e-9)
        top_indices = scores.argsort()[-top_k:][::-1]
        return [self.chunks[i] for i in top_indices if scores[i] > 0.1]

    def answer(self, question: str, top_k: int = 5) -> dict:
        """Recupera i chunk rilevanti e genera la risposta con l'LLM."""
        retrieved = self.retrieve(question, top_k)
        if not retrieved:
            return {
                "answer": "No relevant documents found in the index.",
                "retrieved_chunks": []
            }

        context = "\n\n".join([
            f"Paper: {c['title']}\n"
            f"Authors: {', '.join(c['authors'][:3])}\n"
            f"Year: {c.get('year', 'N/A')}\n"
            + (f"Abstract: {c['abstract'][:500]}\n" if c.get('abstract') else "")
            + f"Topics: {', '.join(c.get('text','').split('Topics:')[-1].split(',')[:3]) if 'Topics:' in c.get('text','') else 'N/A'}"
            for c in retrieved
        ])

        try:
            response = client.models.generate_content(
                model=MODEL,
                contents=f"Context:\n{context}\n\nQuestion: {question}",
                config=types.GenerateContentConfig(
                    system_instruction=(
                        "You are a scientific assistant. Answer the question using ONLY "
                        "the information provided in the context below. "
                        "If the context does not contain enough information, say so explicitly."
                    ),
                    max_output_tokens=MAX_TOKENS,
                    temperature=0,
                    thinking_config=types.ThinkingConfig(thinking_budget=0),
                )
            )
            answer_text = response.text
        except Exception as e:
            answer_text = f"[ERRORE API] {e}"

        return {
            "answer":           answer_text,
            "retrieved_chunks": [c["title"] for c in retrieved]
        }

# ═══════════════════════════════════════════════════════════════════════════════
# SYSTEM 3 — GRAPH RAG
# L'LLM traduce la domanda in una query SPARQL, la esegue sul KG RDF,
# e verbalizza il risultato strutturato in linguaggio naturale.
# ═══════════════════════════════════════════════════════════════════════════════

ONTOLOGY_DESCRIPTION = """
PREFIX akg:  <http://antibiotickg.org/ontology#>
PREFIX ares: <http://antibiotickg.org/resource/>
PREFIX foaf: <http://xmlns.com/foaf/0.1/>

Classes:
  akg:Paper                 — a scientific paper
  akg:Author                — a researcher/author (subClassOf foaf:Person)
  akg:Institution           — a university or research centre
  akg:AcademicInstitution   — subclass of Institution, type=education
  akg:CompanyInstitution    — subclass of Institution, type=company
  akg:NonprofitInstitution  — subclass of Institution, type=nonprofit
  akg:Topic                 — a research concept or topic
  akg:Country               — a country (ISO two-letter code)

Object properties:
  akg:cites           (Paper → Paper)        — paper A cites paper B
  akg:citedBy         (Paper → Paper)        — inverse of cites: paper A is cited by paper B
  akg:hasAuthor       (Paper → Author)        — paper has an author
  akg:affiliatedWith  (Author → Institution)  — author belongs to institution
  akg:about           (Paper → Topic)         — paper covers a topic
  akg:locatedIn       (Institution → Country) — institution is located in country
  akg:coAuthorWith    (Author → Author)       — symmetric: two authors co-wrote a paper together

Datatype properties:
  akg:hasTitle            (Paper        → xsd:string)
  akg:hasYear             (Paper        → xsd:integer)
  akg:hasCitationCount    (Paper        → xsd:integer)
  akg:hasDOI              (Paper        → xsd:string)
  akg:hasPublicationType  (Paper        → xsd:string)   — e.g. article, review, book-chapter
  akg:isOpenAccess        (Paper        → xsd:boolean)
  akg:hasPrimaryTopic     (Paper        → xsd:string)   — main OpenAlex taxonomy topic
  akg:hasName             (Author       → xsd:string)
  akg:hasInstitutionName  (Institution  → xsd:string)
  akg:hasCountryCode      (Country      → xsd:string)   — two-letter ISO code, e.g. US, IT
  akg:hasTopicName        (Topic        → xsd:string)

Notes:
  - Use akg:citedBy directly instead of inverting akg:cites when the question
    asks "who cites paper X" — both directions are materialized in the graph.
  - Use akg:AcademicInstitution / akg:CompanyInstitution / akg:NonprofitInstitution
    directly in the query's rdf:type when the question asks about a specific
    institution type, instead of filtering on a separate type property.
  - akg:coAuthorWith is symmetric: querying in either direction returns the
    same pairs.
  - IMPORTANT: questions often refer to a paper informally by first author's
    surname and year (e.g. "the Stokes 2020 paper", "Smith et al. 2019") rather
    than by its exact title. The graph does NOT contain such informal aliases —
    only exact values extracted from OpenAlex (akg:hasTitle, akg:hasName,
    akg:hasYear). NEVER try to match the informal reference against akg:hasTitle.
    Instead, resolve it by author surname + year:
      ?paper akg:hasAuthor ?author .
      ?author akg:hasName ?authorName .
      FILTER (CONTAINS(?authorName, "Stokes"))
      ?paper akg:hasYear 2020 .
    This correctly identifies the paper without guessing its title.

Examples:
  Question: "Who are the authors of the Stokes 2020 paper on antibiotic discovery?"
  Correct query:
    SELECT DISTINCT ?authorName WHERE {
      ?paper akg:hasAuthor ?firstAuthor .
      ?firstAuthor akg:hasName ?firstAuthorName .
      FILTER (CONTAINS(?firstAuthorName, "Stokes"))
      ?paper akg:hasYear 2020 .
      ?paper akg:hasAuthor ?author .
      ?author akg:hasName ?authorName .
    }

  Question: "Which institutions are affiliated with authors of the Smith 2019 paper?"
  Correct query:
    SELECT DISTINCT ?institutionName WHERE {
      ?paper akg:hasAuthor ?namedAuthor .
      ?namedAuthor akg:hasName ?name .
      FILTER (CONTAINS(?name, "Smith"))
      ?paper akg:hasYear 2019 .
      ?paper akg:hasAuthor ?author .
      ?author akg:affiliatedWith ?institution .
      ?institution akg:hasInstitutionName ?institutionName .
    }

"""

class GraphRAG:
    def __init__(self, kg_path: str = KG_PATH):
        print("[System 3] Caricamento Knowledge Graph...")
        self.g = Graph()
        self.g.parse(kg_path, format="turtle")
        print(f"  Triple caricate: {len(self.g)}")

    def generate_sparql(self, question: str) -> str:
        """Usa l'LLM per tradurre la domanda in una query SPARQL."""
        try:
            response = client.models.generate_content(
                model=MODEL,
                contents=(
                    f"Ontology schema:\n{ONTOLOGY_DESCRIPTION}\n\n"
                    f"Question: {question}\n\n"
                    "Generate the SPARQL query:"
                ),
                config=types.GenerateContentConfig(
                    system_instruction=(
                        "You are a SPARQL expert. Given a question and an ontology schema, "
                        "generate a valid SPARQL SELECT query that answers the question. "
                        "Return ONLY the SPARQL query, no explanation, no markdown code blocks."
                    ),
                    max_output_tokens=MAX_TOKENS,
                    temperature=0,
                    thinking_config=types.ThinkingConfig(thinking_budget=0),
                )
            )
            return response.text.strip()
        except Exception as e:
            return f"[ERRORE API — impossibile generare SPARQL] {e}"

    def execute_sparql(self, query: str) -> list[dict]:
        """Esegue la query SPARQL sul grafo e restituisce i risultati."""
        try:
            results = self.g.query(query)
            rows = []
            for row in results:
                rows.append({
                    str(var): str(val)
                    for var, val in zip(results.vars, row)
                    if val is not None
                })
            return rows
        except Exception as e:
            return [{"error": str(e)}]

    def verbalize(self, question: str, sparql_results: list[dict]) -> str:
        """Verbalizza i risultati SPARQL in linguaggio naturale."""
        if not sparql_results:
            return "The query returned no results."
        if "error" in sparql_results[0]:
            return f"SPARQL execution error: {sparql_results[0]['error']}"

        results_text = json.dumps(sparql_results[:20], indent=2)
        try:
            response = client.models.generate_content(
                model=MODEL,
                contents=(
                    f"Question: {question}\n\n"
                    f"Query results:\n{results_text}\n\n"
                    "Answer:"
                ),
                config=types.GenerateContentConfig(
                    system_instruction=(
                        "You are a scientific assistant. Given structured query results "
                        "from a Knowledge Graph, provide a clear and concise natural language "
                        "answer to the original question. Use only the provided data."
                    ),
                    max_output_tokens=MAX_TOKENS,
                    temperature=0,
                    thinking_config=types.ThinkingConfig(thinking_budget=0),
                )
            )
            return response.text
        except Exception as e:
            return f"[ERRORE API — impossibile verbalizzare] {e}"

    def answer(self, question: str) -> dict:
        """Pipeline completa: domanda → SPARQL → esecuzione → risposta."""
        sparql_query   = self.generate_sparql(question)
        sparql_results = self.execute_sparql(sparql_query)
        answer_text    = self.verbalize(question, sparql_results)
        return {
            "answer":         answer_text,
            "sparql_query":   sparql_query,
            "sparql_results": sparql_results,
            "query_valid":    "error" not in (sparql_results[0] if sparql_results else {}),
            "results_count":  len(sparql_results)
        }


# ── TEST RAPIDO ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    test_question = "Who are the authors of the Stokes 2020 paper on antibiotic discovery?"

    print("\n" + "="*60)
    print(f"Domanda di test: {test_question}")
    print("="*60)

    print("\n[System 1 — Parametric LLM]")
    print(system1_parametric_llm(test_question))

    print("\n[System 2 — Document RAG]")
    doc_rag = DocumentRAG()
    ans2 = doc_rag.answer(test_question)
    print(ans2["answer"])
    print(f"Chunk recuperati: {ans2['retrieved_chunks']}")

    print("\n[System 3 — Graph RAG]")
    graph_rag = GraphRAG()
    ans3 = graph_rag.answer(test_question)
    print(f"Query SPARQL:\n{ans3['sparql_query']}")
    print(f"Risposta: {ans3['answer']}")
