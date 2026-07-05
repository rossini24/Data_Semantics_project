"""
01_collect_data.py
------------------
Raccoglie i dati da OpenAlex partendo dal seed paper di Stokes et al. 2020
ed espande la rete di citazioni fino a 2-3 salti.

Parametri configurabili:
  SEED_DOI       : DOI del paper di partenza
  MAX_HOPS       : numero massimo di salti (2 o 3)
  MIN_CITATIONS  : soglia minima di citazioni per includere un paper
  MAX_PAPERS     : numero massimo di paper totali da raccogliere
  EMAIL          : la tua email (per accedere al pool "polite" di OpenAlex, più veloce)

Output:
  data/papers.json    : metadati di tutti i paper raccolti
  data/authors.json   : metadati degli autori
  data/edges.json     : relazioni di citazione (paper_id -> paper_id)
"""

import requests
import json
import time
import os
from collections import deque

# ── CONFIGURAZIONE ────────────────────────────────────────────────────────────
SEED_DOI      = "10.1016/j.cell.2020.01.021"   # Stokes et al. 2020
MAX_HOPS      = 2                               # espandiamo a 2 salti per ora
MIN_CITATIONS = 5                               # escludi paper con meno di 5 citazioni
MAX_PAPERS    = 300                             # limite massimo paper totali
EMAIL         = "la_tua_email@campus.unimib.it" # CAMBIA con la tua email reale

BASE_URL      = "https://api.openalex.org"
HEADERS       = {"User-Agent": f"AntibioticKG-RAG/1.0 (mailto:{EMAIL})"}
OUTPUT_DIR    = "data"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── FUNZIONI DI SUPPORTO ──────────────────────────────────────────────────────

def get_work_by_doi(doi: str) -> dict | None:
    """Recupera un paper da OpenAlex tramite DOI."""
    url = f"{BASE_URL}/works/doi:{doi}"
    resp = requests.get(url, headers=HEADERS)
    if resp.status_code == 200:
        return resp.json()
    print(f"  [WARN] DOI non trovato: {doi} (status {resp.status_code})")
    return None


def get_work_by_id(openalex_id: str) -> dict | None:
    """Recupera un paper da OpenAlex tramite ID OpenAlex (es. W2741809807)."""
    url = f"{BASE_URL}/works/{openalex_id}"
    resp = requests.get(url, headers=HEADERS)
    if resp.status_code == 200:
        return resp.json()
    return None


def get_citing_works(openalex_id: str, min_citations: int = MIN_CITATIONS) -> list[dict]:
    """
    Recupera i paper che citano un dato paper (citing works).
    Filtra per numero minimo di citazioni e pagina automaticamente.
    """
    results = []
    page = 1
    per_page = 50

    while True:
        url = (
            f"{BASE_URL}/works"
            f"?filter=cites:{openalex_id},cited_by_count:>{min_citations}"
            f"&per-page={per_page}&page={page}"
            f"&select=id,doi,title,publication_year,cited_by_count,"
            f"authorships,concepts,referenced_works"
        )
        resp = requests.get(url, headers=HEADERS)
        if resp.status_code != 200:
            break

        data = resp.json()
        batch = data.get("results", [])
        if not batch:
            break

        results.extend(batch)
        total = data.get("meta", {}).get("count", 0)
        if len(results) >= total or len(results) >= MAX_PAPERS:
            break

        page += 1
        time.sleep(0.2)  # rispetta il rate limit OpenAlex

    return results


def extract_paper_info(work: dict) -> dict:
    """Estrae i campi rilevanti da un oggetto work di OpenAlex."""
    authorships = work.get("authorships", [])
    authors = []
    for a in authorships:
        author = a.get("author", {})
        institutions = [
            inst.get("display_name", "")
            for inst in a.get("institutions", [])
            if inst.get("display_name")
        ]
        authors.append({
            "id": author.get("id", "").replace("https://openalex.org/", ""),
            "name": author.get("display_name", ""),
            "institutions": institutions
        })

    concepts = [
        c.get("display_name", "")
        for c in work.get("concepts", [])
        if c.get("score", 0) > 0.3  # solo concetti rilevanti
    ]

    return {
        "id": work.get("id", "").replace("https://openalex.org/", ""),
        "doi": work.get("doi", ""),
        "title": work.get("title", ""),
        "year": work.get("publication_year"),
        "cited_by_count": work.get("cited_by_count", 0),
        "authors": authors,
        "concepts": concepts,
        "referenced_works": [
            r.replace("https://openalex.org/", "")
            for r in work.get("referenced_works", [])
        ]
    }


# ── RACCOLTA DATI CON BFS ─────────────────────────────────────────────────────

def collect_citation_network():
    print("=" * 60)
    print("AntibioticKG-RAG — Raccolta dati da OpenAlex")
    print(f"Seed: {SEED_DOI}")
    print(f"Max hops: {MAX_HOPS} | Min citations: {MIN_CITATIONS} | Max papers: {MAX_PAPERS}")
    print("=" * 60)

    papers = {}   # id -> paper_info
    edges  = []   # lista di (source_id, target_id)
    queue  = deque()  # (openalex_id, hop_level)
    visited = set()

    # Passo 1: recupera il seed paper
    print(f"\n[1/3] Recupero seed paper...")
    seed = get_work_by_doi(SEED_DOI)
    if not seed:
        print("ERRORE: impossibile recuperare il seed paper. Controlla la connessione e l'email.")
        return

    seed_info = extract_paper_info(seed)
    seed_id   = seed_info["id"]
    papers[seed_id] = seed_info
    visited.add(seed_id)
    queue.append((seed_id, 0))

    print(f"  Seed: {seed_info['title']} ({seed_info['year']})")
    print(f"  ID OpenAlex: {seed_id}")
    print(f"  Citato da: {seed_info['cited_by_count']} paper")

    # Passo 2: espansione BFS
    print(f"\n[2/3] Espansione rete di citazioni (BFS, max {MAX_HOPS} salti)...")
    while queue:
        current_id, hop = queue.popleft()

        if hop >= MAX_HOPS:
            continue
        if len(papers) >= MAX_PAPERS:
            print(f"  Raggiunto limite massimo di {MAX_PAPERS} paper.")
            break

        current_title = papers[current_id]["title"][:60]
        print(f"  [hop {hop+1}] Espando: {current_title}...")

        citing = get_citing_works(current_id)
        print(f"    → {len(citing)} paper citanti trovati")

        for work in citing:
            if len(papers) >= MAX_PAPERS:
                break

            paper_info = extract_paper_info(work)
            paper_id   = paper_info["id"]

            # aggiungi il paper se non già visto
            if paper_id not in visited:
                papers[paper_id] = paper_info
                visited.add(paper_id)
                queue.append((paper_id, hop + 1))

            # aggiungi il lato (paper_id cita current_id)
            edges.append({
                "source": paper_id,
                "target": current_id,
                "type": "cites"
            })

        time.sleep(0.3)

    # Passo 3: estrai autori unici
    print(f"\n[3/3] Estrazione autori...")
    authors = {}
    for paper in papers.values():
        for a in paper["authors"]:
            if a["id"] and a["id"] not in authors:
                authors[a["id"]] = {
                    "id": a["id"],
                    "name": a["name"],
                    "institutions": a["institutions"]
                }

    # Salvataggio
    with open(f"{OUTPUT_DIR}/papers.json", "w") as f:
        json.dump(list(papers.values()), f, indent=2, ensure_ascii=False)

    with open(f"{OUTPUT_DIR}/authors.json", "w") as f:
        json.dump(list(authors.values()), f, indent=2, ensure_ascii=False)

    with open(f"{OUTPUT_DIR}/edges.json", "w") as f:
        json.dump(edges, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print("RACCOLTA COMPLETATA")
    print(f"  Paper raccolti : {len(papers)}")
    print(f"  Autori unici   : {len(authors)}")
    print(f"  Relazioni cita : {len(edges)}")
    print(f"  File salvati in: {OUTPUT_DIR}/")
    print("=" * 60)


if __name__ == "__main__":
    collect_citation_network()
