"""
01b_fetch_abstracts.py
----------------------
Recupera gli abstract completi dei paper da OpenAlex.
Va eseguito DOPO 01_collect_data.py.

OpenAlex restituisce gli abstract come "inverted index" (formato invertito),
questo script li ricostruisce come testo leggibile.

Input:  data/papers.json
Output: data/abstracts.json  {paper_id: "testo abstract"}
"""

import requests
import json
import time
import os

EMAIL    = "la_tua_email@campus.unimib.it"  # CAMBIA con la tua email reale
BASE_URL = "https://api.openalex.org"
HEADERS  = {"User-Agent": f"AntibioticKG-RAG/1.0 (mailto:{EMAIL})"}

def reconstruct_abstract(inverted_index: dict) -> str:
    """
    OpenAlex fornisce gli abstract come indice invertito:
    {"word": [posizione1, posizione2], ...}
    Questa funzione ricostruisce il testo originale.
    """
    if not inverted_index:
        return ""
    # trova la lunghezza massima
    max_pos = max(pos for positions in inverted_index.values() for pos in positions)
    words = [""] * (max_pos + 1)
    for word, positions in inverted_index.items():
        for pos in positions:
            words[pos] = word
    return " ".join(w for w in words if w)


def fetch_abstracts():
    print("=" * 60)
    print("AntibioticKG-RAG — Fetch abstract da OpenAlex")
    print("=" * 60)

    with open("data/papers.json") as f:
        papers = json.load(f)

    abstracts = {}
    total = len(papers)

    for i, paper in enumerate(papers):
        paper_id = paper["id"]
        print(f"  [{i+1}/{total}] {paper.get('title', '')[:50]}...")

        url = f"{BASE_URL}/works/{paper_id}?select=id,abstract_inverted_index"
        resp = requests.get(url, headers=HEADERS)

        if resp.status_code == 200:
            data = resp.json()
            inv_index = data.get("abstract_inverted_index")
            abstract  = reconstruct_abstract(inv_index) if inv_index else ""
            abstracts[paper_id] = abstract
            if abstract:
                print(f"    ✓ Abstract: {len(abstract)} caratteri")
            else:
                print(f"    — Abstract non disponibile")
        else:
            print(f"    [WARN] Status {resp.status_code}")
            abstracts[paper_id] = ""

        time.sleep(0.2)  # rispetta rate limit

    with open("data/abstracts.json", "w") as f:
        json.dump(abstracts, f, indent=2, ensure_ascii=False)

    available = sum(1 for v in abstracts.values() if v)
    print("\n" + "=" * 60)
    print(f"Abstract recuperati: {available}/{total}")
    print(f"File salvato: data/abstracts.json")
    print("=" * 60)


if __name__ == "__main__":
    fetch_abstracts()
