# AntibioticKG-RAG

Graph RAG vs Document RAG vs Parametric LLM  
Question Answering su reti di citazioni scientifiche — dominio: scoperta di antibiotici  
Corso: Data Semantics 25/26

---

## Struttura del progetto

```
antibiotic_kg_rag/
├── 01_collect_data.py      # Raccolta dati da OpenAlex
├── 01b_fetch_abstracts.py  # Fetch abstract completi (esegui dopo 01)
├── 02_build_kg.py          # Costruzione Knowledge Graph RDF
├── 03_systems.py           # I tre sistemi di QA
├── 04_evaluation.py        # Confronto sulle 30 domande
├── 05_analyze.py           # Analisi risultati e grafico finale
├── data/                   # Creata da script 01
└── results/                # Creata da script 04
```

---

## Setup

```bash
pip install requests rdflib anthropic sentence-transformers numpy matplotlib
export ANTHROPIC_API_KEY=sk-...
```

---

## Ordine di esecuzione

### Passo 1 — Raccolta dati (~20 min, si lascia girare)
Apri `01_collect_data.py`, cambia `EMAIL` con la tua email reale, poi:
```bash
python 01_collect_data.py
```
Produce: `data/papers.json`, `data/authors.json`, `data/edges.json`

### Passo 1b — Fetch abstract (~15 min)
Apri `01b_fetch_abstracts.py`, cambia `EMAIL`, poi:
```bash
python 01b_fetch_abstracts.py
```
Produce: `data/abstracts.json`

### Passo 2 — Costruzione Knowledge Graph
```bash
python 02_build_kg.py
```
Produce: `data/antibiotic_kg.ttl`

### Passo 3 — Verifica rapida (opzionale ma consigliata)
```python
from rdflib import Graph
g = Graph()
g.parse("data/antibiotic_kg.ttl", format="turtle")
print(f"Triple totali: {len(g)}")
```

### Passo 4 — Valutazione comparativa
```bash
python 04_evaluation.py
```
Produce: `results/evaluation_results.json`, `results/scores.csv`

**IMPORTANTE**: dopo l'esecuzione, aprire `scores.csv` e compilare
manualmente `sys1_correct`, `sys2_correct`, `sys3_correct` (0/1)
confrontando con le risposte in `evaluation_results.json`.

Tassonomia errori da usare:
- System 1: `invention` / `entity_confusion` / `incomplete`
- System 2: `chunk_not_retrieved` / `not_in_chunk` / `poor_synthesis`
- System 3: `malformed_query` / `wrong_relation` / `data_absent`

### Passo 5 — Analisi e grafico
```bash
python 05_analyze.py
```
Produce: `results/accuracy_vs_hop.png`

---

## Modello embedding (System 2)

Usiamo `sentence-transformers/all-MiniLM-L6-v2` come baseline leggera (~80MB, no GPU).
Per risultati migliori, cambiare `EMBED_MODEL` in `03_systems.py` con `BAAI/bge-m3`
(riferimento: arxiv:2402.03216, suggerito dal prof Pozzi).

---

## Domande di valutazione

Le 30 domande in `04_evaluation.py` hanno ground truth parzialmente template.
Dopo aver eseguito lo script 01 e avere i dati reali in `data/papers.json`,
completare i campi `ground_truth` con i valori veri prima della valutazione.

Distribuzione:
- 10 single-hop  (S01–S10)
- 10 double-hop  (D01–D10)
- 10 triple-hop  (T01–T10)

---

## Valutazione inter-annotatore

Entrambi i membri del gruppo annotano **indipendentemente** le stesse
risposte su una copia separata del CSV, poi confrontano i disaccordi
prima di fissare i punteggi finali.
