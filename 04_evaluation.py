"""
04_evaluation.py
----------------
Esegue il confronto tra i tre sistemi sulle 30 domande di valutazione.

Le domande sono strutturate ispirandosi ai tipi di HotpotQA (Yang et al. 2018):
  - BRIDGE ENTITY: richiedono di attraversare due entità in catena
    (es. "in quale istituzione lavora il primo autore del paper che cita X?")
    → discriminano nettamente i tre sistemi: Graph RAG vince, Doc RAG fatica,
      LLM puro allucina
  - COMPARISON: richiedono di confrontare due entità su una proprietà
    (es. "quale dei due paper ha più citazioni?")
    → contro-caso: Document RAG può vincere perché sintetizza contesto,
      Graph RAG risponde in modo secco, LLM puro può cavarsela su paper famosi
  - NARRATIVE: richiedono comprensione discorsiva del testo
    → Document RAG dovrebbe vincere, Graph RAG non ha il contesto testuale

Divise anche per hop depth (1/2/3+) per rispondere alla RQ2:
  come degrada l'accuratezza di ciascun sistema all'aumentare dei salti?

Output:
  results/evaluation_results.json
  results/scores.csv
"""

import json
import csv
import os
from systems import system1_parametric_llm, DocumentRAG, GraphRAG

os.makedirs("results", exist_ok=True)

# ── SET DI DOMANDE ────────────────────────────────────────────────────────────
# NOTA: i ground_truth con [verificare...] vanno completati dopo aver
# eseguito 01_collect_data.py, guardando i valori reali in data/papers.json

QUESTIONS = [

    # ══════════════════════════════════════════════════════════════════════════
    # BRIDGE ENTITY — 1 HOP
    # Un solo salto: trovare un fatto diretto su un'entità.
    # Previsto: tutti e tre i sistemi dovrebbero cavarsela,
    # ma LLM puro inizia già a mostrare incertezza su fatti specifici.
    # ══════════════════════════════════════════════════════════════════════════
    {
        "id": "B1_01", "hop": 1, "type": "bridge",
        "question": "Who are the authors of the paper 'A Deep Learning Approach to Antibiotic Discovery' by Stokes et al. 2020?",
        "ground_truth": "Jonathan M. Stokes, Kevin Yang, Kyle Swanson, Wengong Jin, Andrés Cubillos-Ruiz, Nina M. Donghia, Craig R. MacNair, Shawn French, Lindsey A. Carfrae, Zohar Bloom-Ackermann, Victoria M. Tran, Anush Chiappino-Pepe, Ahmed H. Badran, Ian W. Andrews, Emma J. Chory, George M. Church, Eric D. Brown, Tommi S. Jaakkola, Regina Barzilay, James J. Collins"
    },
    {
        "id": "B1_02", "hop": 1, "type": "bridge",
        "question": "To which institution is James J. Collins affiliated?",
        "ground_truth": "Massachusetts Institute of Technology (MIT)"
    },
    {
        "id": "B1_03", "hop": 1, "type": "bridge",
        "question": "In what year was the Stokes 2020 antibiotic discovery paper published?",
        "ground_truth": "2020"
    },
    {
        "id": "B1_04", "hop": 1, "type": "bridge",
        "question": "What research topics does the Stokes 2020 paper cover according to OpenAlex?",
        "ground_truth": "[verificare in data/papers.json: campo concepts del seed paper]"
    },
    {
        "id": "B1_05", "hop": 1, "type": "bridge",
        "question": "How many times has the Stokes 2020 paper been cited?",
        "ground_truth": "[verificare in data/papers.json: campo cited_by_count del seed paper]"
    },

    # ══════════════════════════════════════════════════════════════════════════
    # BRIDGE ENTITY — 2 HOP
    # Due salti in catena: trovare A, poi trovare B tramite A.
    # Previsto: Graph RAG vince nettamente (segue la catena nel grafo),
    # Document RAG fatica (le due info raramente nello stesso chunk),
    # LLM puro allucina.
    # ══════════════════════════════════════════════════════════════════════════
    {
        "id": "B2_01", "hop": 2, "type": "bridge",
        "question": "To which institution is the first author of a paper that directly cites Stokes 2020 affiliated?",
        "ground_truth": "[verificare: edges.json → primo paper citante → autori → istituzione]"
    },
    {
        "id": "B2_02", "hop": 2, "type": "bridge",
        "question": "Which authors affiliated with MIT have published papers that cite Stokes 2020?",
        "ground_truth": "[verificare: papers citanti seed → autori → filtra per MIT]"
    },
    {
        "id": "B2_03", "hop": 2, "type": "bridge",
        "question": "Which research topics appear in papers that cite Stokes 2020?",
        "ground_truth": "[verificare: papers citanti seed → aggregare concepts]"
    },
    {
        "id": "B2_04", "hop": 2, "type": "bridge",
        "question": "Which papers citing Stokes 2020 were published in 2023 or later?",
        "ground_truth": "[verificare: edges.json target=seed → filtra year >= 2023]"
    },
    {
        "id": "B2_05", "hop": 2, "type": "bridge",
        "question": "Which authors have published more than one paper that cites Stokes 2020?",
        "ground_truth": "[verificare: papers citanti → autori con count > 1]"
    },

    # ══════════════════════════════════════════════════════════════════════════
    # BRIDGE ENTITY — 3 HOP
    # Tre salti: catena lunga A → B → C.
    # Previsto: Graph RAG mantiene accuratezza (segue il grafo),
    # Document RAG crolla (impossibile trovare 3 info nello stesso chunk),
    # LLM puro crolla.
    # ══════════════════════════════════════════════════════════════════════════
    {
        "id": "B3_01", "hop": 3, "type": "bridge",
        "question": "Which authors affiliated with the same institution as James J. Collins have published papers that cite a paper that cites Stokes 2020?",
        "ground_truth": "[Collins → MIT → autori MIT → papers che citano papers citanti seed]"
    },
    {
        "id": "B3_02", "hop": 3, "type": "bridge",
        "question": "Which institutions appear in papers at hop-2 distance from Stokes 2020 but not in papers at hop-1 distance?",
        "ground_truth": "[istituzioni hop2] - [istituzioni hop1]"
    },
    {
        "id": "B3_03", "hop": 3, "type": "bridge",
        "question": "Which topics appear in papers at hop-2 distance from Stokes 2020 but not in papers that directly cite it?",
        "ground_truth": "[topics hop2] - [topics hop1]"
    },
    {
        "id": "B3_04", "hop": 3, "type": "bridge",
        "question": "Which authors have co-authored with at least two different first-authors of papers that directly cite Stokes 2020?",
        "ground_truth": "[papers citanti seed → primo autore → co-autori → filtra count >= 2]"
    },
    {
        "id": "B3_05", "hop": 3, "type": "bridge",
        "question": "Which institution has produced the highest number of papers within 2 citation hops of Stokes 2020?",
        "ground_truth": "[tutti i paper nel grafo → autori → istituzioni → conta paper per istituzione]"
    },

    # ══════════════════════════════════════════════════════════════════════════
    # COMPARISON — 1 HOP
    # Confronto diretto tra due entità su una proprietà.
    # Previsto: tutti e tre i sistemi competitivi, LLM puro può sorprendere
    # su paper molto noti.
    # ══════════════════════════════════════════════════════════════════════════
    {
        "id": "C1_01", "hop": 1, "type": "comparison",
        "question": "Which paper in the dataset has the highest citation count?",
        "ground_truth": "[verificare in data/papers.json: max cited_by_count]"
    },
    {
        "id": "C1_02", "hop": 1, "type": "comparison",
        "question": "How many papers in the dataset were published after 2022?",
        "ground_truth": "[calcolare da data/papers.json: count year > 2022]"
    },


    # ══════════════════════════════════════════════════════════════════════════
    # COMPARISON — 2 HOP
    # Confronto che richiede prima di recuperare le entità (1 salto),
    # poi di confrontarle (secondo salto implicito).
    # Previsto: Document RAG può vincere perché sintetizza contesto ricco,
    # Graph RAG risponde in modo secco (solo il fatto, senza narrativa).
    # ══════════════════════════════════════════════════════════════════════════
    {
        "id": "C2_01", "hop": 2, "type": "comparison",
        "question": "Do any papers that cite Stokes 2020 share authors with each other?",
        "ground_truth": "[verificare: papers citanti → autori → intersezione tra paper diversi]"
    },
    {
        "id": "C2_02", "hop": 2, "type": "comparison",
        "question": "Do papers published after 2022 that cite Stokes 2020 have on average more or fewer citations than those published before 2022?",
        "ground_truth": "[calcolare: media cited_by_count per year <= 2022 vs > 2022]"
    },

    # ══════════════════════════════════════════════════════════════════════════
    # COMPARISON — 3 HOP
    # Confronto che richiede prima di ricostruire due sottografi (2-3 salti),
    # poi di confrontarli. Il Graph RAG può rispondere in modo preciso ma secco,
    # il Document RAG fatica a recuperare entrambe le catene nello stesso chunk.
    # ══════════════════════════════════════════════════════════════════════════
    {
        "id": "C3_01", "hop": 3, "type": "comparison",
        "question": "Which has more distinct institutions: papers at hop-1 or papers at hop-2 distance from Stokes 2020?",
        "ground_truth": "[calcolare: istituzioni distinte nei paper a hop-1 vs hop-2]"
    },
    {
        "id": "C3_02", "hop": 3, "type": "comparison",
        "question": "Do authors who appear at both hop-1 and hop-2 distance from Stokes 2020 tend to be affiliated with more institutions than those who appear only at hop-1?",
        "ground_truth": "[calcolare: media istituzioni per autori in hop1∩hop2 vs solo hop1]"
    },
    {
        "id": "C3_03", "hop": 3, "type": "comparison",
        "question": "Which topic appears in more papers across the entire citation network: machine learning or drug resistance?",
        "ground_truth": "[contare: papers con concept 'machine learning' vs 'drug resistance' in tutto il grafo]"
    },
    {
        "id": "C3_04", "hop": 3, "type": "comparison",
        "question": "Are there more papers with authors from a single institution or papers with authors from multiple institutions among those citing Stokes 2020?",
        "ground_truth": "[calcolare: papers con 1 istituzione vs papers con >1 istituzione tra i citanti]"
    },
    {
        "id": "C3_05", "hop": 3, "type": "comparison",
        "question": "Which year produced more papers at hop-2 distance from Stokes 2020: 2022 or 2023?",
        "ground_truth": "[calcolare: count papers hop-2 con year=2022 vs year=2023]"
    },

    # ══════════════════════════════════════════════════════════════════════════
    # NARRATIVE — 1-2 HOP
    # Richiedono comprensione del testo, non solo navigazione di relazioni.
    # Previsto: Document RAG vince (recupera abstract esplicativi),
    # Graph RAG produce risposta secca o incompleta,
    # LLM puro può avere conoscenza parametrica parziale.
    # ══════════════════════════════════════════════════════════════════════════
    {
        "id": "N_01", "hop": 1, "type": "narrative",
        "question": "Why is the Stokes 2020 paper considered a landmark in computational drug discovery?",
        "ground_truth": "[sintetizzare dall'abstract di Stokes 2020: primo uso di deep learning per scoprire halicin, composto strutturalmente nuovo, efficace su patogeni resistenti]"
    },
    {
        "id": "N_02", "hop": 1, "type": "narrative",
        "question": "What is halicin and why is it significant according to the Stokes 2020 paper?",
        "ground_truth": "[dall'abstract: composto antibiotico scoperto tramite rete neurale, efficace contro batteri resistenti tra cui M. tuberculosis e A. baumannii pan-resistant]"
    },
    {
        "id": "N_03", "hop": 2, "type": "narrative",
        "question": "Based on their abstracts, what methodological approaches do papers citing Stokes 2020 use to extend or improve upon the original work?",
        "ground_truth": "[sintetizzare dagli abstract dei paper citanti: es. diversi tipi di reti neurali, diversi target batterici, approcci multi-task]"
    },
    {
        "id": "N_04", "hop": 2, "type": "narrative",
        "question": "What limitations of the Stokes 2020 approach are discussed in papers that cite it?",
        "ground_truth": "[sintetizzare dagli abstract dei paper citanti che discutono limiti dell'approccio originale]"
    },
    {
        "id": "N_05", "hop": 2, "type": "narrative",
        "question": "How has the field of machine learning for antibiotic discovery evolved since Stokes 2020, based on the papers in the dataset?",
        "ground_truth": "[sintetizzare dalla collezione di abstract: trend temporali, nuovi approcci, nuovi target]"
    },
]


# ── ESECUZIONE VALUTAZIONE ────────────────────────────────────────────────────

def run_evaluation():
    print("=" * 60)
    print("AntibioticKG-RAG — Valutazione comparativa")
    print(f"Domande totali: {len(QUESTIONS)}")
    print(f"  Bridge Entity (1-2-3 hop): {sum(1 for q in QUESTIONS if q['type']=='bridge')}")
    print(f"  Comparison    (1-2-3 hop): {sum(1 for q in QUESTIONS if q['type']=='comparison')}")
    print(f"  Narrative     (1-2 hop):   {sum(1 for q in QUESTIONS if q['type']=='narrative')}")
    print("=" * 60)

    doc_rag   = DocumentRAG()
    graph_rag = GraphRAG()
    results   = []

    for i, q in enumerate(QUESTIONS):
        print(f"\n[{i+1}/{len(QUESTIONS)}] {q['id']} (hop={q['hop']}, type={q['type']})")
        print(f"  Q: {q['question'][:70]}...")

        result = {
            "id":           q["id"],
            "hop":          q["hop"],
            "type":         q["type"],
            "question":     q["question"],
            "ground_truth": q["ground_truth"],
        }

        print("  → System 1 (LLM puro)...")
        result["sys1_answer"] = system1_parametric_llm(q["question"])

        print("  → System 2 (Document RAG)...")
        ans2 = doc_rag.answer(q["question"])
        result["sys2_answer"] = ans2["answer"]
        result["sys2_chunks"] = ans2["retrieved_chunks"]

        print("  → System 3 (Graph RAG)...")
        ans3 = graph_rag.answer(q["question"])
        result["sys3_answer"]        = ans3["answer"]
        result["sys3_sparql"]        = ans3["sparql_query"]
        result["sys3_query_valid"]   = ans3["query_valid"]
        result["sys3_results_count"] = ans3["results_count"]

        results.append(result)

    # Salva risultati completi
    with open("results/evaluation_results.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Genera CSV da compilare manualmente
    with open("results/scores.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "id", "hop", "type", "question",
            "sys1_correct", "sys2_correct", "sys3_correct",
            "sys3_query_valid",
            "sys1_error_type", "sys2_error_type", "sys3_error_type",
            "notes"
        ])
        for r in results:
            writer.writerow([
                r["id"], r["hop"], r["type"], r["question"],
                "", "", "",  # da compilare: 0/1
                1 if r.get("sys3_query_valid") else 0,
                "", "", "", ""
            ])

    print("\n" + "=" * 60)
    print("VALUTAZIONE COMPLETATA")
    print("  results/evaluation_results.json — risposte complete")
    print("  results/scores.csv              — da compilare manualmente (0/1)")
    print("\nTassonomia errori:")
    print("  System 1: invention / entity_confusion / incomplete")
    print("  System 2: chunk_not_retrieved / not_in_chunk / poor_synthesis")
    print("  System 3: malformed_query / wrong_relation / data_absent")
    print("=" * 60)


if __name__ == "__main__":
    run_evaluation()
