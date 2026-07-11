"""
04_evaluation.py
----------------
Confronto tra i tre sistemi su 30 domande di valutazione.

Le domande sono ispirate ai tipi di HotpotQA (Yang et al. 2018):
  - BRIDGE ENTITY: attraversano relazioni dell'ontologia in catena
  - COMPARISON: confrontano due insiemi di entità su una proprietà
  - NARRATIVE: richiedono comprensione discorsiva del testo

Le domande sono volutamente generiche e basate sulle relazioni
dell'ontologia (cites, hasAuthor, affiliatedWith, locatedIn,
coAuthorWith, about) piuttosto che su specifici paper o autori.
Questo rende la valutazione indipendente dalla conoscenza del dominio.

Output:
  results/evaluation_results.json
  results/scores.csv
"""

import json
import csv
import os
import importlib.util

spec = importlib.util.spec_from_file_location("systems", "03_systems.py")
systems_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(systems_module)
system1_parametric_llm = systems_module.system1_parametric_llm
DocumentRAG            = systems_module.DocumentRAG
GraphRAG               = systems_module.GraphRAG

try:
    from ragas import evaluate
    from ragas.metrics import faithfulness, answer_relevancy
    from datasets import Dataset
    import litellm
    from ragas.llms import llm_factory

    HAS_RAGAS = True
    print("[INFO] RAGAS disponibile")

    # Configurazione LLM per RAGAS: usa Gemini invece del default OpenAI,
    # riusando la stessa GOOGLE_API_KEY già presente nel file .env
    os.environ["GEMINI_API_KEY"] = os.environ.get("GOOGLE_API_KEY", "")
    ragas_llm = llm_factory(
        "gemini/gemini-2.5-flash",
        provider="litellm",
        client=litellm.completion,
    )
except ImportError:
    HAS_RAGAS = False
    print("[WARN] RAGAS non installato — pip install ragas datasets litellm")

os.makedirs("results", exist_ok=True)


def exact_match(answer: str, ground_truth: str) -> int:
    if not ground_truth or ground_truth.startswith("["):
        return -1
    answer_lower = answer.lower()
    gt_lower = ground_truth.lower()
    if len(ground_truth.split()) <= 5:
        return 1 if gt_lower in answer_lower else 0
    key_terms = [t.strip(".,;:()") for t in gt_lower.split() if len(t) > 3]
    if not key_terms:
        return 0
    matches = sum(1 for t in key_terms if t in answer_lower)
    return 1 if matches / len(key_terms) >= 0.6 else 0


def compute_ragas(question: str, answer: str, contexts: list) -> dict:
    if not HAS_RAGAS or not contexts:
        return {"faithfulness": -1, "answer_relevancy": -1}
    try:
        dataset = Dataset.from_dict({
            "question": [question],
            "answer":   [answer],
            "contexts": [contexts],
        })
        result = evaluate(dataset, metrics=[faithfulness, answer_relevancy], llm=ragas_llm)
        return {
            "faithfulness":     round(float(result["faithfulness"]), 3),
            "answer_relevancy": round(float(result["answer_relevancy"]), 3),
        }
    except Exception as e:
        print(f"    [WARN] RAGAS error: {e}")
        return {"faithfulness": -1, "answer_relevancy": -1}


# ── SET DI DOMANDE ────────────────────────────────────────────────────────────
# Le domande attraversano le relazioni: cites, hasAuthor, affiliatedWith,
# locatedIn, coAuthorWith, about. Nessuna nomina specifici autori o paper
# (eccetto il seed paper, che è il punto di partenza noto del progetto).

QUESTIONS = [

    # ══════════════════════════════════════════════════════════════════════════
    # BRIDGE ENTITY — 1 HOP
    # Un salto diretto su una relazione dell'ontologia.
    # ══════════════════════════════════════════════════════════════════════════
    {
        "id": "B1_01", "hop": 1, "type": "bridge",
        "question": "How many papers in the dataset are open access?",
        "ground_truth": "445 out of 600 papers in the dataset are open access"
    },
    {
        "id": "B1_02", "hop": 1, "type": "bridge",
        "question": "What publication types are present in the dataset and how are they distributed?",
        "ground_truth": "Article (302 papers), review (288), book-chapter (5), editorial (4), preprint (1)"
    },
    {
        "id": "B1_03", "hop": 1, "type": "bridge",
        "question": "How many papers in the dataset were published after 2022?",
        "ground_truth": "357 papers were published after 2022"
    },
    {
        "id": "B1_04", "hop": 1, "type": "bridge",
        "question": "What are the most common research topics across all papers in the dataset?",
        "ground_truth": "Computer science (381 papers), Artificial intelligence (267), Machine learning (162), Data science (154), Medicine (113), Biology (109), Drug discovery (103)"
    },
    {
        "id": "B1_05", "hop": 1, "type": "bridge",
        "question": "How many papers in the dataset directly cite the seed paper 'A Deep Learning Approach to Antibiotic Discovery' by Stokes et al. 2020?",
        "ground_truth": "206 papers in the dataset directly cite Stokes 2020. This is a subset of the 2213 total citations the paper has on OpenAlex: the dataset was built with a cap of 200 hop-1 papers (minimum 20 citations each) plus 6 additional papers discovered via referenced_works reconciliation."
    },

    # ══════════════════════════════════════════════════════════════════════════
    # BRIDGE ENTITY — 2 HOP
    # Due relazioni in catena (es. Paper → Author → Institution).
    # ══════════════════════════════════════════════════════════════════════════
    {
        "id": "B2_01", "hop": 2, "type": "bridge",
        "question": "Which countries are most represented among authors of papers that directly cite the Stokes 2020 seed paper?",
        "ground_truth": "US (95 papers), China/CN (52), UK/GB (37), Canada/CA (20), Germany/DE (16), Switzerland/CH (16)"
    },
    {
        "id": "B2_02", "hop": 2, "type": "bridge",
        "question": "How many authors appear in more than one paper that cites the Stokes 2020 seed paper?",
        "ground_truth": "97 authors appear in more than one paper that directly cites Stokes 2020"
    },
    {
        "id": "B2_03", "hop": 2, "type": "bridge",
        "question": "How many academic institutions (universities and research centres) are represented in papers that cite the Stokes 2020 seed paper?",
        "ground_truth": "351 academic institutions of type education are represented among authors of papers citing Stokes 2020"
    },
    {
        "id": "B2_04", "hop": 2, "type": "bridge",
        "question": "How many papers that cite the Stokes 2020 seed paper cover the topic of machine learning?",
        "ground_truth": "63 papers that directly cite Stokes 2020 cover the topic of Machine learning"
    },
    {
        "id": "B2_05", "hop": 2, "type": "bridge",
        "question": "How many citation relationships exist between papers that directly cite the Stokes 2020 seed paper (i.e. hop-1 papers that also cite each other)?",
        "ground_truth": "281 citation relationships exist between papers at hop-1 distance from Stokes 2020"
    },

    # ══════════════════════════════════════════════════════════════════════════
    # BRIDGE ENTITY — 3 HOP
    # Tre relazioni in catena. Queste domande usano relazioni diverse:
    # isOpenAccess + affiliatedWith + locatedIn, hasPublicationType + about,
    # cites + about. Non dipendono tutte dal seed paper.
    # ══════════════════════════════════════════════════════════════════════════
    {
        "id": "B3_01", "hop": 3, "type": "bridge",
        "question": "How many papers in the dataset have at least one author affiliated with a company institution?",
        "ground_truth": "86 papers have at least one author affiliated with an institution of type company"
    },
    {
        "id": "B3_02", "hop": 3, "type": "bridge",
        "question": "Which institution has produced the highest number of papers within 2 citation hops of Stokes 2020?",
        "ground_truth": "Harvard University (42 papers), followed by Massachusetts Institute of Technology (25), Stanford University (24), Broad Institute (22), Chinese Academy of Sciences (22)"
    },
    {
        "id": "B3_03", "hop": 3, "type": "bridge",
        "question": "How many open access papers in the dataset have at least one author affiliated with a non-US institution?",
        "ground_truth": "361 open access papers have at least one author from a non-US institution"
    },
    {
        "id": "B3_04", "hop": 3, "type": "bridge",
        "question": "How many distinct institutions appear in papers at hop-2 distance from Stokes 2020 but not in papers at hop-1 distance?",
        "ground_truth": "1059 institutions appear exclusively in hop-2 papers (hop-1 has 654 distinct institutions, hop-2 has 1339)"
    },
    {
        "id": "B3_05", "hop": 3, "type": "bridge",
        "question": "How many papers in the dataset cite papers that cover the topic of drug discovery, without themselves covering that topic?",
        "ground_truth": "196 papers cite papers covering the topic of Drug discovery without themselves being classified under that topic"
    },

    # ══════════════════════════════════════════════════════════════════════════
    # COMPARISON — 1 HOP
    # Confronti diretti su proprietà del grafo. Le risposte non sono
    # ovvie a priori — questo è il punto: testare se i sistemi riescono
    # a ragionare su aggregazioni di dati strutturati.
    # ══════════════════════════════════════════════════════════════════════════
    {
        "id": "C1_01", "hop": 1, "type": "comparison",
        "question": "Do review papers in the dataset have on average more or fewer citations than article papers?",
        "ground_truth": "Review papers have on average more citations (242.5) than article papers (203.7)"
    },
    {
        "id": "C1_02", "hop": 1, "type": "comparison",
        "question": "Do open access papers in the dataset have on average more or fewer citations than non-open access papers?",
        "ground_truth": "Open access and non-open access papers have almost the same average citations (221.6 vs 228.9), with non-OA papers slightly higher — contrary to what one might expect"
    },
    {
        "id": "C1_03", "hop": 1, "type": "comparison",
        "question": "Do most authors in the dataset collaborate exclusively with co-authors from the same institution, or do they have cross-institution collaborations?",
        "ground_truth": "Among the 4657 authors in the dataset, 40 have no co-authors at all (sole authors). Of the remaining 4617 authors with at least one co-author, the vast majority (3952, 84.9% of all authors) have at least one co-author from a different institution. Only 665 authors (14.3%) collaborate exclusively within their own institution — indicating that cross-institutional collaboration is the dominant pattern in this research network."
    },

    # ══════════════════════════════════════════════════════════════════════════
    # COMPARISON — 2 HOP
    # Confronti che richiedono prima di costruire due insiemi attraverso
    # una relazione, poi di confrontarli su una proprietà.
    # ══════════════════════════════════════════════════════════════════════════
    {
        "id": "C2_01", "hop": 2, "type": "comparison",
        "question": "Do papers with authors from 3 or more different countries tend to have more citations than papers with authors from a single country?",
        "ground_truth": "Yes — papers with authors from 3 or more countries have on average more citations (268.4) than papers with authors from a single country (205.9)"
    },
    {
        "id": "C2_02", "hop": 2, "type": "comparison",
        "question": "Are there more papers with authors from academic institutions or papers with authors from company institutions among those with 3 or more authors?",
        "ground_truth": "Academic institution papers (495) greatly outnumber company institution papers (81) among papers with 3 or more authors"
    },

    # ══════════════════════════════════════════════════════════════════════════
    # COMPARISON — 3 HOP
    # Confronti che attraversano 3 relazioni prima di confrontare.
    # ══════════════════════════════════════════════════════════════════════════
    {
        "id": "C3_01", "hop": 3, "type": "comparison",
        "question": "Which has more distinct author countries: papers covering the topic of drug discovery or papers covering artificial intelligence?",
        "ground_truth": "Papers on artificial intelligence involve authors from more countries (58) than papers on drug discovery (43)"
    },
    {
        "id": "C3_02", "hop": 3, "type": "comparison",
        "question": "Among papers that directly cite Stokes 2020, are there more papers that are also cited by other papers in the dataset, or papers that receive no citations from within the dataset?",
        "ground_truth": "Among the 206 papers directly citing Stokes 2020, 125 are also cited by other papers within the dataset, while 81 receive no internal citations — meaning the majority of hop-1 papers are embedded in the broader citation network, not just leaves"
    },
    {
        "id": "C3_03", "hop": 3, "type": "comparison",
        "question": "Which has more distinct institutions: papers at hop-1 or papers at hop-2 distance from Stokes 2020?",
        "ground_truth": "Hop-2 papers have more distinct institutions (1339) than hop-1 papers (654)"
    },
    {
        "id": "C3_04", "hop": 3, "type": "comparison",
        "question": "Among papers citing Stokes 2020, are there more papers with single-institution authorship or multi-institution authorship?",
        "ground_truth": "Multi-institution papers (155) greatly outnumber single-institution papers (48) among papers citing Stokes 2020"
    },
    {
        "id": "C3_05", "hop": 3, "type": "comparison",
        "question": "Do papers with more than 10 authors tend to be articles or reviews, and does this pattern hold for papers with fewer than 5 authors?",
        "ground_truth": "Papers with more than 10 authors are predominantly articles (70) over reviews (29). For papers with fewer than 5 authors, the pattern reverses: reviews (124) outnumber articles (98) — suggesting that large collaborative works favour the article format while smaller-team contributions more often take the review format"
    },

    # ══════════════════════════════════════════════════════════════════════════
    # NARRATIVE — valutate con RAGAS + manuale
    # 2 hop-1, 3 hop-2
    # ══════════════════════════════════════════════════════════════════════════
    {
        "id": "N_01", "hop": 1, "type": "narrative",
        "question": "Why is the Stokes 2020 paper considered a landmark in computational drug discovery?",
        "ground_truth": "The Stokes 2020 paper is considered a landmark because it was the first to use a deep learning model trained on molecular properties to screen over 100 million compounds and identify halicin — a structurally novel antibiotic with broad-spectrum activity against drug-resistant pathogens including M. tuberculosis and A. baumannii. The approach demonstrated that AI could discover antibiotics with mechanisms distinct from existing drugs."
    },
    {
        "id": "N_02", "hop": 1, "type": "narrative",
        "question": "What limitation of machine learning models does the paper 'Exposing the Limitations of Molecular Machine Learning with Activity Cliffs' identify when predicting molecular bioactivity?",
        "ground_truth": "The paper identifies 'activity cliffs' — pairs of structurally similar molecules with very different potency — as a key limitation. All 24 methods tested struggle with these cases, and descriptor-based methods outperform more complex deep learning models on this specific challenge."
    },
    {
        "id": "N_03", "hop": 2, "type": "narrative",
        "question": "Based on their abstracts, what methodological approaches do papers citing Stokes 2020 use to extend or improve upon the original work?",
        "ground_truth": "Papers citing Stokes 2020 extend the work through: graph neural networks for molecular design, generative AI for de novo drug design, large language models and foundation models applied to chemistry, multi-task learning across multiple molecular properties, and active learning to reduce experimental costs."
    },
    {
        "id": "N_04", "hop": 2, "type": "narrative",
        "question": "According to papers that cite Stokes 2020, what makes antimicrobial resistance prediction via machine learning particularly challenging?",
        "ground_truth": "Key challenges include: scarcity and quality of labelled training data, difficulty generalising models to new pathogens, limited interpretability of deep learning models, and the gap between in silico predictions and clinical efficacy."
    },
    {
        "id": "N_05", "hop": 2, "type": "narrative",
        "question": "Based on papers in the citation network around Stokes 2020, what are the main strategies researchers propose to combat antimicrobial resistance beyond deep learning-based antibiotic discovery?",
        "ground_truth": "Complementary strategies discussed in the citation network include the One Health approach (coordinating human, animal and environmental health), nanomaterials and nanocarriers as drug delivery systems, and targeting bacterial persisters — all going beyond the ML-based screening approach of Stokes 2020."
    },
]


# ── ESECUZIONE ────────────────────────────────────────────────────────────────
def run_evaluation():
    print("=" * 60)
    print("AntibioticKG-RAG — Valutazione comparativa")
    print(f"Domande totali: {len(QUESTIONS)}")
    print(f"  Bridge:     {sum(1 for q in QUESTIONS if q['type']=='bridge')}")
    print(f"  Comparison: {sum(1 for q in QUESTIONS if q['type']=='comparison')}")
    print(f"  Narrative:  {sum(1 for q in QUESTIONS if q['type']=='narrative')}")
    print("=" * 60)

    doc_rag   = DocumentRAG()
    graph_rag = GraphRAG()
    results   = []

    for i, q in enumerate(QUESTIONS):
        print(f"\n[{i+1}/{len(QUESTIONS)}] {q['id']} (hop={q['hop']}, {q['type']})")
        print(f"  Q: {q['question'][:70]}...")

        result = {
            "id": q["id"], "hop": q["hop"], "type": q["type"],
            "question": q["question"], "ground_truth": q["ground_truth"],
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

        # Exact match per bridge/comparison
        if q["type"] in ("bridge", "comparison"):
            result["sys1_auto"] = exact_match(result["sys1_answer"], q["ground_truth"])
            result["sys2_auto"] = exact_match(result["sys2_answer"], q["ground_truth"])
            result["sys3_auto"] = exact_match(result["sys3_answer"], q["ground_truth"])
            print(f"  Exact match → sys1={result['sys1_auto']} sys2={result['sys2_auto']} sys3={result['sys3_auto']}")
        else:
            result["sys1_auto"] = result["sys2_auto"] = result["sys3_auto"] = -1

        # RAGAS per narrative
        if q["type"] == "narrative" and HAS_RAGAS:
            ragas2 = compute_ragas(q["question"], result["sys2_answer"], result.get("sys2_chunks", []))
            result["sys2_faithfulness"]     = ragas2["faithfulness"]
            result["sys2_answer_relevancy"] = ragas2["answer_relevancy"]
            sparql_ctx = json.dumps(ans3.get("sparql_results", [])[:5])
            ragas3 = compute_ragas(q["question"], result["sys3_answer"],
                                   [sparql_ctx] if sparql_ctx != "[]" else [])
            result["sys3_faithfulness"]     = ragas3["faithfulness"]
            result["sys3_answer_relevancy"] = ragas3["answer_relevancy"]
        else:
            result["sys2_faithfulness"] = result["sys2_answer_relevancy"] = -1
            result["sys3_faithfulness"] = result["sys3_answer_relevancy"] = -1

        results.append(result)

    with open("results/evaluation_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    with open("results/scores.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "id", "hop", "type", "question",
            "sys1_correct", "sys2_correct", "sys3_correct",
            "sys3_query_valid",
            "sys1_error_type", "sys2_error_type", "sys3_error_type",
            "sys2_faithfulness", "sys2_answer_relevancy",
            "sys3_faithfulness", "sys3_answer_relevancy",
            "notes"
        ])
        for r in results:
            writer.writerow([
                r["id"], r["hop"], r["type"], r["question"],
                r.get("sys1_auto",""), r.get("sys2_auto",""), r.get("sys3_auto",""),
                1 if r.get("sys3_query_valid") else 0,
                "", "", "",
                r.get("sys2_faithfulness",""), r.get("sys2_answer_relevancy",""),
                r.get("sys3_faithfulness",""), r.get("sys3_answer_relevancy",""),
                ""
            ])

    print("\n" + "=" * 60)
    print("COMPLETATO")
    print("  results/evaluation_results.json")
    print("  results/scores.csv")
    print("\nFase 4 — Revisione manuale:")
    print("  Correggere -1, verificare 0/1 automatici, compilare error_type")
    print("\nTassonomia errori:")
    print("  Sys1: invention / entity_confusion / incomplete")
    print("  Sys2: chunk_not_retrieved / not_in_chunk / poor_synthesis")
    print("  Sys3: malformed_query / wrong_relation / triple_flip / data_absent")
    print("=" * 60)


if __name__ == "__main__":
    run_evaluation()
