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
    # Confronti diretti su proprietà del grafo, senza attraversare relazioni
    # (anno, publication type, primary topic — proprietà singole del Paper).
    # ══════════════════════════════════════════════════════════════════════════
    {
        "id": "C1_01", "hop": 1, "type": "comparison",
        "question": "Do papers published in 2020 or earlier have a higher or lower average citation count than papers published after 2020?",
        "ground_truth": "Papers published in 2020 or earlier have a much higher average citation count (339.48, n=42) than papers published after 2020 (214.73, n=558) — older papers have had more time to accumulate citations."
    },
    {
        "id": "C1_02", "hop": 1, "type": "comparison",
        "question": "Do papers published in 2023 have a higher or lower average citation count than papers published in 2024?",
        "ground_truth": "Papers published in 2023 have a slightly higher average citation count (219.97, n=181) than papers published in 2024 (207.14, n=140)."
    },
    {
        "id": "C1_03", "hop": 1, "type": "comparison",
        "question": "Are papers on \"Computational Drug Discovery Methods\" more or less likely to be open access than papers on \"Artificial Intelligence in Healthcare and Education\"?",
        "ground_truth": "Papers on \"Artificial Intelligence in Healthcare and Education\" are more likely to be open access (93.4%, 57/61) than papers on \"Computational Drug Discovery Methods\" (74.5%, 76/102)."
    },

    # ══════════════════════════════════════════════════════════════════════════
    # COMPARISON — 2 HOP
    # Confronti che attraversano una relazione (Paper → Author → Institution /
    # Country) prima di aggregare. Categorie non mutuamente esclusive — vedi
    # le percentuali di sovrapposizione riportate nei singoli ground_truth.
    # ══════════════════════════════════════════════════════════════════════════
    {
        "id": "C2_01", "hop": 2, "type": "comparison",
        "question": "Do papers with ≥1 author from an academic institution have higher/lower avg citations than papers with ≥1 author from a company institution?",
        "ground_truth": "Papers with a company-affiliated author have a slightly higher average citation count (253.95, n=86) than papers with an academic-affiliated author (223.03, n=571). Note: categories are not mutually exclusive — 91.9% of company-affiliated papers also have an academic-affiliated author."
    },
    {
        "id": "C2_02", "hop": 2, "type": "comparison",
        "question": "Are papers with ≥1 company-affiliated author more/less likely to be open access than papers with ≥1 academic-affiliated author?",
        "ground_truth": "Company-affiliated papers are more likely to be open access (81.4%, 70/86) than academic-affiliated papers (73.9%, 422/571). Note: categories are not mutually exclusive — 91.9% of company-affiliated papers also have an academic-affiliated author."
    },
    {
        "id": "C2_03", "hop": 2, "type": "comparison",
        "question": "Do papers with ≥1 academic-affiliated author have higher/lower avg citations than papers with ≥1 nonprofit-affiliated author?",
        "ground_truth": "Nonprofit-affiliated papers have a notably higher average citation count (323.64, n=74) than academic-affiliated papers (223.03, n=571). Note: categories are not mutually exclusive — 98.6% of nonprofit-affiliated papers also have an academic-affiliated author."
    },
    {
        "id": "C2_04", "hop": 2, "type": "comparison",
        "question": "Do papers with ≥1 US-affiliated author have higher/lower avg citations than papers with ≥1 China-affiliated author?",
        "ground_truth": "US-affiliated papers have a higher average citation count (280.43, n=244) than China-affiliated papers (213.12, n=176). Note: categories are not mutually exclusive — 21.7% of US-affiliated papers also have a China-affiliated author, and 30.1% of China-affiliated papers also have a US-affiliated author."
    },
    {
        "id": "C2_05", "hop": 2, "type": "comparison",
        "question": "Are papers with ≥1 US-affiliated author more/less likely to be open access than papers with ≥1 China-affiliated author?",
        "ground_truth": "US-affiliated papers are much more likely to be open access (77.0%, 188/244) than China-affiliated papers (54.0%, 95/176). Note: same non-mutual-exclusivity caveat as C2_04."
    },

    # ══════════════════════════════════════════════════════════════════════════
    # COMPARISON — 3 HOP
    # Confronti con filtro combinato (istituzione + paese) o join più lungo
    # (topic → autore → istituzione → paese) prima di aggregare.
    # ══════════════════════════════════════════════════════════════════════════
    {
        "id": "C3_01", "hop": 3, "type": "comparison",
        "question": "Among papers with ≥1 company-affiliated author, do US-based ones have higher/lower avg citations than China-based ones?",
        "ground_truth": "Among company-affiliated papers, China-based ones have a higher average citation count (448.67, n=15) than US-based ones (339.62, n=45) — note: small China subsample (n=15), and 40% of it overlaps with the US-based group (mixed-affiliation papers)."
    },
    {
        "id": "C3_02", "hop": 3, "type": "comparison",
        "question": "Do papers about \"Antibiotics\" involve authors from more distinct countries than papers about \"Nanotechnology\"?",
        "ground_truth": "Papers about \"Antibiotics\" involve authors from 38 distinct countries, versus 21 for papers about \"Nanotechnology\"."
    },

    # ══════════════════════════════════════════════════════════════════════════
    # NARRATIVE — 1 HOP
    # Ancorate a un singolo paper con abstract verificato, domanda su un solo
    # fatto/aspetto discorsivo.
    # ══════════════════════════════════════════════════════════════════════════
    {
        "id": "N1_01", "hop": 1, "type": "narrative",
        "question": "What limitation of machine learning models does the paper 'Exposing the Limitations of Molecular Machine Learning with Activity Cliffs' identify when predicting molecular bioactivity?",
        "ground_truth": "The paper shows that machine and deep learning models struggle to accurately predict potency for activity cliffs (pairs of structurally similar molecules with large potency differences). Benchmarking 24 approaches on 30 targets, it found that all methods struggled on activity cliffs, with descriptor-based ML approaches actually outperforming more complex deep learning methods — motivating the MoleculeACE benchmarking platform and dedicated activity-cliff-aware evaluation metrics."
    },
    {
        "id": "N1_02", "hop": 1, "type": "narrative",
        "question": "According to the paper on predicting antimicrobial resistance from whole-genome sequencing, which four machine learning methods did the researchers compare, and for which four antibiotics?",
        "ground_truth": "Logistic regression, support vector machine, random forest, and convolutional neural network, evaluated for predicting resistance to ciprofloxacin, cefotaxime, ceftazidime, and gentamicin."
    },
    {
        "id": "N1_03", "hop": 1, "type": "narrative",
        "question": "According to the paper on ARTS 2.0, what genome mining strategy does the tool use to prioritize promising biosynthetic gene clusters for novel antibiotics?",
        "ground_truth": "ARTS uses a target-directed genome mining approach: it predicts the likely mode of action of a compound encoded by an uncharacterized biosynthetic gene cluster based on the presence of resistance target genes within or near that cluster, helping prioritize clusters encoding antibiotics with novel modes of action."
    },
    {
        "id": "N1_04", "hop": 1, "type": "narrative",
        "question": "According to the paper introducing GraphINVENT, which of the six graph neural network-based generative models tested performed best against the benchmark metrics?",
        "ground_truth": "The gated-graph neural network performed best among the six GNN-based generative models compared, benchmarked using the MOSES distribution-based metrics."
    },
    {
        "id": "N1_05", "hop": 1, "type": "narrative",
        "question": "Among papers that cite Stokes 2020, are generative/molecular-design approaches or docking/virtual-screening approaches more common as a follow-on methodology?",
        "ground_truth": "Generative/molecular-design approaches are markedly more common: 29 of the 146 citing papers with abstracts reference generative or latent-space design methods, versus 13 referencing docking or virtual screening — more than double."
    },

    # ══════════════════════════════════════════════════════════════════════════
    # NARRATIVE — 2 HOP
    # Singolo paper, ma la domanda combina due fatti/aspetti dello stesso
    # abstract (metodo + localizzazione, causa + soluzione, due valori numerici).
    # ══════════════════════════════════════════════════════════════════════════
    {
        "id": "N2_01", "hop": 2, "type": "narrative",
        "question": "Why do \"activity cliffs\" pose a particular challenge for machine learning models predicting molecular bioactivity, according to the paper that discusses them?",
        "ground_truth": "Activity cliffs are pairs of molecules that are highly similar in structure but show large differences in potency. Because ML/DL models largely rely on structural similarity to predict activity, these cases break the usual similarity-property assumption, causing models to underperform on them specifically — the paper found this true across all 24 tested approaches on 30 targets, with simpler descriptor-based methods outperforming deep learning on these edge cases."
    },
    {
        "id": "N2_02", "hop": 2, "type": "narrative",
        "question": "According to the paper titled \"Potent antibiotic design via guided search from antibacterial activity evaluations,\" what method does it propose to generate new antibiotic candidates, and in which country are its authors' institutions located?",
        "ground_truth": "The paper proposes MDAGS (Molecular Design via Attribute-Guided Search), which builds an antibacterial-activity latent space and guides optimization of compounds within it to generate novel candidates with strong antibacterial activity, without requiring extensive costly experimental evaluation. All authors' institutions are located in China (Xidian University)."
    },
    {
        "id": "N2_03", "hop": 2, "type": "narrative",
        "question": "According to the paper titled \"Benchmarking AlphaFold-enabled molecular docking predictions for antibiotic discovery,\" why did the initial docking approach show weak performance, and what technique did the researchers use to improve it?",
        "ground_truth": "The initial AlphaFold2-based docking approach performed weakly (average auROC of 0.48) due to widespread promiscuity among the tested proteins and compounds. Performance was improved by rescoring docking poses with machine learning-based approaches, reaching average auROCs as high as 0.63, with ensembles of rescoring functions further improving prediction accuracy and the true-positive-to-false-positive ratio."
    },
    {
        "id": "N2_04", "hop": 2, "type": "narrative",
        "question": "According to the paper titled \"Predicting cell-penetrating peptides using machine learning algorithms and navigating in their chemical space,\" what accuracy did the proposed method (BChemRF-CPPred) achieve on the PDB-based test, and how does this compare to the FASTA-based test?",
        "ground_truth": "BChemRF-CPPred achieved 90.66% accuracy (AUC = 0.9365) on the PDB-based independent test, versus 86.5% accuracy (AUC = 0.9216) on the FASTA-based independent test — the PDB-based (structure-based) input outperformed the FASTA-based (sequence-only) input."
    },
    {
        "id": "N2_05", "hop": 2, "type": "narrative",
        "question": "Based on their abstracts, what methodological approaches do papers citing Stokes 2020 use to extend or improve upon the original work?",
        "ground_truth": "Across all 146 citing papers with available abstracts, the most common methodological threads are: generative/latent-space molecular design (29 papers, e.g. MDAGS W4318392954, GraphINVENT W3107551345), natural-product-based approaches (14, e.g. plant flavonoids W3164905478), docking/virtual screening (13, e.g. AlphaFold-enabled docking W4294719209), peptide-based discovery (12, e.g. AMPSphere W4399367032), genomics/metagenomics-driven approaches (11, e.g. whole-genome AMR prediction W3203901993, ARTS 2.0 genome mining W3026412036), and smaller clusters on graph neural networks (8), explainable AI (8), and open-source property-prediction toolkits (7, e.g. Chemprop, DeepPurpose, ADMETlab 3.0). Generative/design-based work is the single largest follow-on category, more than double the next most common approach."
    },

    # ══════════════════════════════════════════════════════════════════════════
    # NARRATIVE — 3 HOP
    # Sintesi tra più paper del network citazionale (o un singolo paper con
    # contenuto più articolato), non un singolo fatto isolato.
    # ══════════════════════════════════════════════════════════════════════════
    {
        "id": "N3_01", "hop": 3, "type": "narrative",
        "question": "Based on papers in the citation network around Stokes 2020, what are the main strategies researchers propose to combat antimicrobial resistance beyond deep learning-based antibiotic discovery?",
        "ground_truth": "Across all 146 citing papers with abstracts, non-DL strategies cluster into: natural-product-derived antibacterials (14 papers — plant flavonoids W3164905478, a 459-compound ethnobotanical review W3099939732); genome mining for resistance-conferring gene clusters as a route to new antibiotics rather than prediction (ARTS 2.0, W3026412036); targeting resistance mechanisms directly via efflux pump inhibitors (3 papers, W4377970794); clinical/translational strategies — antimicrobial stewardship using electronic health record data (W3127516550); and broader policy-level alternatives — narrow-spectrum drugs, bacteriophage therapy (3 papers), monoclonal antibodies, vaccines (2 papers), and improved diagnostics (9 papers, W4291020097)."
    },
    {
        "id": "N3_02", "hop": 3, "type": "narrative",
        "question": "Across papers that cite Stokes 2020 and focus on genomic approaches to antimicrobial resistance, what different strategies do they use — predicting resistance directly from sequence data versus mining genomes for resistance-conferring gene clusters?",
        "ground_truth": "Two distinct genomic strategies appear: (1) direct AMR prediction from whole-genome sequencing using classifiers such as LR, SVM, RF and CNN trained on genomic encodings, aiming to replace slow phenotypic susceptibility testing (W3203901993); and (2) target-directed genome mining (ARTS 2.0, W3026412036), which instead searches genomes for biosynthetic gene clusters carrying their own resistance genes to prioritize discovery of new antibiotics with novel mechanisms. The first predicts resistance in known pathogens; the second discovers new antibiotic candidates via bacterial self-resistance genes. A third review (W3127516550) frames ML-for-AMR across three domains: genomic prediction, mechanism-of-action discovery, and antimicrobial stewardship using electronic health record data."
    },
    {
        "id": "N3_03", "hop": 3, "type": "narrative",
        "question": "Do any papers that cite Stokes 2020 raise concerns about the interpretability of deep learning models, and if so, in which research contexts?",
        "ground_truth": "Yes. A review on interpreting deep neural networks for psychiatric research (W3094739916) explicitly frames DNNs as a 'black box' needing better interpretability tools, applying this concern outside drug discovery entirely. A broader integrative review of deep learning in drug discovery (W4309490745) separately discusses how explainable AI supports drug discovery problems. Both treat interpretability as an open challenge for deep learning generally, citing Stokes 2020 as a landmark example rather than addressing its interpretability directly."
    },
    {
        "id": "N3_04", "hop": 3, "type": "narrative",
        "question": "Beyond antibiotic and drug discovery, in what other scientific domain does one of the papers citing Stokes 2020 apply a similar machine learning paradigm, and for what purpose?",
        "ground_truth": "The Self-Driving Laboratories review (W4401535000) cites Stokes 2020 while discussing automated, ML-guided experimental discovery, extending the same 'let a model guide which candidates to test experimentally' paradigm from antibiotic discovery to materials science and chemistry more broadly, via autonomous experimental platforms."
    },
    {
        "id": "N3_05", "hop": 3, "type": "narrative",
        "question": "According to the review on machine learning applications to antimicrobial resistance as an emerging model for translational research, what three domains of ML-AMR research does it identify?",
        "ground_truth": "(1) Prediction of AMR using genomic data; (2) use of ML to gain insight into the cellular functions disrupted by antibiotics, underpinning mechanism-of-action discovery; (3) application of ML for antimicrobial stewardship using data extracted from electronic health records."
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

    with open("results/scores.csv", "w", newline="", encoding="utf-8") as f:
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
