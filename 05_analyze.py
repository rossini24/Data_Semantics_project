"""
05_analyze.py
-------------
Legge scores.csv (compilato manualmente) e produce:
  - accuratezza per sistema e per livello di hop
  - hallucination rate per il sistema 1
  - SPARQL validity rate per il sistema 3
  - tabella riassuntiva (stampa a console)
  - grafico accuratezza-vs-hop (results/accuracy_vs_hop.png)

Questo script è il cuore della RQ2: mostra come degrada ogni sistema
all'aumentare della profondità relazionale richiesta.
"""

import csv
import json
from collections import defaultdict

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("[WARN] matplotlib non installato — il grafico non verrà generato")
    print("       Installa con: pip install matplotlib")


def analyze():
    print("=" * 60)
    print("AntibioticKG-RAG — Analisi risultati")
    print("=" * 60)

    # Carica scores.csv
    rows = []
    with open("results/scores.csv") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    if not rows:
        print("ERRORE: scores.csv è vuoto.")
        return

    # ── METRICHE PER HOP ──────────────────────────────────────────────────────
    hop_data = defaultdict(lambda: {"sys1": [], "sys2": [], "sys3": [], "sparql_valid": []})

    total_sys1_correct = 0
    total_sys2_correct = 0
    total_sys3_correct = 0
    total_sparql_valid = 0

    for row in rows:
        hop = int(row["hop"])
        s1  = int(row["sys1_correct"]) if row["sys1_correct"] else 0
        s2  = int(row["sys2_correct"]) if row["sys2_correct"] else 0
        s3  = int(row["sys3_correct"]) if row["sys3_correct"] else 0
        sv  = int(row["sys3_query_valid"]) if row["sys3_query_valid"] else 0

        hop_data[hop]["sys1"].append(s1)
        hop_data[hop]["sys2"].append(s2)
        hop_data[hop]["sys3"].append(s3)
        hop_data[hop]["sparql_valid"].append(sv)

        total_sys1_correct += s1
        total_sys2_correct += s2
        total_sys3_correct += s3
        total_sparql_valid += sv

    n = len(rows)

    # ── TABELLA RIASSUNTIVA ───────────────────────────────────────────────────
    print("\n── ACCURATEZZA GLOBALE ──────────────────────────────────")
    print(f"  System 1 (LLM puro)  : {total_sys1_correct}/{n} = {total_sys1_correct/n*100:.1f}%")
    print(f"  System 2 (Doc RAG)   : {total_sys2_correct}/{n} = {total_sys2_correct/n*100:.1f}%")
    print(f"  System 3 (Graph RAG) : {total_sys3_correct}/{n} = {total_sys3_correct/n*100:.1f}%")
    print(f"  SPARQL validity      : {total_sparql_valid}/{n} = {total_sparql_valid/n*100:.1f}%")

    print("\n── ACCURATEZZA PER HOP ──────────────────────────────────")
    print(f"  {'Hop':<6} {'N':<5} {'Sys1':<12} {'Sys2':<12} {'Sys3':<12} {'SPARQL valid'}")
    print(f"  {'-'*60}")

    hops = sorted(hop_data.keys())
    hop_acc = {"sys1": [], "sys2": [], "sys3": []}

    for hop in hops:
        d   = hop_data[hop]
        n_h = len(d["sys1"])
        a1  = sum(d["sys1"]) / n_h if n_h else 0
        a2  = sum(d["sys2"]) / n_h if n_h else 0
        a3  = sum(d["sys3"]) / n_h if n_h else 0
        sv  = sum(d["sparql_valid"]) / n_h if n_h else 0

        hop_acc["sys1"].append(a1)
        hop_acc["sys2"].append(a2)
        hop_acc["sys3"].append(a3)

        print(f"  {hop:<6} {n_h:<5} {a1*100:>6.1f}%      {a2*100:>6.1f}%      {a3*100:>6.1f}%      {sv*100:.1f}%")

    # ── ANALISI ERRORI ────────────────────────────────────────────────────────
    error_counts = {
        "sys1": defaultdict(int),
        "sys2": defaultdict(int),
        "sys3": defaultdict(int),
    }
    for row in rows:
        for sys in ["sys1", "sys2", "sys3"]:
            err = row.get(f"{sys}_error_type", "").strip()
            if err:
                error_counts[sys][err] += 1

    print("\n── DISTRIBUZIONE ERRORI ─────────────────────────────────")
    for sys, label in [("sys1", "LLM puro"), ("sys2", "Doc RAG"), ("sys3", "Graph RAG")]:
        print(f"\n  {label}:")
        if error_counts[sys]:
            for err_type, count in sorted(error_counts[sys].items(), key=lambda x: -x[1]):
                print(f"    {err_type}: {count}")
        else:
            print("    (nessun errore classificato ancora)")

    # ── GRAFICO ACCURATEZZA VS HOP ────────────────────────────────────────────
    if HAS_MATPLOTLIB:
        fig, ax = plt.subplots(figsize=(8, 5))
        hop_labels = [f"Hop {h}" for h in hops]

        ax.plot(hop_labels, [v*100 for v in hop_acc["sys1"]],
                marker="o", label="System 1 — Parametric LLM", color="#C0392B", linewidth=2)
        ax.plot(hop_labels, [v*100 for v in hop_acc["sys2"]],
                marker="s", label="System 2 — Document RAG", color="#2980B9", linewidth=2)
        ax.plot(hop_labels, [v*100 for v in hop_acc["sys3"]],
                marker="^", label="System 3 — Graph RAG", color="#1C5D5E", linewidth=2)

        ax.set_xlabel("Hop Depth", fontsize=12)
        ax.set_ylabel("Accuracy (%)", fontsize=12)
        ax.set_title("Accuracy vs Hop Depth — AntibioticKG-RAG", fontsize=13)
        ax.set_ylim(0, 105)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig("results/accuracy_vs_hop.png", dpi=150)
        print("\n  Grafico salvato: results/accuracy_vs_hop.png")
        plt.close()
    else:
        print("\n  [INFO] Installa matplotlib per generare il grafico automaticamente.")

    print("\n" + "=" * 60)
    print("Analisi completata.")
    print("=" * 60)


if __name__ == "__main__":
    analyze()
