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


def parse_correct(val):
    if val is None:
        return None
    v = str(val).strip()
    return None if v in ("", "-1") else int(v)


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
    type_hop_data = defaultdict(lambda: {"sys1": [], "sys2": [], "sys3": [], "sparql_valid": []})

    total_sys1_correct = 0
    total_sys2_correct = 0
    total_sys3_correct = 0
    total_sparql_valid = 0

    for row in rows:
        hop = int(row["hop"])


        s1 = parse_correct(row["sys1_correct"])
        s2 = parse_correct(row["sys2_correct"])
        s3 = parse_correct(row["sys3_correct"])
        if s1 is None or s2 is None or s3 is None:
            continue  # riga non ancora valutata, esclusa dai conteggi
        sv  = 1 if str(row["sys3_query_valid"]).strip().lower() in ("1", "true") else 0

        qtype = row["type"]
        type_hop_data[(qtype, hop)]["sys1"].append(s1)
        type_hop_data[(qtype, hop)]["sys2"].append(s2)
        type_hop_data[(qtype, hop)]["sys3"].append(s3)
        type_hop_data[(qtype, hop)]["sparql_valid"].append(sv)

        total_sys1_correct += s1
        total_sys2_correct += s2
        total_sys3_correct += s3
        total_sparql_valid += sv

    n = len(rows)
    print("\n── ACCURATEZZA GLOBALE ──────────────────────────────────")
    print(f"  System 1 (LLM puro)  : {total_sys1_correct}/{n} = {total_sys1_correct/n*100:.1f}%")
    print(f"  System 2 (Doc RAG)   : {total_sys2_correct}/{n} = {total_sys2_correct/n*100:.1f}%")
    print(f"  System 3 (Graph RAG) : {total_sys3_correct}/{n} = {total_sys3_correct/n*100:.1f}%")
    print(f"  SPARQL validity      : {total_sparql_valid}/{n} = {total_sparql_valid/n*100:.1f}%")
    # ── TABELLA RIASSUNTIVA ──────────────────────────────────────────────────
    print("\n── ACCURATEZZA PER TIPO E HOP (RQ2, corretta) ────────────")
    print("  Bridge, comparison e narrative hanno hop con significati diversi:")
    print("  per bridge è profondità di join SPARQL, per comparison/narrative")
    print("  è complessità concettuale — mescolarli in una sola curva è fuorviante.\n")

    types_present = sorted(set(t for (t, h) in type_hop_data.keys()))

    for qtype in types_present:
        print(f"  ── {qtype.upper()} ──")
        print(f"  {'Hop':<6} {'N':<5} {'Sys1':<12} {'Sys2':<12} {'Sys3':<12} {'SPARQL valid'}")
        print(f"  {'-'*60}")
        hops_this_type = sorted(h for (t, h) in type_hop_data.keys() if t == qtype)
        for hop in hops_this_type:
            d = type_hop_data[(qtype, hop)]
            n_h = len(d["sys1"])
            a1 = sum(d["sys1"]) / n_h if n_h else 0
            a2 = sum(d["sys2"]) / n_h if n_h else 0
            a3 = sum(d["sys3"]) / n_h if n_h else 0
            sv = sum(d["sparql_valid"]) / n_h if n_h else 0
            print(f"  {hop:<6} {n_h:<5} {a1*100:>6.1f}%      {a2*100:>6.1f}%      {a3*100:>6.1f}%      {sv*100:.1f}%")
        print()
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

    
    # ── GRAFICI ACCURATEZZA VS HOP — UNO PER TIPO ─────────────────────────────
    if HAS_MATPLOTLIB:
        colors = {"sys1": "#C0392B", "sys2": "#2980B9", "sys3": "#1C5D5E"}
        labels = {"sys1": "System 1 — Parametric LLM",
                  "sys2": "System 2 — Document RAG",
                  "sys3": "System 3 — Graph RAG"}
        markers = {"sys1": "o", "sys2": "s", "sys3": "^"}

        n_types = len(types_present)
        fig, axes = plt.subplots(1, n_types, figsize=(6 * n_types, 5), squeeze=False)
        axes = axes[0]

        for ax, qtype in zip(axes, types_present):
            hops_this_type = sorted(h for (t, h) in type_hop_data.keys() if t == qtype)
            hop_labels = [f"Hop {h}" for h in hops_this_type]

            for sys in ["sys1", "sys2", "sys3"]:
                accs = []
                for hop in hops_this_type:
                    d = type_hop_data[(qtype, hop)]
                    n_h = len(d[sys])
                    accs.append(sum(d[sys]) / n_h * 100 if n_h else 0)
                ax.plot(hop_labels, accs, marker=markers[sys], label=labels[sys],
                        color=colors[sys], linewidth=2)

            ax.set_xlabel("Hop Depth", fontsize=11)
            ax.set_ylabel("Accuracy (%)", fontsize=11)
            ax.set_title(qtype.capitalize(), fontsize=12)
            ax.set_ylim(0, 105)
            ax.grid(True, alpha=0.3)

        axes[0].legend(fontsize=9, loc="upper right")
        fig.suptitle("Accuracy vs Hop Depth, by question type — AntibioticKG-RAG", fontsize=13)
        plt.tight_layout()
        plt.savefig("results/accuracy_vs_hop.png", dpi=150)
        print("\n  Grafico salvato: results/accuracy_vs_hop.png (un pannello per tipo)")
        plt.close()
    else:
        print("\n  [INFO] Installa matplotlib per generare il grafico automaticamente.")

    print("\n" + "=" * 60)
    print("Analisi completata.")
    print("=" * 60)


if __name__ == "__main__":
    analyze()
