"""
02b_materialize_inferences.py
------------------------------
Materializza le inferenze OWL sul Knowledge Graph già costruito da 02_build_kg.py.

A differenza di Protégé + HermiT (che calcola le inferenze solo "a video",
in memoria, senza mai scriverle su disco), questo script usa la libreria
owlrl per calcolare le stesse inferenze e SALVARLE fisicamente in un nuovo
file .ttl — pronto per essere interrogato con SPARQL da 03_systems.py
(Sistema 3 — Graph RAG), che altrimenti non vedrebbe triple come citedBy,
la direzione mancante di coAuthorWith, o l'appartenenza di Author a
foaf:Person.

Le inferenze materializzate coprono:
  - owl:inverseOf       (cites -> citedBy)
  - owl:SymmetricProperty (coAuthorWith in entrambe le direzioni)
  - rdfs:subClassOf     (Author -> foaf:Person, sottoclassi di Institution)
  - owl:disjointWith    (nessun effetto sulle triple normali, serve solo
                          per la verifica di consistenza, non genera nuove
                          triple utili alle query)

Input:
  data/antibiotic_kg.ttl              : il grafo "asserito", generato da 02_build_kg.py

Output:
  data/antibiotic_kg_inferred.ttl     : il grafo arricchito con le triple inferite,
                                         da usare nel Sistema 3 (Graph RAG)
"""

from rdflib import Graph
import owlrl

INPUT_PATH  = "data/antibiotic_kg.ttl"
OUTPUT_PATH = "data/antibiotic_kg_inferred.ttl"


def materialize_inferences():
    print("=" * 60)
    print("AntibioticKG-RAG — Materializzazione inferenze OWL")
    print("=" * 60)

    print(f"\n[1/3] Caricamento grafo asserito da {INPUT_PATH}...")
    g = Graph()
    g.parse(INPUT_PATH, format="turtle")
    triples_before = len(g)
    print(f"  Triple asserite (prima dell'inferenza): {triples_before}")

    print(f"\n[2/3] Calcolo inferenze OWL-RL (owlrl)...")
    # OWLRL_Semantics copre inverseOf, SymmetricProperty, subClassOf,
    # equivalentClass e altri costrutti OWL-RL standard.
    owlrl.DeductiveClosure(owlrl.OWLRL_Semantics).expand(g)
    triples_after = len(g)
    new_triples = triples_after - triples_before
    print(f"  Triple totali (dopo l'inferenza): {triples_after}")
    print(f"  Nuove triple inferite: {new_triples}")

    print(f"\n[3/3] Filtraggio triple non valide e salvataggio...")
    from rdflib import URIRef, BNode

    OWL_NOTHING = URIRef("http://www.w3.org/2002/07/owl#Nothing")

    g_clean = Graph()
    for prefix, namespace in g.namespaces():
        g_clean.bind(prefix, namespace)

    removed = 0
    for s, p, o in g:
        if not isinstance(s, (URIRef, BNode)):
            removed += 1
            continue
        if s == OWL_NOTHING or o == OWL_NOTHING:
            removed += 1
            continue
        g_clean.add((s, p, o))

    print(f"  Triple scartate (soggetto non valido o owl:Nothing): {removed}")
    print(f"  Triple valide rimaste: {len(g_clean)}")

    g_clean.serialize(destination=OUTPUT_PATH, format="turtle", encoding="utf-8")






    print("\n" + "=" * 60)
    print("MATERIALIZZAZIONE COMPLETATA")
    print(f"  Triple asserite originali : {triples_before}")
    print(f"  Triple inferite aggiunte  : {new_triples}")
    print(f"  Triple totali finali      : {triples_after}")
    print(f"  File salvato              : {OUTPUT_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    materialize_inferences()
