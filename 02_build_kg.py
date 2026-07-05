"""
02_build_kg.py
--------------
Costruisce il Knowledge Graph RDF/Turtle dai dati raccolti da OpenAlex.

Input:
  data/papers.json
  data/authors.json
  data/edges.json

Output:
  data/antibiotic_kg.ttl   : il Knowledge Graph completo in formato Turtle
"""

import json
from rdflib import Graph, Namespace, URIRef, Literal, RDF, RDFS, OWL, XSD

# ── NAMESPACE ─────────────────────────────────────────────────────────────────
AKG  = Namespace("http://antibiotickg.org/ontology#")
ARES = Namespace("http://antibiotickg.org/resource/")

def clean_uri(text: str) -> str:
    """Trasforma un testo in una stringa usabile come URI."""
    return text.strip().replace(" ", "_").replace("/", "-").replace(":", "-")


def build_kg():
    print("=" * 60)
    print("AntibioticKG-RAG — Costruzione Knowledge Graph RDF")
    print("=" * 60)

    # Carica i dati
    with open("data/papers.json") as f:
        papers = json.load(f)
    with open("data/authors.json") as f:
        authors = json.load(f)
    with open("data/edges.json") as f:
        edges = json.load(f)

    g = Graph()
    g.bind("akg",  AKG)
    g.bind("ares", ARES)
    g.bind("owl",  OWL)
    g.bind("xsd",  XSD)

    # ── DEFINIZIONE ONTOLOGIA (classi e proprietà) ────────────────────────────
    print("\n[1/4] Definizione ontologia...")

    # Classi
    for cls_name in ["Paper", "Author", "Institution", "Topic"]:
        cls = AKG[cls_name]
        g.add((cls, RDF.type, OWL.Class))
        g.add((cls, RDFS.label, Literal(cls_name)))

    # Proprietà oggetto
    obj_props = {
        "cites":        ("Paper",       "Paper"),
        "hasAuthor":    ("Paper",       "Author"),
        "affiliatedWith": ("Author",    "Institution"),
        "about":        ("Paper",       "Topic"),
    }
    for prop_name, (domain, range_) in obj_props.items():
        prop = AKG[prop_name]
        g.add((prop, RDF.type,        OWL.ObjectProperty))
        g.add((prop, RDFS.domain,     AKG[domain]))
        g.add((prop, RDFS.range,      AKG[range_]))
        g.add((prop, RDFS.label,      Literal(prop_name)))

    # Proprietà datatype
    dt_props = {
        "hasTitle":      ("Paper",  XSD.string),
        "hasYear":       ("Paper",  XSD.integer),
        "hasCitationCount": ("Paper", XSD.integer),
        "hasDOI":        ("Paper",  XSD.string),
        "hasName":       ("Author", XSD.string),
        "hasInstitutionName": ("Institution", XSD.string),
        "hasTopicName":  ("Topic",  XSD.string),
    }
    for prop_name, (domain, xsd_type) in dt_props.items():
        prop = AKG[prop_name]
        g.add((prop, RDF.type,    OWL.DatatypeProperty))
        g.add((prop, RDFS.domain, AKG[domain]))
        g.add((prop, RDFS.range,  xsd_type))

    print(f"  Classi definite: 4")
    print(f"  Proprietà oggetto: {len(obj_props)}")
    print(f"  Proprietà datatype: {len(dt_props)}")

    # ── ISTANZE: AUTORI ───────────────────────────────────────────────────────
    print("\n[2/4] Generazione triple: Autori e Istituzioni...")
    institutions = {}  # nome -> URI

    for author in authors:
        author_uri = ARES[f"author/{clean_uri(author['id'])}"]
        g.add((author_uri, RDF.type,       AKG.Author))
        g.add((author_uri, AKG.hasName,    Literal(author["name"])))

        for inst_name in author.get("institutions", []):
            if not inst_name:
                continue
            if inst_name not in institutions:
                inst_uri = ARES[f"institution/{clean_uri(inst_name)}"]
                g.add((inst_uri, RDF.type,                AKG.Institution))
                g.add((inst_uri, AKG.hasInstitutionName,  Literal(inst_name)))
                institutions[inst_name] = inst_uri

            g.add((author_uri, AKG.affiliatedWith, institutions[inst_name]))

    print(f"  Autori: {len(authors)}")
    print(f"  Istituzioni: {len(institutions)}")

    # ── ISTANZE: PAPER E TOPIC ────────────────────────────────────────────────
    print("\n[3/4] Generazione triple: Paper e Topic...")
    topics = {}  # nome -> URI

    for paper in papers:
        paper_uri = ARES[f"paper/{clean_uri(paper['id'])}"]
        g.add((paper_uri, RDF.type,               AKG.Paper))
        g.add((paper_uri, AKG.hasTitle,            Literal(paper["title"] or "")))
        g.add((paper_uri, AKG.hasCitationCount,    Literal(paper["cited_by_count"], datatype=XSD.integer)))

        if paper.get("doi"):
            g.add((paper_uri, AKG.hasDOI, Literal(paper["doi"])))

        if paper.get("year"):
            g.add((paper_uri, AKG.hasYear, Literal(paper["year"], datatype=XSD.integer)))

        # collega autori
        for a in paper.get("authors", []):
            if a["id"]:
                author_uri = ARES[f"author/{clean_uri(a['id'])}"]
                g.add((paper_uri, AKG.hasAuthor, author_uri))

        # collega topic/concetti
        for concept in paper.get("concepts", []):
            if concept not in topics:
                topic_uri = ARES[f"topic/{clean_uri(concept)}"]
                g.add((topic_uri, RDF.type,         AKG.Topic))
                g.add((topic_uri, AKG.hasTopicName, Literal(concept)))
                topics[concept] = topic_uri
            g.add((paper_uri, AKG.about, topics[concept]))

    print(f"  Paper: {len(papers)}")
    print(f"  Topic: {len(topics)}")

    # ── RELAZIONI DI CITAZIONE ────────────────────────────────────────────────
    print("\n[4/4] Generazione triple: Citazioni...")
    paper_ids = {p["id"] for p in papers}
    valid_edges = 0

    for edge in edges:
        src = edge["source"]
        tgt = edge["target"]
        # includi solo relazioni tra paper nel grafo
        if src in paper_ids and tgt in paper_ids:
            src_uri = ARES[f"paper/{clean_uri(src)}"]
            tgt_uri = ARES[f"paper/{clean_uri(tgt)}"]
            g.add((src_uri, AKG.cites, tgt_uri))
            valid_edges += 1

    print(f"  Relazioni cita valide: {valid_edges} / {len(edges)}")

    # ── SALVATAGGIO ───────────────────────────────────────────────────────────
    output_path = "data/antibiotic_kg.ttl"
    g.serialize(destination=output_path, format="turtle")

    print("\n" + "=" * 60)
    print("KNOWLEDGE GRAPH COSTRUITO")
    print(f"  Triple totali  : {len(g)}")
    print(f"  File salvato   : {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    build_kg()
