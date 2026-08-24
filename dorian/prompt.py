from itertools import groupby

from jinja2 import Template
from SPARQLWrapper import JSON, SPARQLWrapper


def make_zeroshot_prompt(
    template_path: str, text: str, entity_idxs: tuple[int, int]
) -> str:
    template = Template(open(template_path, "r").read())

    # preprocess the text
    i, j = entity_idxs
    entity = f"[[{text[i : j + 1]}]]"
    text = text[:i] + f"{entity}" + text[j + 1 :]

    return template.render(text=text, entity=entity)


def make_dorian_prompt(
    template_path: str,
    query_path: str,
    text: str,
    entity_idxs: tuple[int, int],
    document_source: str,
    endpoint: str,
) -> str:
    # preprocess the text
    i, j = entity_idxs
    entity = text[i : j + 1]

    # query the graph
    query_template = Template(open(query_path, "r").read())
    query = query_template.render(entity=entity, source=document_source)

    try:
        sparql = SPARQLWrapper(endpoint)
        sparql.setReturnFormat(JSON)
        sparql.setQuery(query)
        ret = sparql.queryAndConvert()

        rows = [
            {var: value["value"] for var, value in bind.items()}
            for bind in ret["results"]["bindings"]
        ]
        rows = sorted(rows, key=lambda d: d["eventId"])

        triples = []
        for eid, event_rows in groupby(rows, lambda d: d["eventId"]):
            event_rows = list(event_rows)
            arguments = {(er["argument"], er["argumentRole"]) for er in event_rows}
            triples.append(
                {
                    "frame": event_rows[0]["eventFrame"],
                    "activation": event_rows[0]["frameActivation"],
                    "arguments": list(arguments),
                    "eid": eid,
                }
            )
    except Exception:
        triples = []

    # preprocess the text
    template = Template(open(template_path, "r").read())
    i, j = entity_idxs
    entity = f"[[{entity}]]"
    text = text[:i] + f"{entity}" + text[j + 1 :]

    return template.render(
        text=text,
        entity=entity,
        triples=triples,
    )
