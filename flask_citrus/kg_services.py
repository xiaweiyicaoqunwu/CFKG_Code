import hashlib


UNSAFE_CYPHER_KEYWORDS = frozenset(
    {
        "CREATE",
        "MERGE",
        "DELETE",
        "DETACH",
        "SET",
        "REMOVE",
        "DROP",
        "CALL",
        "LOAD",
        "FOREACH",
        "USE",
        "SHOW",
        "GRANT",
        "DENY",
        "REVOKE",
        "ALTER",
        "INSERT",
        "START",
        "STOP",
        "TERMINATE",
        "PROFILE",
        "EXPLAIN",
        "UNION",
    }
)


PREDICTION_RULES = {
    "citrus": {
        "odor": "hasOdor",
        "taste": "hasTaste",
        "mouthfeel": "hasMouthfeel",
    },
    "compound": {
        "odor": "activate",
        "taste": "activate",
    },
}


def color_for_entity_type(entity_type):
    digest = hashlib.sha256(entity_type.encode("utf-8")).digest()
    hue = int.from_bytes(digest[:2], "big") % 360
    saturation = 65 + digest[2] % 16
    lightness = 45 + digest[3] % 11
    return f"hsl({hue}, {saturation}%, {lightness}%)"


def build_entity_colors(entity_types):
    return {
        entity_type: color_for_entity_type(entity_type)
        for entity_type in sorted(set(entity_types))
    }


def quote_cypher_identifier(identifier):
    normalized_identifier = identifier.replace("\\u0060", "`")
    return f"`{normalized_identifier.replace('`', '``')}`"


def _cypher_tokens_outside_literals(query):
    tokens = []
    delimiter = None
    index = 0

    while index < len(query):
        character = query[index]

        if delimiter is not None:
            if character == "\\" and index + 1 < len(query):
                index += 2
                continue
            if character == delimiter:
                if index + 1 < len(query) and query[index + 1] == delimiter:
                    index += 2
                    continue
                delimiter = None
            index += 1
            continue

        if character in {"'", '"', "`"}:
            delimiter = character
            index += 1
            continue

        if character.isalpha() or character == "_":
            start = index
            index += 1
            while index < len(query) and (
                query[index].isalnum() or query[index] == "_"
            ):
                index += 1
            tokens.append(query[start:index].upper())
            continue

        index += 1

    return None if delimiter is not None else tokens


def is_safe_read_only_cypher(query):
    if not isinstance(query, str) or not query.strip():
        return False
    if ";" in query or "//" in query or "/*" in query or "*/" in query:
        return False

    tokens = _cypher_tokens_outside_literals(query)
    if not tokens:
        return False
    if tokens[0] == "MATCH":
        pass
    elif tokens[:2] == ["OPTIONAL", "MATCH"]:
        pass
    else:
        return False
    if "RETURN" not in tokens:
        return False
    return UNSAFE_CYPHER_KEYWORDS.isdisjoint(tokens)


def load_entity_types(graph_connection):
    if graph_connection is None:
        return []

    label_rows = graph_connection.run(
        "CALL db.labels() YIELD label RETURN label ORDER BY label"
    ).data()
    return [row["label"] for row in label_rows]


def load_relationship_types(graph_connection):
    if graph_connection is None:
        return []

    relationship_rows = graph_connection.run(
        "CALL db.relationshipTypes() YIELD relationshipType "
        "RETURN relationshipType ORDER BY relationshipType"
    ).data()
    return [row["relationshipType"] for row in relationship_rows]


def load_graph_schema(graph_connection):
    return (
        load_entity_types(graph_connection),
        load_relationship_types(graph_connection),
    )


def build_complex_prediction_batch(source_id, source_type, id2type, relation2id):
    rules = PREDICTION_RULES.get(source_type)
    if rules is None:
        raise ValueError(f"Unsupported prediction source type: {source_type}")

    candidates = []
    for candidate_id, candidate_type in sorted(id2type.items()):
        relation_name = rules.get(candidate_type)
        if candidate_id == source_id or relation_name is None:
            continue
        if relation_name not in relation2id:
            raise ValueError(f"Relation '{relation_name}' not found in model schema.")
        candidates.append(
            (candidate_id, relation_name, relation2id[relation_name])
        )
    return candidates


def classify_prediction_results(results, existing_links):
    normalized_links = {
        (relation, target_type, name.casefold())
        for relation, target_type, name in existing_links
    }
    existing = []
    new = []
    for result in results:
        key = (
            result["relation"],
            result["type"],
            result["name"].casefold(),
        )
        (existing if key in normalized_links else new).append(result)
    return existing, new
