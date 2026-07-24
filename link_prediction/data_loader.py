import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from py2neo import Graph
from sklearn.model_selection import train_test_split


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

# Only these Neo4j node labels and relationship types are included.
# Neo4j labels and relationship types are case-sensitive.
ALLOWED_NODE_TYPES = [
    "citrus",
    "compound",
    "flavor",
    "metabolism",
    "mouthfeel",
    "odor",
    "production_area",
    "taste",
]

ALLOWED_RELATION_TYPES = [
    "activate",
    "composedOf",
    "hasCompound",
    "hasMouthfeel",
    "hasOdor",
    "hasTaste",
    "metabolicProduction",
    "precursor",
    "producedIn",
    "subordination",
]


def load_data_from_neo4j():
    print("Connecting to Neo4j...")
    graph = Graph(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    query = """
    MATCH (h)-[r]->(t)
    WHERE type(r) IN $allowed_relation_types
      AND any(label IN labels(h) WHERE label IN $allowed_node_types)
      AND any(label IN labels(t) WHERE label IN $allowed_node_types)
    WITH
        h,
        r,
        t,
        head(
            [label IN labels(h) WHERE label IN $allowed_node_types]
        ) AS head_type,
        head(
            [label IN labels(t) WHERE label IN $allowed_node_types]
        ) AS tail_type
    RETURN
        head_type + '_' + toString(id(h)) AS head_id,
        h.name AS head_name,
        head_type,
        properties(h) AS head_props,
        type(r) AS relation,
        tail_type + '_' + toString(id(t)) AS tail_id,
        t.name AS tail_name,
        tail_type,
        properties(t) AS tail_props
    """

    print("Running Neo4j query...")
    data = graph.run(
        query,
        allowed_node_types=ALLOWED_NODE_TYPES,
        allowed_relation_types=ALLOWED_RELATION_TYPES,
    ).to_data_frame()

    if data.empty:
        raise ValueError(
            "No triples matched the configured node-label and relationship-type "
            "whitelists. Check whether the Neo4j names and letter case are correct."
        )

    print(f"Loaded {len(data)} triples.")
    print(
        f"Included entities: "
        f"{len(set(data['head_id']).union(set(data['tail_id'])))}"
    )
    print(
        "Included node types: "
        f"{sorted(set(data['head_type']).union(set(data['tail_type'])))}"
    )
    print(f"Included relation types: {sorted(data['relation'].unique())}")
    return data


def normalize_properties(props):
    if not isinstance(props, dict):
        return {}

    normalized = {}
    for key, value in props.items():
        if value is None:
            continue
        if isinstance(value, (str, int, float, bool)):
            normalized[str(key)] = value
        elif isinstance(value, (list, tuple, set)):
            normalized[str(key)] = [str(item) for item in value if item is not None]
        else:
            normalized[str(key)] = str(value)
    return normalized


def add_inverse_relations(df):
    inverse_df = df.copy()
    inverse_df["head_id"] = df["tail_id"].values
    inverse_df["head_name"] = df["tail_name"].values
    inverse_df["head_type"] = df["tail_type"].values
    inverse_df["head_props"] = df["tail_props"].values
    inverse_df["tail_id"] = df["head_id"].values
    inverse_df["tail_name"] = df["head_name"].values
    inverse_df["tail_type"] = df["head_type"].values
    inverse_df["tail_props"] = df["head_props"].values
    inverse_df["relation"] = "inverse_" + df["relation"].astype(str).values

    return pd.concat([df, inverse_df], ignore_index=True)


def encode_triplets(df, entity2id, relation2id):
    triplets = []
    for _, row in df.iterrows():
        h = entity2id[row["head_id"]]
        r = relation2id[row["relation"]]
        t = entity2id[row["tail_id"]]
        triplets.append([h, r, t])

    return np.array(triplets, dtype=np.int64)


def process_data(df):
    entities = sorted(set(df["head_id"]).union(set(df["tail_id"])))
    relations = sorted(
        set(df["relation"]).union(
            {f"inverse_{relation}" for relation in df["relation"].astype(str)}
        )
    )

    entity2id = {entity: idx for idx, entity in enumerate(entities)}
    relation2id = {relation: idx for idx, relation in enumerate(relations)}

    id2name = {}
    id2type = {}
    id2attributes = {}

    for _, row in df.iterrows():
        h_idx = entity2id[row["head_id"]]
        t_idx = entity2id[row["tail_id"]]

        if h_idx not in id2name:
            id2name[h_idx] = row["head_name"] if row["head_name"] else row["head_id"]
            id2type[h_idx] = row["head_type"]
            id2attributes[h_idx] = normalize_properties(row.get("head_props", {}))

        if t_idx not in id2name:
            id2name[t_idx] = row["tail_name"] if row["tail_name"] else row["tail_id"]
            id2type[t_idx] = row["tail_type"]
            id2attributes[t_idx] = normalize_properties(row.get("tail_props", {}))

    print("Splitting triples before inverse relation expansion...")
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    valid_df, test_df = train_test_split(test_df, test_size=0.5, random_state=42)

    print("Adding inverse relations to each split...")
    train_data = encode_triplets(add_inverse_relations(train_df), entity2id, relation2id)
    valid_data = encode_triplets(add_inverse_relations(valid_df), entity2id, relation2id)
    test_data = encode_triplets(add_inverse_relations(test_df), entity2id, relation2id)

    print(f"Train: {len(train_data)}, Valid: {len(valid_data)}, Test: {len(test_data)}")

    data_pkg = {
        "train": train_data,
        "valid": valid_data,
        "test": test_data,
        "entity2id": entity2id,
        "relation2id": relation2id,
        "id2name": id2name,
        "id2type": id2type,
        "id2attributes": id2attributes,
        "num_nodes": len(entity2id),
        "num_rels": len(relation2id),
    }

    with open(DATA_DIR / "dataset.pkl", "wb") as f:
        pickle.dump(data_pkg, f)
    print(f"Dataset saved to {DATA_DIR / 'dataset.pkl'}")


if __name__ == "__main__":
    df = load_data_from_neo4j()
    process_data(df)
