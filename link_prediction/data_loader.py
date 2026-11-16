import os
import pickle
from pathlib import Path

import numpy as np
from py2neo import Graph
from sklearn.model_selection import train_test_split


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")


def load_data_from_neo4j():
    print("Connecting to Neo4j...")
    graph = Graph(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    query = """
    MATCH (h)-[r]->(t)
    RETURN
        labels(h)[0] + '_' + toString(id(h)) AS head_id,
        h.name AS head_name,
        labels(h)[0] AS head_type,
        properties(h) AS head_props,
        type(r) AS relation,
        labels(t)[0] + '_' + toString(id(t)) AS tail_id,
        t.name AS tail_name,
        labels(t)[0] AS tail_type,
        properties(t) AS tail_props
    """

    print("Running Neo4j query...")
    data = graph.run(query).to_data_frame()
    print(f"Loaded {len(data)} triples.")
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
    # 只使用 Neo4j 中实际存在的关系类型，不额外生成 inverse_ 关系。
    relations = sorted(set(df["relation"]))

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

    # 保留原始关系方向，按 80% / 10% / 10% 划分三元组。
    print("Splitting original triples (train/valid/test = 8:1:1)...")
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    valid_df, test_df = train_test_split(test_df, test_size=0.5, random_state=42)

    train_data = encode_triplets(train_df, entity2id, relation2id)
    valid_data = encode_triplets(valid_df, entity2id, relation2id)
    test_data = encode_triplets(test_df, entity2id, relation2id)

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
