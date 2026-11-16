"""将 xlsx2json.py 生成的实体及关系 JSON 导入 Neo4j，文件按类型命名。

依赖：python -m pip install neo4j
先检查：python json2neo4j.py --check
再导入：python json2neo4j.py
可用 --input 指定其他 JSON 目录；密码从 NEO4J_PASSWORD 或运行时输入读取。
使用 Neo4j 5.26+。重复导入同一输出按稳定 ID 合并；分批提交，不清空数据库。
化合物 ID 由转换脚本优先使用完整 InChIKey、其次使用 PubChem CID 生成。
导入时沿用这些 ID，不另行根据名称推断化合物身份。
修改标识符规则后，请先重新运行 xlsx2json.py 生成 JSON，再执行导入。
"""
import argparse
import getpass
import json
import math
import os
from collections import defaultdict
from pathlib import Path


# =========================== 修改连接配置 ===========================
JSON_DIR = Path(__file__).resolve().parent / "json_output" / "data_source"
URI = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
USER = os.environ.get("NEO4J_USER", "neo4j")
DATABASE = os.environ.get("NEO4J_DATABASE", "neo4j")
BATCH_SIZE = 1000
# ===================================================================

# 与转换脚本使用相同本体；两份文件可各自独立运行，不依赖项目包。
ONTOLOGY = {
    "hasCompound": {("citrus", "compound")},
    "activate": {("compound", "odor"), ("compound", "taste")},
    "producedIn": {("citrus", "production_area")},
    "precursor": {("compound", "metabolism")},
    "metabolicProduction": {("metabolism", "compound")},
    "hasOdor": {("citrus", "odor")}, "hasTaste": {("citrus", "taste")},
    "hasMouthfeel": {("citrus", "mouthfeel")},
    "composedOf": {("flavor", "odor"), ("flavor", "taste"), ("flavor", "mouthfeel")},
}
LABELS = {label for pairs in ONTOLOGY.values() for pair in pairs for label in pair}


def read_records(path):
    records = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(records, list):
        raise ValueError(f"JSON 顶层应为数组: {path}")
    for item in records:
        if not isinstance(item, dict) or not isinstance(item.get("id"), str) or not item["id"]:
            raise ValueError(f"记录缺少字符串 id: {path}")
        if not isinstance(item.get("source"), list) or not item["source"] or any(not isinstance(v, str) or not v.strip() for v in item["source"]):
            raise ValueError(f"source 应为非空字符串数组: {path}")
        for key, value in item.items():
            values = value if isinstance(value, list) else [value]
            if any(not isinstance(v, (str, int, float, bool)) or isinstance(v, float) and not math.isfinite(v) for v in values):
                raise ValueError(f"Neo4j 不支持该属性值: {path.name}/{key}")
            kinds = {"number" if type(v) in (int, float) else type(v).__name__ for v in values}
            if isinstance(value, list) and len(kinds) > 1:
                raise ValueError(f"属性数组应使用相同值类型: {path.name}/{key}")
    return records


def load_json(directory):
    """在连接数据库前检查实体、关系及端点；不需要 manifest 或配置文件。"""
    directory = Path(directory)
    nodes, index, edges = {}, {}, []
    files = sorted((directory / "entities").glob("*.json"))
    if not files:
        raise ValueError(f"未找到实体 JSON: {directory / 'entities'}")
    for path in files:
        if path.stem not in LABELS:
            raise ValueError(f"未知实体类型: {path.stem}")
        nodes[path.stem] = read_records(path)
        for item in nodes[path.stem]:
            if not isinstance(item.get("name"), str) or not item["name"].strip():
                raise ValueError(f"实体缺少 name: {path}")
            if item["id"] in index:
                raise ValueError(f"重复实体 ID: {item['id']}")
            index[item["id"]] = (path.stem, item["name"])
    for path in sorted((directory / "relations").glob("*.json")):
        if path.stem not in ONTOLOGY:
            raise ValueError(f"未知关系类型: {path.stem}")
        for item in read_records(path):
            pair = (item.get("head_label"), item.get("tail_label"))
            if pair not in ONTOLOGY[path.stem]:
                raise ValueError(f"关系方向不符合本体: {path.name}")
            for side in ("head", "tail"):
                endpoint = index.get(item.get(side + "_id"))
                if endpoint is None or endpoint[0] != item[side + "_label"] or item.get(endpoint[0]) != endpoint[1]:
                    raise ValueError(f"关系端点不存在或名称不一致: {path.name}/{item['id']}")
            edges.append({**item, "type": path.stem})
    if len({r["id"] for r in edges}) != len(edges):
        raise ValueError("存在重复关系 ID")
    return nodes, edges


def write_batch(tx, query, rows):
    record = tx.run(query, rows=rows).single()
    if record is None or record["processed"] != len(rows):
        raise ValueError("本批次未能匹配全部端点，事务已回滚")


def import_json(directory, check_only=False):
    nodes, edges = load_json(directory)
    counts = {"entities": sum(map(len, nodes.values())), "relations": len(edges)}
    if check_only:
        print(f"离线检查通过：{counts}")
        return counts
    if BATCH_SIZE < 1:
        raise ValueError("BATCH_SIZE 必须大于 0")
    from neo4j import GraphDatabase       # 离线检查不需要安装 Neo4j Driver

    password = os.environ.get("NEO4J_PASSWORD") or getpass.getpass("Neo4j password: ")
    # 所有标签／关系名已由上方白名单校验；属性值均以参数传递。
    merge_source = "SET n.source = reduce(a = coalesce(n.source, []), s IN row.source | CASE WHEN s IN a THEN a ELSE a + s END)"
    with GraphDatabase.driver(URI, auth=(USER, password)) as driver:
        driver.verify_connectivity()
        with driver.session(database=DATABASE) as session:
            for label, records in nodes.items():
                session.run(f"CREATE CONSTRAINT cfkg_{label}_id IF NOT EXISTS FOR (n:`{label}`) REQUIRE n.id IS UNIQUE").consume()
                query = (f"UNWIND $rows AS row MERGE (n:`{label}` {{id: row.id}}) SET n += row.properties "
                         + merge_source + " RETURN count(n) AS processed")
                rows = [{"id": item["id"], "source": item["source"],
                         "properties": {k: v for k, v in item.items() if k not in {"id", "source"}}} for item in records]
                for start in range(0, len(rows), BATCH_SIZE):
                    session.execute_write(write_batch, query, rows[start:start + BATCH_SIZE])
            groups = defaultdict(list)
            metadata = {"id", "source", "type", "head_id", "tail_id", "head_label", "tail_label"} | LABELS
            for edge in edges:
                groups[(edge["type"], edge["head_label"], edge["tail_label"])].append({
                    "id": edge["id"], "head_id": edge["head_id"], "tail_id": edge["tail_id"],
                    "source": edge["source"], "properties": {k: v for k, v in edge.items() if k not in metadata}})
            for (kind, head, tail), rows in groups.items():
                query = (f"UNWIND $rows AS row MATCH (a:`{head}` {{id: row.head_id}}), (b:`{tail}` {{id: row.tail_id}}) "
                         f"MERGE (a)-[n:`{kind}` {{id: row.id}}]->(b) SET n += row.properties "
                         + merge_source + " RETURN count(n) AS processed")
                for start in range(0, len(rows), BATCH_SIZE):
                    session.execute_write(write_batch, query, rows[start:start + BATCH_SIZE])
    print(f"导入完成：{counts}")
    return counts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="实体／关系 JSON → Neo4j")
    parser.add_argument("--input", type=Path, default=JSON_DIR, help="包含 entities 和 relations 的目录")
    parser.add_argument("--check", action="store_true", help="只检查 JSON，不连接数据库")
    args = parser.parse_args()
    import_json(args.input, args.check)
