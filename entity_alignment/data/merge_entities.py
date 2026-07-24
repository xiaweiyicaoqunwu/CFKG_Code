import json
import os
import sys
import time

from py2neo import Graph
from tqdm import tqdm


# ================= 配置区域 =================
# 请根据实际情况修改数据库连接信息
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "your_password")

NODES_FILE = "./data/nodes.jsonl"
ALIGNMENT_FILE = "./data/alignment_results.jsonl"

# 大模型采用 OpenAI 兼容接口。优先读取通用变量，其次读取现有服务变量。
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
LLM_API_KEY = (
    os.getenv("LLM_API_KEY")
    or DASHSCOPE_API_KEY
    or os.getenv("OPENAI_API_KEY")
)
LLM_BASE_URL = os.getenv("LLM_BASE_URL")
if not LLM_BASE_URL and DASHSCOPE_API_KEY:
    LLM_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

LLM_MODEL = os.getenv("LLM_MODEL", "qwen3-max")
LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "3"))
LLM_RETRY_BASE_SECONDS = float(
    os.getenv("LLM_RETRY_BASE_SECONDS", "2")
)
MAX_PROMPT_VALUE_LENGTH = int(
    os.getenv("MAX_PROMPT_VALUE_LENGTH", "4000")
)
# ===========================================


class PropertyResolutionError(RuntimeError):
    """属性冲突无法可靠解析时抛出的异常。"""


class AlignmentGroupBuilder:
    """将相互关联的对齐结果归并为待融合组。"""

    def __init__(self):
        self.parent = {}

    def find_group(self, node_id):
        if node_id not in self.parent:
            self.parent[node_id] = node_id
        if self.parent[node_id] != node_id:
            self.parent[node_id] = self.find_group(self.parent[node_id])
        return self.parent[node_id]

    def put_in_same_group(self, first_id, second_id):
        first_group = self.find_group(first_id)
        second_group = self.find_group(second_id)
        if first_group != second_group:
            self.parent[second_group] = first_group

    def get_groups(self):
        """返回 {group_id: [member_id1, member_id2, ...]}。"""
        groups = {}
        for node_id in self.parent:
            group_id = self.find_group(node_id)
            groups.setdefault(group_id, []).append(node_id)

        # 只有包含至少两个节点的组才需要执行融合。
        return {
            group_id: members
            for group_id, members in groups.items()
            if len(members) > 1
        }


def create_llm_client():
    """创建用于属性冲突判断的 OpenAI 兼容客户端。"""
    if not LLM_API_KEY:
        raise RuntimeError(
            "未配置大模型 API Key。请设置 LLM_API_KEY、"
            "DASHSCOPE_API_KEY 或 OPENAI_API_KEY。"
        )

    try:
        from openai import OpenAI
    except ImportError as error:
        raise RuntimeError(
            "缺少 openai SDK，请先运行：pip install openai"
        ) from error

    client_options = {"api_key": LLM_API_KEY}
    if LLM_BASE_URL:
        client_options["base_url"] = LLM_BASE_URL
    return OpenAI(**client_options)


def load_node_info(filepath):
    """加载 nodes.jsonl 到内存映射：id -> {name, label}。"""
    print(f"正在加载节点信息: {filepath} ...")
    info_map = {}
    with open(filepath, "r", encoding="utf-8") as file:
        for line in file:
            data = json.loads(line)
            info_map[data["node_id"]] = {
                "name": data["name"],
                "label": data["label"],
            }
    return info_map


def build_alignment_groups(filepath):
    """根据连锁对齐关系构建待融合组。"""
    print(f"正在解析对齐关系: {filepath} ...")
    group_builder = AlignmentGroupBuilder()

    with open(filepath, "r", encoding="utf-8") as file:
        for line in file:
            data = json.loads(line)
            main_id = data["main_id"]
            aligned_ids = data["aligned_entities"]

            # 主实体及其全部对齐实体属于同一个组。
            for aligned_id in aligned_ids:
                group_builder.put_in_same_group(main_id, aligned_id)

    return group_builder.get_groups()


def get_node_from_db(graph, label, name):
    """根据 label 和 name 从 Neo4j 获取节点。"""
    return graph.nodes.match(label, name=name).first()


def split_source_values(value):
    """将 source 的字符串或数组表示拆分为单个来源。"""
    if value is None:
        return []

    raw_values = value if isinstance(value, (list, tuple)) else [value]
    sources = []
    for raw_value in raw_values:
        if raw_value is None:
            continue
        sources.extend(
            part.strip()
            for part in str(raw_value).split(",")
            if part.strip()
        )
    return sources


def merge_property_source(nodes):
    """合并 source 属性，去重并用逗号分隔。"""
    sources = set()
    for node in nodes:
        sources.update(split_source_values(node.get("source")))
    return ",".join(sorted(sources))


def is_missing_property_value(value):
    """判断属性值是否属于空值或常见缺失值占位符。"""
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip().lower() in {"", "none", "null", "nan"}
    return False


def canonical_property_value(value):
    """生成用于候选值去重和缓存的稳定表示。"""
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        default=str,
    )


def collect_unique_property_values(nodes, property_name):
    """按出现顺序收集某个属性的非空、不重复候选值。"""
    values = []
    seen = set()

    for node in nodes:
        value = node.get(property_name)
        if is_missing_property_value(value):
            continue

        canonical_value = canonical_property_value(value)
        if canonical_value not in seen:
            seen.add(canonical_value)
            values.append(value)

    return values


def value_for_prompt(value):
    """限制传给大模型的单个候选值长度，原值本身不会被修改。"""
    text = canonical_property_value(value)
    if len(text) <= MAX_PROMPT_VALUE_LENGTH:
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return text
    return text[:MAX_PROMPT_VALUE_LENGTH] + "...[truncated]"


def parse_llm_json(content):
    """从纯 JSON 或 Markdown 代码块中解析大模型结果。"""
    if not isinstance(content, str):
        raise ValueError("大模型响应内容不是字符串")

    text = content.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("大模型响应中没有 JSON 对象")
    return json.loads(text[start : end + 1])


def choose_property_value_with_llm(
    llm_client,
    entity_label,
    entity_names,
    property_name,
    candidates,
    decision_cache,
):
    """让大模型从现有候选值中选择正确且规范的一个。"""
    cache_key = (
        entity_label,
        property_name,
        tuple(canonical_property_value(value) for value in candidates),
    )
    if cache_key in decision_cache:
        return decision_cache[cache_key]

    candidate_payload = [
        {
            "index": index,
            "value": value_for_prompt(value),
        }
        for index, value in enumerate(candidates)
    ]
    context_names = [
        str(name)
        for name in entity_names
        if not is_missing_property_value(name)
    ][:50]

    system_prompt = """
You are a data-quality expert responsible for resolving conflicting property
values during knowledge-graph entity merging. Select exactly one value from
the supplied candidates. Evaluate factual correctness first, then
standardization, precision, completeness, and suitability for structured
knowledge-graph storage. Never create, rewrite, combine, or normalize a new
value. Return only a JSON object in this form:
{"selected_index": 0, "reason": "brief reason"}
""".strip()

    user_prompt = (
        f"Entity label: {entity_label}\n"
        f"Aligned entity names: "
        f"{json.dumps(context_names, ensure_ascii=False)}\n"
        f"Property: {property_name}\n"
        f"Candidates: "
        f"{json.dumps(candidate_payload, ensure_ascii=False)}"
    )

    last_error = None
    for attempt in range(1, LLM_MAX_RETRIES + 1):
        try:
            completion = llm_client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0,
            )
            content = completion.choices[0].message.content
            result = parse_llm_json(content)
            selected_index = result.get("selected_index")

            if isinstance(selected_index, bool) or not isinstance(
                selected_index,
                int,
            ):
                raise ValueError("selected_index 必须是整数")
            if not 0 <= selected_index < len(candidates):
                raise ValueError("selected_index 超出候选值范围")

            selected_value = candidates[selected_index]
            reason = str(result.get("reason", "")).strip()
            decision_cache[cache_key] = (selected_value, reason)
            return selected_value, reason
        except Exception as error:
            last_error = error
            if attempt < LLM_MAX_RETRIES:
                wait_seconds = LLM_RETRY_BASE_SECONDS * (2 ** (attempt - 1))
                time.sleep(wait_seconds)

    raise PropertyResolutionError(
        f"属性 {property_name!r} 在重试 {LLM_MAX_RETRIES} 次后"
        f"仍无法完成大模型判断：{last_error}"
    )


def resolve_group_properties(
    group_nodes,
    entity_label,
    llm_client,
    decision_cache,
    group_id,
):
    """解析组内全部节点属性，并返回写入主节点的最终属性。"""
    property_names = sorted(
        {
            property_name
            for node in group_nodes
            for property_name in dict(node)
            if property_name != "source"
        }
    )
    entity_names = [node.get("name") for node in group_nodes]
    resolved_properties = {}

    for property_name in property_names:
        candidates = collect_unique_property_values(
            group_nodes,
            property_name,
        )
        if not candidates:
            continue

        if len(candidates) == 1:
            resolved_properties[property_name] = candidates[0]
            continue

        selected_value, reason = choose_property_value_with_llm(
            llm_client=llm_client,
            entity_label=entity_label,
            entity_names=entity_names,
            property_name=property_name,
            candidates=candidates,
            decision_cache=decision_cache,
        )
        resolved_properties[property_name] = selected_value
        print(
            f"\n[属性冲突] 组 {group_id}，属性 {property_name!r}："
            f"候选 {len(candidates)} 个，保留 "
            f"{canonical_property_value(selected_value)}"
            + (f"；原因：{reason}" if reason else "")
        )

    # source 保持原有处理方式：拆分、去重、排序后以逗号连接。
    resolved_properties["source"] = merge_property_source(group_nodes)
    return resolved_properties


def filter_conflicting_compounds(group_nodes):
    """过滤 CAS 或 PubChem ID 与组内基准节点明确冲突的化合物。"""
    if not group_nodes:
        return []

    valid_nodes = [group_nodes[0]]
    base_node = group_nodes[0]
    base_cas = base_node.get("cas_number")
    base_pubchem = base_node.get("compoundid_pubchem")

    for candidate in group_nodes[1:]:
        candidate_cas = candidate.get("cas_number")
        candidate_pubchem = candidate.get("compoundid_pubchem")

        cas_conflict = (
            not is_missing_property_value(base_cas)
            and not is_missing_property_value(candidate_cas)
            and base_cas != candidate_cas
        )
        pubchem_conflict = (
            not is_missing_property_value(base_pubchem)
            and not is_missing_property_value(candidate_pubchem)
            and base_pubchem != candidate_pubchem
        )

        if not cas_conflict and not pubchem_conflict:
            valid_nodes.append(candidate)

    return valid_nodes


def choose_head_node(group_nodes, entity_label):
    """选择稳定的主节点；最终属性仍由属性解析结果决定。"""
    if entity_label == "citrus":
        ordered_nodes = sorted(
            group_nodes,
            key=lambda node: (
                len(str(node.get("name", ""))),
                str(node.get("name", "")),
            ),
        )
        return ordered_nodes[0], ordered_nodes[1:]
    return group_nodes[0], group_nodes[1:]


def merge_group_in_neo4j(
    graph,
    head_node,
    other_nodes,
    resolved_properties,
):
    """在一个 Cypher 事务中更新属性并融合节点及关系。"""
    cypher = """
    MATCH (head)
    WHERE id(head) = $head_id
    SET head += $resolved_properties
    WITH head
    MATCH (other)
    WHERE id(other) IN $other_ids
    WITH head, collect(other) AS others
    WITH [head] + others AS nodes
    CALL apoc.refactor.mergeNodes(
        nodes,
        {properties: "discard", mergeRels: true}
    )
    YIELD node
    RETURN node
    """

    graph.run(
        cypher,
        head_id=head_node.identity,
        other_ids=[node.identity for node in other_nodes],
        resolved_properties=resolved_properties,
    ).evaluate()


def process_groups(graph, groups, node_info_map, llm_client):
    """逐组校验节点、解析属性冲突并执行融合。"""
    print("开始执行实体融合...")
    decision_cache = {}

    with tqdm(total=len(groups)) as progress_bar:
        for group_id, members in groups.items():
            try:
                group_nodes = []
                seen_identities = set()
                target_label = None

                # 从 Neo4j 获取组内节点，并保证组内 Label 一致。
                for member_id in members:
                    info = node_info_map.get(member_id)
                    if not info:
                        continue

                    name = info["name"]
                    label = info["label"]
                    if target_label is None:
                        target_label = label
                    elif target_label != label:
                        print(
                            f"\n跳过组 {group_id} 中 Label 不一致的节点："
                            f"{name!r} ({label})"
                        )
                        continue

                    db_node = get_node_from_db(graph, label, name)
                    if (
                        db_node is not None
                        and db_node.identity not in seen_identities
                    ):
                        seen_identities.add(db_node.identity)
                        group_nodes.append(db_node)

                if len(group_nodes) < 2:
                    continue

                # 化合物仍保留原有的 CAS/PubChem 明确冲突过滤规则。
                if target_label == "compound":
                    group_nodes = filter_conflicting_compounds(group_nodes)

                if len(group_nodes) < 2:
                    continue

                head_node, other_nodes = choose_head_node(
                    group_nodes,
                    target_label,
                )

                try:
                    resolved_properties = resolve_group_properties(
                        group_nodes=group_nodes,
                        entity_label=target_label,
                        llm_client=llm_client,
                        decision_cache=decision_cache,
                        group_id=group_id,
                    )
                except PropertyResolutionError as error:
                    # 无法可靠选择属性时跳过整个组，避免产生不可逆的错误融合。
                    print(f"\n跳过组 {group_id}：{error}")
                    continue

                merge_group_in_neo4j(
                    graph=graph,
                    head_node=head_node,
                    other_nodes=other_nodes,
                    resolved_properties=resolved_properties,
                )
            except Exception as error:
                print(f"\n融合组 {group_id} 时出错：{error}")
            finally:
                progress_bar.update(1)

    print("实体融合完成。")


def main():
    try:
        llm_client = create_llm_client()
        print(f"大模型属性冲突判断已启用：{LLM_MODEL}")
    except Exception as error:
        print(f"初始化大模型客户端失败：{error}")
        sys.exit(1)

    try:
        graph = Graph(
            NEO4J_URI,
            auth=(NEO4J_USER, NEO4J_PASSWORD),
        )
        graph.run("RETURN 1").evaluate()
        print("成功连接到 Neo4j。")
    except Exception as error:
        print(f"连接 Neo4j 失败：{error}")
        sys.exit(1)

    node_info = load_node_info(NODES_FILE)
    groups = build_alignment_groups(ALIGNMENT_FILE)
    print(f"共生成 {len(groups)} 个待融合的实体组。")
    process_groups(graph, groups, node_info, llm_client)


if __name__ == "__main__":
    main()
