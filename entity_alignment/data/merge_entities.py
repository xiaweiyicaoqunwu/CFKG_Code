import json
import sys
import os
from py2neo import Graph, Node
from tqdm import tqdm

# ================= 配置区域 =================
# 请根据实际情况修改数据库连接信息
NEO4J_URI = os.getenv('NEO4J_URI', 'bolt://localhost:7687')
NEO4J_USER = os.getenv('NEO4J_USER', 'neo4j')
NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD', 'your_password') 

NODES_FILE = "./data/nodes.jsonl"
ALIGNMENT_FILE = "./data/alignment_results.jsonl"
# ===========================================

class UnionFind:
    """并查集类，用于处理节点的连锁对齐关系"""
    def __init__(self):
        self.parent = {}

    def find(self, i):
        if i not in self.parent:
            self.parent[i] = i
        if self.parent[i] != i:
            self.parent[i] = self.find(self.parent[i])
        return self.parent[i]

    def union(self, i, j):
        root_i = self.find(i)
        root_j = self.find(j)
        if root_i != root_j:
            self.parent[root_j] = root_i

    def get_clusters(self):
        """返回所有连通分量 {root_id: [member_id1, member_id2, ...]}"""
        clusters = {}
        for node in self.parent:
            root = self.find(node)
            if root not in clusters:
                clusters[root] = []
            clusters[root].append(node)
        
        # 过滤掉只有单个节点的分组（无需对齐）
        return {k: v for k, v in clusters.items() if len(v) > 1}

def load_node_info(filepath):
    """加载 nodes.jsonl 到内存映射: id -> {name, label}"""
    print(f"正在加载节点信息: {filepath} ...")
    info_map = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            info_map[data['node_id']] = {
                'name': data['name'],
                'label': data['label']
            }
    return info_map

def build_alignment_clusters(filepath):
    """读取对齐文件并使用并查集构建融合簇"""
    print(f"正在解析对齐关系: {filepath} ...")
    uf = UnionFind()
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            main_id = data['main_id']
            aligned_ids = data['aligned_entities']
            
            # 将主实体和所有候选实体放入同一个集合
            for aid in aligned_ids:
                uf.union(main_id, aid)
    
    return uf.get_clusters()

def get_node_from_db(graph, label, name):
    """根据 label 和 name 从 Neo4j 获取节点"""
    # 注意：Neo4j 中 name 必须是唯一的或者是我们需要匹配的键
    # 使用 graph.nodes.match 查找
    node = graph.nodes.match(label, name=name).first()
    return node

def merge_property_source(nodes):
    """合并 source 属性，去重并用逗号分隔"""
    sources = set()
    for n in nodes:
        val = n.get("source")
        if val:
            # 假设原数据中 source 可能是 "A" 或 "A,B"
            parts = [s.strip() for s in val.split(',')]
            sources.update(parts)
    
    return ",".join(sorted(sources))

def process_clusters(graph, clusters, node_info_map):
    """执行具体的对齐逻辑"""
    print("开始执行实体对齐...")
    
    # 进度条
    pbar = tqdm(total=len(clusters))
    
    for root_id, members in clusters.items():
        # 1. 获取该组所有节点的基础信息
        cluster_nodes = [] # 存储 (node_id, py2neo_node_object)
        
        # 预检查：获取组内第一个有效信息的label，用于一致性校验
        target_label = None
        
        # 2. 从 Neo4j 查询节点对象
        for mid in members:
            if mid not in node_info_map:
                continue # nodes.jsonl 中没有该ID，跳过
            
            info = node_info_map[mid]
            name = info['name']
            label = info['label']
            
            if target_label is None:
                target_label = label
            elif target_label != label:
                # 违反要求3：候选实体与主实体label必须相同。
                # 在并查集逻辑下，这意味着整个组的 label 应该一致。
                # 如果出现不一致，我们可以选择跳过该特定节点，或者记录错误。
                # 这里选择跳过该节点
                continue

            db_node = get_node_from_db(graph, label, name)
            if db_node:
                cluster_nodes.append(db_node)
        
        if len(cluster_nodes) < 2:
            pbar.update(1)
            continue

        # 3. 处理 Compound 特殊逻辑 (Requirement 5)
        if target_label == "compound":
            # 筛选逻辑：两两之间不能有冲突。
            # 简单策略：选定一个基准（第一个），剔除与之冲突的。
            # 或者更严格：只要组内存在任意冲突，就不合并？
            # 题目描述："若对齐前的两节点...不相同，则不执行对齐"。
            # 意味着我们要过滤掉那些导致冲突的节点。
            
            valid_nodes = []
            # 我们以列表中的第一个节点作为基准（Base）
            base_node = cluster_nodes[0]
            valid_nodes.append(base_node)
            
            base_cas = base_node.get("cas_number")
            base_pubchem = base_node.get("compoundid_pubchem")
            
            for i in range(1, len(cluster_nodes)):
                candidate = cluster_nodes[i]
                cand_cas = candidate.get("cas_number")
                cand_pubchem = candidate.get("compoundid_pubchem")
                
                conflict = False
                # CAS Check
                if base_cas and cand_cas and base_cas != "None" and cand_cas != "None" and base_cas != cand_cas:
                    conflict = True
                # PubChem Check
                if base_pubchem and cand_pubchem and base_pubchem != "None" and cand_pubchem != "None" and base_pubchem != cand_pubchem:
                    conflict = True
                
                if not conflict:
                    valid_nodes.append(candidate)
            
            cluster_nodes = valid_nodes

        if len(cluster_nodes) < 2:
            pbar.update(1)
            continue

        # 4. 确定主节点 (Head Node) 和处理 Citrus 命名 (Requirement 4)
        head_node = None
        other_nodes = []
        
        merged_source = merge_property_source(cluster_nodes)
        
        if target_label == "citrus":
            # 找出名字最短的节点作为 Head
            # 按 name 长度排序，长度相同按 name 字母序（保证确定性）
            sorted_nodes = sorted(cluster_nodes, key=lambda n: (len(n["name"]), n["name"]))
            head_node = sorted_nodes[0]
            others = sorted_nodes[1:]
            
            # 处理 synonyms
            original_synonyms = head_node.get("synonyms", "")
            syn_set = set([s.strip() for s in original_synonyms.split(',') if s.strip()])
            
            for n in others:
                syn_set.add(n["name"])
                # 也要保留被合并节点的原有同义词
                n_syn = n.get("synonyms")
                if n_syn:
                    syn_set.update([s.strip() for s in n_syn.split(',') if s.strip()])
            
            # 从 synonyms 中移除主节点现在的名字
            if head_node["name"] in syn_set:
                syn_set.remove(head_node["name"])
            
            new_synonyms = ",".join(sorted(syn_set))
            
            # 更新 Head 节点的属性 (在内存中更新，稍后推送到 DB)
            head_node["synonyms"] = new_synonyms
            other_nodes = others
        else:
            # 非 citrus，默认取第一个作为 Head
            head_node = cluster_nodes[0]
            other_nodes = cluster_nodes[1:]

        # 5. 应用合并后的 Source 属性
        head_node["source"] = merged_source
        
        # 6. 将主节点的属性更新推送到 Neo4j (为了让 APOC merge 时保留这些新值)
        graph.push(head_node)
        
        # 7. 调用 APOC 进行融合
        # 构建节点列表，Head 必须在第一个
        nodes_to_merge = [head_node] + other_nodes
        
        # 使用 apoc.refactor.mergeNodes
        # properties: "discard" 表示如果属性冲突，保留列表第一个节点(head_node)的属性
        # mergeRels: true 表示合并关系
        try:
            query = """
            CALL apoc.refactor.mergeNodes($nodes, {properties: "discard", mergeRels: true})
            YIELD node
            RETURN count(node)
            """
            # py2neo 的 run 方法不支持直接传 Node 对象列表作为参数供 apoc 使用
            # 我们需要传递 node 的 id 或者 identity。
            # 但是 py2neo 的 Node 对象在 graph.push 后有 identity。
            # APOC mergeNodes 接收的是 List<Node>。
            # 在 py2neo 中，我们可以通过传递 id 列表，然后在 cypher 中 match 出来传给 APOC
            
            node_ids = [n.identity for n in nodes_to_merge]
            
            cypher_execution = """
            MATCH (n) WHERE id(n) IN $ids
            WITH collect(n) as nodes
            // 确保 nodes 列表顺序与 ids 顺序一致，因为第一个要是 head
            // 上面的 collect 顺序不一定保证，所以我们需要根据输入 ids 手动排序或者假设
            // 更稳妥的方式是单独 MATCH head 和 others
            CALL apoc.refactor.mergeNodes(nodes, {properties: "discard", mergeRels: true})
            YIELD node
            RETURN node
            """
            
            # 为了保证顺序（Head 第一），我们修改 Cypher
            cypher_ordered = """
            MATCH (head) WHERE id(head) = $head_id
            MATCH (other) WHERE id(other) IN $other_ids
            WITH head, collect(other) as others
            WITH head + others as nodes
            CALL apoc.refactor.mergeNodes(nodes, {properties: "discard", mergeRels: true})
            YIELD node
            RETURN node
            """
            
            other_ids = [n.identity for n in other_nodes]
            graph.run(cypher_ordered, head_id=head_node.identity, other_ids=other_ids)
            
        except Exception as e:
            print(f"Error merging cluster {root_id}: {e}")

        pbar.update(1)
    
    pbar.close()
    print("实体对齐完成。")

def main():
    # 1. 连接数据库
    try:
        graph = Graph(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        # 测试连接
        _ = graph.run("RETURN 1").evaluate()
        print("成功连接到 Neo4j。")
    except Exception as e:
        print(f"连接 Neo4j 失败: {e}")
        sys.exit(1)

    # 2. 加载节点元数据
    node_info = load_node_info(NODES_FILE)

    # 3. 构建对齐组 (处理连锁反应)
    clusters = build_alignment_clusters(ALIGNMENT_FILE)
    print(f"共生成 {len(clusters)} 个待合并的实体组。")

    # 4. 执行对齐
    process_clusters(graph, clusters, node_info)

if __name__ == "__main__":
    main()