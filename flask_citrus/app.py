from flask import Flask, render_template, request, jsonify, send_from_directory, session, redirect, url_for
from py2neo import Graph
import json
import time
from openai import OpenAI
import os
import secrets
from functools import wraps
import torch
import pickle
from kg_services import (
    build_complex_prediction_batch,
    build_entity_colors,
    classify_prediction_results,
    load_entity_types,
    load_relationship_types,
    is_safe_read_only_cypher,
    quote_cypher_identifier,
)
from model import ComplEx

MAX_TOTAL_NODES = 5000  # 全局最大节点数限制
app = Flask(__name__)
FLASK_SECRET_KEY = os.getenv('FLASK_SECRET_KEY') or secrets.token_hex(32)
app.config['SECRET_KEY'] = FLASK_SECRET_KEY

# --- Model Configuration & Loading ---
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model")
DEVICE = torch.device('cpu')

# Global variables to hold model data
complex_model = None
graph_data_resources = None

DASHSCOPE_API_KEY = os.getenv('DASHSCOPE_API_KEY')
client = None
if DASHSCOPE_API_KEY:
    client = OpenAI(
        api_key=DASHSCOPE_API_KEY,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
else:
    print("DASHSCOPE_API_KEY is not configured; Cypher generation is disabled.")

def generate_cypher_from_gpt(question):
    if client is None:
        print("DASHSCOPE_API_KEY is not configured; cannot generate Cypher.")
        return None

    system_prompt = """
    You are a Cypher statement generation expert specifically designed for querying the Citrus Flavor Knowledge Graph.
    The knowledge graph contains the following entity types:

    - citrus: Attributes include name, classification, description, etc.
    - compound: Attributes include name, cas_number, description, etc.
    - flavor: Attributes include name, etc.
    - odor: Attribute includes name.
    - taste: Attribute includes name.
    - mouthfeel: Attribute includes name.
    - production_area: Attributes include name, soil type, annual rainfall, annual temp, etc.
    - metabolism: Attribute includes name.

    Relationships between entities include:

    - hasCompound (citrus hasCompound compound; sometimes with attributes content and unit)
    - activate (compound activate odor or taste; compound->odor, compound->taste)
    - producedIn (citrus producedIn production_area)
    - precursor (compound is a metabolism precursor)
    - metabolicProduction (metabolism metabolicProduction compound)
    - hasOdor (citrus -[hasOdor]-> odor)
    - hasTaste (citrus -[hasTaste]-> taste)
    - hasMouthfeel (citrus -[hasMouthfeel]-> mouthfeel)
    - composedOf (flavor -[composedOf]-> odor or taste)
    - subordination (citrus-[subordination]->citrus)

    Please generate the correct Cypher query statement based on the user's question.
    Requirements:
    1.Return only the Cypher statement, without any additional explanations.
    2.Use appropriate node labels and relationship types.
    3.Pay attention to the correctness of attribute names, such as soil type for production_area.
    4.Use toLower(trim(...)) for case-insensitive matching.
    5.Ensure the syntax is correct and executable.
    6.Citrus connects to odor through citrus -[hasOdor]-> odor, to taste through citrus -[hasTaste]-> taste, and to mouthfeel through citrus -[hasMouthfeel]-> mouthfeel.
    7.Prioritize fuzzy matching unless the user specifically requests otherwise.
    8.The knowledge graph content is in English. If a user asks a question in Chinese, please translate the relevant entities and relationships into English before generating the Cypher.

    Examples:
    Input: "What is the content of Hesperidin in Clementine?"
    Output:
    MATCH (c:citrus)-[r:hasCompound]->(comp:compound)
    WHERE toLower(trim(c.name)) = "clementine"
    AND toLower(trim(comp.name)) = "hesperidin"
    RETURN r.content AS content, r.unit AS unit

    Input: "What constitutes red beet flavor?"
    Output:
    MATCH (f:flavor)
    WHERE toLower(trim(f.name)) CONTAINS "red beet"
    OPTIONAL MATCH (f)-[:composedOf]->(o:odor)
    OPTIONAL MATCH (f)-[:composedOf]->(t:taste)
    RETURN collect(DISTINCT o.name) AS odors, collect(DISTINCT t.name) AS tastes

    If the user's input cannot be translated into a Cypher statement, return only "None".
    """

    user_prompt = f"问题：{question}\n请生成一条Cypher语句。"

    try:
        completion = client.chat.completions.create(
            model="qwen3-max",  # 或 gpt-4o, gpt-4-turbo 等
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3
        )
        cypher_query = completion.choices[0].message.content.strip()
        print("🤖 GPT生成Cypher：", cypher_query)
        return cypher_query
    except Exception as e:
        print("❌ 调用OpenAI失败:", e)
        return None

# Neo4j数据库连接配置
NEO4J_URI = os.getenv('NEO4J_URI', 'bolt://localhost:7687')
NEO4J_USER = os.getenv('NEO4J_USER', 'neo4j')
NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD')

# 连接池配置
graph = None
if NEO4J_PASSWORD:
    print("Connecting to Neo4j; use a read-only Neo4j account.")
    try:
        graph = Graph(
            NEO4J_URI,
            auth=(NEO4J_USER, NEO4J_PASSWORD),
            init_size=5  # 初始连接数
        )
        graph.run("RETURN 1")
        print("Neo4j连接成功")
    except Exception as e:
        print(f"Neo4j连接失败: {e}")
        graph = None
else:
    print("NEO4J_PASSWORD is not configured; database connection is disabled.")

# 动态读取 Neo4j 图谱 schema
def current_graph_schema():
    try:
        entity_types = load_entity_types(graph)
    except Exception:
        app.logger.exception("Failed to load Neo4j entity types.")
        entity_types = []

    try:
        relationships = load_relationship_types(graph)
    except Exception:
        app.logger.exception("Failed to load Neo4j relationship types.")
        relationships = []

    return entity_types, relationships


def current_entity_types():
    if graph is None:
        raise RuntimeError("Graph connection is unavailable.")
    return load_entity_types(graph)


def load_model_resources():
    """Load the ComplEx model and dataset resources."""
    global complex_model, graph_data_resources
    
    if complex_model is not None and graph_data_resources is not None:
        return

    complex_model = None
    graph_data_resources = None

    print("Loading ComplEx model resources...")
    try:
        dataset_path = os.path.join(DATA_DIR, 'dataset.pkl')
        weights_path = os.path.join(DATA_DIR, 'model_weights.pth')

        if not os.path.exists(dataset_path) or not os.path.exists(weights_path):
            print(f"Failed to load ComplEx model: model files not found in {DATA_DIR}")
            return

        with open(dataset_path, 'rb') as f:
            data = pickle.load(f)

        state_dict = torch.load(
            weights_path,
            map_location=DEVICE,
            weights_only=True,
        )
        entity_weights = state_dict['entity_emb.weight']
        relation_weights = state_dict['relation_emb.weight']

        if entity_weights.ndim != 2 or relation_weights.ndim != 2:
            raise ValueError("ComplEx embedding weights must be two-dimensional.")
        if entity_weights.shape[0] != data['num_nodes']:
            raise ValueError(
                "ComplEx entity count does not match dataset num_nodes: "
                f"{entity_weights.shape[0]} != {data['num_nodes']}."
            )
        if relation_weights.shape[0] != data['num_rels']:
            raise ValueError(
                "ComplEx relation count does not match dataset num_rels: "
                f"{relation_weights.shape[0]} != {data['num_rels']}."
            )
        if entity_weights.shape[1] != relation_weights.shape[1]:
            raise ValueError("ComplEx entity and relation embedding widths differ.")
        if entity_weights.shape[1] % 2 != 0:
            raise ValueError("ComplEx embedding width must be even.")

        embedding_dim = entity_weights.shape[1] // 2
        model = ComplEx(data['num_nodes'], data['num_rels'], embedding_dim)
        model.load_state_dict(state_dict)
        model.to(DEVICE)
        model.eval()
        
        complex_model = model
        graph_data_resources = data
        print("ComplEx model loaded successfully.")
    except Exception as e:
        complex_model = None
        graph_data_resources = None
        print(f"Failed to load ComplEx model: {e}")

# Load model on startup
with app.app_context():
    load_model_resources()

# 性能监控装饰器
def timing_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} 执行时间: {end_time - start_time:.3f}秒")
        return result
    return wrapper

def get_and_increment_visits():
    """
    使用 Flask Session 优化访问量统计。
    只有在当前会话中未访问过首页时，才会增加访问次数。
    """
    filename = 'visits.txt'
    # 检查当前 Session 是否已标记为已访问
    if session.get('visited', False):
        # 如果已标记，则不增加计数，直接读取当前总访问量返回
        try:
            with open(filename, 'r') as f:
                visits = int(f.read().strip() or 0)
        except (FileNotFoundError, ValueError):
            visits = 0
        return visits

    # 如果是新会话（未标记 'visited'）
    
    # 1. 设置 Session 标记
    session['visited'] = True
    
    # 2. 增加文件计数
    visits = 0
    try:
        if not os.path.exists(filename):
            with open(filename, 'w') as f:
                f.write('0')
        
        with open(filename, 'r+') as f:
            content = f.read().strip()
            visits = int(content) if content else 0
            visits += 1
            f.seek(0)
            f.write(str(visits))
            f.truncate()
    except Exception as e:
        print(f"❌ 统计文件读写失败: {e}")
        # 如果失败，至少返回当前计算值（可能不准确，但避免应用崩溃）
        return visits

    return visits

@app.before_request
def ensure_lang():
    if 'lang' not in session:
        session['lang'] = 'zh'

@app.route('/toggle_language')
def toggle_language():
    # 获取当前语言，默认为 zh (中文)
    current_lang = session.get('lang', 'zh')
    # 切换语言
    new_lang = 'en' if current_lang == 'zh' else 'zh'
    session['lang'] = new_lang
    # 返回到之前的页面
    return redirect(request.referrer or url_for('home'))

# --- 1. 新首页 (Home) ---
@app.route('/')
def home():
    visits = get_and_increment_visits()
    lang = session.get('lang', 'zh')
    if lang == 'zh':
        return render_template('home_zh.html', visits=visits)
    return render_template('home.html', visits=visits)

# --- 3. 关于页面 (About) ---
@app.route('/about')
def about():
    visits = get_and_increment_visits()
    lang = session.get('lang', 'zh')
    if lang == 'zh':
        return render_template('about_zh.html', visits=visits)
    return render_template('about.html', visits=visits)

# --- 4. 下载页面 (Download) ---
@app.route('/download')
def download_page():
    visits = get_and_increment_visits()
    lang = session.get('lang', 'zh')
    if lang == 'zh':
        return render_template('download_zh.html', visits=visits)
    return render_template('download.html', visits=visits)

# --- 处理文件下载请求 ---
@app.route('/download/file')
def download_file():
    # 假设文件存放在 static/data 目录下
    directory = os.path.join(app.root_path, 'static', 'data')
    return send_from_directory(directory, 'graph.json', as_attachment=True)

@app.route('/graph')
def index():
    lang = session.get('lang', 'zh')
    template = 'index.html' if lang == 'en' else 'index_zh.html'
    entity_types, relationships = current_graph_schema()
    return render_template(template, 
                         entity_types=entity_types,
                         relationships=relationships,
                         entity_colors=build_entity_colors(entity_types))


def get_node_display_value(node):
    """按 name、content、value 的优先级获取节点显示值。"""
    for property_name in ('name', 'content', 'value'):
        property_value = node.get(property_name)
        if property_value is not None and (
            not isinstance(property_value, str) or property_value.strip()
        ):
            return property_value
    return 'Unknown'


def create_entity_data(node, is_target=False, preferred_type=None):
    """创建实体数据，优化属性处理"""
    properties = dict(node)
    # 限制属性数量，避免传输过大
    important_props = {k: v for k, v in properties.items() 
                      if k in ['name', 'cas_number', 'description', 'id'] 
                      or len(str(v)) < 100}  # 只保留短属性
    
    base_label = get_node_display_value(node)
    label = f"★ {base_label}" if is_target else base_label
    entity_types = sorted(node.labels)
    if preferred_type in entity_types:
        entity_type = preferred_type
    elif entity_types:
        entity_type = entity_types[0]
    else:
        entity_type = 'unlabeled'

    data = {
        'id': node.identity,
        'label': label,
        'type': entity_type,
        'properties': important_props,
        'is_target': is_target
    }
    
    return data

def create_relationship_data(relationship):
    """创建关系数据"""
    properties = dict(relationship)
    important_props = {k: v for k, v in properties.items() 
                      if len(str(v)) < 100}
    
    return {
        'id': relationship.identity,
        'source': relationship.start_node.identity,
        'target': relationship.end_node.identity,
        'type': type(relationship).__name__,
        'properties': important_props
    }

@app.route('/search_simple', methods=['POST'])
@timing_decorator
def search_entity_simple():
    """返回指定实体 3 跳以内的子图（不会卡住版本）"""
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({'error': 'Invalid request.'}), 400
    for field in ('entity_type', 'entity_name', 'cas_number'):
        if field in data and not isinstance(data[field], str):
            return jsonify({'error': 'Invalid request.'}), 400

    entity_type = data.get('entity_type', '').strip()
    entity_name = data.get('entity_name')
    if entity_name is not None:
        entity_name = entity_name.strip()
    cas_number = data.get('cas_number')
    if cas_number is not None:
        cas_number = cas_number.strip()

    if graph is None:
        return jsonify({'error': 'Failed to load graph schema.'}), 503

    try:
        max_per_hop = 50  # 默认值
        max_per_hop_input = data.get('max_per_hop')
        if max_per_hop_input:
            try:
                # 尝试转换为整数
                value = int(max_per_hop_input)
                # 验证范围 (1-500)
                if 1 <= value <= 500:
                    max_per_hop = value
            except (TypeError, ValueError):
                # 无效输入时保持默认值
                pass
        
        hop_count = 3  # 默认3跳
        hop_count_input = data.get('hop_count')
        if hop_count_input:
            try:
                value = int(hop_count_input)
                if 1 <= value <= 5:  # 验证1-5跳
                    hop_count = value
            except (TypeError, ValueError):
                pass

        if not entity_type:
            return jsonify({'error': 'Please select the entity type!'}), 400

        try:
            entity_types = current_entity_types()
        except Exception:
            app.logger.exception("Failed to load Neo4j entity types.")
            return jsonify({'error': 'Failed to load graph schema.'}), 503
        if entity_type not in entity_types:
            return jsonify({'error': 'Unknown entity type'}), 400
        entity_label = quote_cypher_identifier(entity_type)

        # Step 1 —— 找到目标节点
        if entity_type == 'compound' and cas_number:
            match_query = """
            MATCH (n:`compound` {cas_number:$value})
            RETURN n LIMIT 1
            """
            params = {'value': cas_number}
        else:
            match_query = f"""
            MATCH (n:{entity_label})
            WHERE toLower(n.name) = toLower($value) OR n.synonyms IS NOT NULL AND toLower(n.synonyms) = toLower($value) 
            RETURN n LIMIT 1
            """
            params = {'value': entity_name}

        result = graph.run(match_query, **params).data()
        if not result:
            return jsonify({'error': '未找到对应实体'}), 404

        start_node = result[0]['n']
        start_id = start_node.identity

        # 所有节点与边存这里
        nodes = {start_id: start_node}
        rels = set()

        # 当前层的 frontier
        frontier = [start_id]

        # BFS
        for _ in range(hop_count):
            if not frontier:
                break

            hop_query = """
            MATCH (n)-[r]-(m)
            WHERE id(n) IN $ids
            RETURN n, r, m
            LIMIT $limit
            """

            hop_res = graph.run(
                hop_query,
                ids=frontier,
                limit=max_per_hop
            ).data()

            next_frontier = []

            for row in hop_res:
                n = row["n"]
                m = row["m"]
                r = row["r"]

                # 记录节点
                if n.identity not in nodes:
                    nodes[n.identity] = n
                if m.identity not in nodes:
                    nodes[m.identity] = m
                    next_frontier.append(m.identity)

                # 记录关系（避免重复）
                rels.add((r.identity, r))

            frontier = next_frontier
        
        if len(nodes) > MAX_TOTAL_NODES:
            # 保留起始节点和最近2跳
            nodes = {k: v for k, v in nodes.items() if getattr(v, 'distance', 0) <= 2}
            # 添加警告信息到响应
            warning = f"结果已截断，仅显示前{MAX_TOTAL_NODES}个节点"
            print(warning)

        # 统一格式整理输出
        entities_data = []
        for node_id, node in nodes.items():
            is_target = (node_id == start_id)
            entities_data.append(create_entity_data(
                node,
                is_target=is_target,
                preferred_type=entity_type if is_target else None,
            ))
        relationships_data = [create_relationship_data(r) for (_, r) in rels]

        return jsonify({
            "entities": entities_data,
            "relationships": relationships_data,
            "counts": {
                "entities": len(entities_data),
                "relationships": len(relationships_data)
            }
        })

    except Exception:
        app.logger.exception("Search failed.")
        return jsonify({'error': 'Search failed on server.'}), 500


@app.route('/entity_details/<int:entity_id>')
def get_entity_details(entity_id):
    if not graph:
        return jsonify({'error': 'Failed to connect database!'}), 500
    
    try:
        query = """
        MATCH (n)
        WHERE id(n) = $entity_id
        RETURN n
        """
        result = graph.run(query, entity_id=entity_id).data()
        
        if not result:
            return jsonify({'error': '实体不存在'}), 404
        
        node = result[0]['n']
        entity_data = create_entity_data(node)
        return jsonify({
            'id': node.identity,
            'type': entity_data['type'],
            'properties': dict(node)  # 返回完整属性
        })
        
    except Exception as e:
        return jsonify({'error': f'获取详情失败: {str(e)}'}), 500

@app.route('/relationship_details/<int:rel_id>')
def get_relationship_details(rel_id):
    if not graph:
        return jsonify({'error': '数据库连接失败'}), 500
    
    try:
        query = """
        MATCH ()-[r]-()
        WHERE id(r) = $rel_id
        RETURN r, startNode(r) as start, endNode(r) as end
        """
        result = graph.run(query, rel_id=rel_id).data()
        
        if not result:
            return jsonify({'error': '关系不存在'}), 404
        
        relationship = result[0]['r']
        start_node = result[0]['start']
        end_node = result[0]['end']
        
        return jsonify({
            'id': relationship.identity,
            'type': type(relationship).__name__,
            'start_node': {'id': start_node.identity, 'name': get_node_display_value(start_node)},
            'end_node': {'id': end_node.identity, 'name': get_node_display_value(end_node)},
            'properties': dict(relationship)
        })
        
    except Exception as e:
        return jsonify({'error': f'获取详情失败: {str(e)}'}), 500


@app.route('/test_connection')
def test_connection():
    """测试数据库连接和基本功能"""
    if not graph:
        return jsonify({'status': 'error', 'message': 'Failed to connect database!'})
    
    try:
        # 测试查询
        result = graph.run("MATCH (n) RETURN count(n) as count").data()
        count = result[0]['count'] if result else 0
        
        # 测试实体类型
        entity_types = graph.run("""
        CALL db.labels() YIELD label
        RETURN collect(label) as labels
        """).data()
        
        return jsonify({
            'status': 'success',
            'message': '数据库连接正常',
            'node_count': count,
            'entity_types': entity_types[0]['labels'] if entity_types else []
        })
        
    except Exception as e:
        return jsonify({'status': 'error', 'message': f'测试失败: {str(e)}'})
    
@app.route('/infer_flavor', methods=['POST'])
@timing_decorator
def infer_flavor():
    if complex_model is None:
        return jsonify({'error': 'Model not loaded correctly on server.'}), 500

    try:
        data_req = request.get_json(silent=True)
        if not isinstance(data_req, dict):
            return jsonify({'error': 'Request body must be a JSON object.'}), 400

        citrus_name = data_req.get('citrus_name')
        if not isinstance(citrus_name, str) or not citrus_name.strip():
            return jsonify({'error': 'Please input citrus name!'}), 400
        citrus_name = citrus_name.strip()

        id2name = graph_data_resources['id2name']
        id2type = graph_data_resources['id2type']
        relation2id = graph_data_resources['relation2id']

        requested_name = citrus_name.strip().casefold()
        source_matches = [
            (entity_id, entity_name, id2type[entity_id])
            for entity_id, entity_name in id2name.items()
            if (
                entity_name.casefold() == requested_name
                and id2type[entity_id] in {'citrus', 'compound'}
            )
        ]

        if not source_matches:
            return jsonify({
                'error': (
                    f"Input '{citrus_name}' not found as citrus or compound "
                    "in knowledge graph."
                )
            }), 404
        if len(source_matches) > 1:
            return jsonify({
                'error': (
                    f"Ambiguous input '{citrus_name}': multiple citrus or "
                    "compound entities share this name."
                )
            }), 409

        source_id, citrus_name, source_type = source_matches[0]

        candidates = build_complex_prediction_batch(
            source_id,
            source_type,
            id2type,
            relation2id,
        )
        if not candidates:
            return jsonify({
                'citrus': citrus_name,
                'existing_results': [],
                'new_results': [],
            })

        candidate_ids = [candidate_id for candidate_id, _, _ in candidates]
        relation_ids = [relation_id for _, _, relation_id in candidates]
        head_tensor = torch.full(
            (len(candidates),), source_id, dtype=torch.long, device=DEVICE
        )
        relation_tensor = torch.tensor(
            relation_ids, dtype=torch.long, device=DEVICE
        )
        tail_tensor = torch.tensor(
            candidate_ids, dtype=torch.long, device=DEVICE
        )

        with torch.no_grad():
            probabilities = torch.sigmoid(
                complex_model(head_tensor, relation_tensor, tail_tensor)
            ).cpu().tolist()

        raw_results = []
        for (candidate_id, relation_name, _), score in zip(
            candidates, probabilities
        ):
            if score > 0.5:
                raw_results.append({
                    'name': id2name[candidate_id],
                    'type': id2type[candidate_id],
                    'relation': relation_name,
                    'score': float(score),
                })

        existing_links = set()
        if graph:
            cypher_check = """
            MATCH (c)-[r]->(t)
            WHERE (($source_type='citrus' AND c:citrus) OR
                   ($source_type='compound' AND c:compound))
            AND toLower(c.name) = toLower($name)
            AND (t:odor OR t:taste OR t:mouthfeel)
            RETURN type(r) AS relation, labels(t) AS target_labels,
                   t.name AS target_name
            """
            check_res = graph.run(
                cypher_check,
                name=citrus_name,
                source_type=source_type,
            ).data()
            sensory_types = {'odor', 'taste', 'mouthfeel'}
            for row in check_res:
                target_name = row.get('target_name')
                if target_name is None:
                    continue
                target_types = set(row.get('target_labels') or []) & sensory_types
                for target_type in target_types:
                    existing_links.add((
                        row['relation'],
                        target_type,
                        target_name.casefold(),
                    ))

        existing_results, new_results = classify_prediction_results(
            raw_results,
            existing_links,
        )
        existing_results.sort(key=lambda x: x['score'], reverse=True)
        new_results.sort(key=lambda x: x['score'], reverse=True)

        return jsonify({
            'citrus': citrus_name,
            'existing_results': existing_results,
            'new_results': new_results[:20],
        })

    except ValueError as e:
        print(f"ComplEx inference schema error: {e}")
        return jsonify({'error': f'Inference failed: {str(e)}'}), 500
    except Exception as e:
        import traceback
        print(f"Inference error: {traceback.format_exc()}")
        return jsonify({'error': 'Inference failed on server.'}), 500

def format_query_result(result_list):
    if not result_list:
        return "未查询到结果"

    merged = {}  # 存储键 -> 值集合

    # 遍历每条记录
    for record in result_list:
        if not isinstance(record, dict):
            continue
        for k, v in record.items():
            # 统一转为字符串
            if isinstance(v, list):
                values = [str(x) for x in v if x is not None]
            else:
                values = [str(v)] if v is not None else []

            if k not in merged:
                merged[k] = set(values)
            else:
                merged[k].update(values)

    # 合并输出
    formatted_parts = []
    for k, values in merged.items():
        # 去重后排序，防止重复顺序不固定
        value_str = ", ".join(sorted(values))
        formatted_parts.append(f"{k}: {value_str}")

    return ", ".join(formatted_parts)


@app.route('/ask', methods=['POST'])
@timing_decorator
def ask_question():
    """问答系统：根据查询结果判断是否再次让大模型解释"""
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({'answer': 'Invalid request.'}), 400
    question = data.get('question')
    if not isinstance(question, str):
        return jsonify({'answer': 'Invalid request.'}), 400
    question = question.strip()
    if not question:
        return jsonify({'answer': 'Invalid request.'}), 400

    if not graph:
        return jsonify({'answer': 'Failed to connect database!'})

    lang = session.get('lang', 'zh')
    lang_name = "Chinese" if lang == 'zh' else "English"

    try:
        print(f"🟢 用户问题: {question}")

        # Step 1: 调用GPT生成Cypher
        cypher_query = generate_cypher_from_gpt(question)
        if not cypher_query:
            return jsonify({'answer': 'Failed to generate Cypher. Please try again!'})
        execute_query = cypher_query.strip().lower() != "none"
        if not execute_query:
            print("🔹 生成的Cypher为'None'，跳过数据库查询")
        elif not is_safe_read_only_cypher(cypher_query):
            app.logger.warning(
                "Generated Cypher rejected by read-only safety policy. "
                "This validation does not replace a read-only Neo4j account."
            )
            return jsonify({
                'answer': 'Generated query was rejected by the read-only safety policy.'
            })

        print(f"🧠 生成的Cypher:\n{cypher_query}")

        # Step 2: 执行 Cypher 查询（带异常处理）
        db_result = None
        
        # 尝试执行查询
        if execute_query:
            try:
                db_result = graph.run(cypher_query).data()
                print(f"💾 查询成功，返回 {len(db_result)} 条结果")
            except Exception as query_error:
                # 捕获查询异常，记录错误但继续流程
                print(f"❌ Cypher执行失败: {str(query_error)}")
                # 重要：设置db_result为None触发降级
                db_result = None

        # ================================
        # 情况 A：成功获取到非空数据库结果
        # ================================
        if db_result:  # 注意：空列表[]会被视为False
            formatted_db_info = format_query_result(db_result)
            print("🔹 使用数据库结果生成专业回答...")

            final_prompt = f"""
Please answer the question based on the information provided.

Question: {question}

Information: {formatted_db_info}

Please provide a clear, professional, accurate, and research-oriented response based on the knowledge. Your answer should not contain phrases such as "According to information in the neo4j database" or "I found in your neo4j query". Answer in {lang_name}.
"""

            completion = client.chat.completions.create(
                model="qwen3-max",
                messages=[
                    {"role": "system", "content": "You are a professional citrus breeding scientist and citrus flavor expert. Your name is Citrus Flavor BOT."},
                    {"role": "user", "content": final_prompt}
                ],
                temperature=0.2
            )

            final_answer = completion.choices[0].message.content.strip()
            return jsonify({'answer': final_answer})

        # ================================
        # 情况 B：数据库无结果/查询失败/无需查询 → 直接让模型回答
        # ================================
        else:
            # 区分不同场景的日志
            if db_result is None:
                print("🔹 数据库查询失败，降级为直接回答")
            elif not db_result:  # 空列表
                print("🔹 Neo4j查询无结果，由大模型直接回答")
            else:
                print("🔹 无需数据库查询，由大模型直接回答")

            prompt = f"{question} Answer in {lang_name}."
            completion = client.chat.completions.create(
                model="qwen3-max",
                messages=[
                    {"role": "system", "content": "You are a professional citrus breeding scientist and citrus flavor expert. Your name is Citrus Flavor BOT."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2
            )

            final_answer = completion.choices[0].message.content.strip()
            return jsonify({'answer': final_answer})

    except Exception:
        app.logger.exception("Failed to answer question.")
        return jsonify({
            'answer': 'Failed to answer question due to a server error.'
        }), 500


@app.route('/suggestions', methods=['POST'])
def get_suggestions():
    """根据用户输入前缀和可选的实体类型，从Neo4j中获取匹配的实体建议。"""
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({'error': 'Invalid request.'}), 400
    for field in ('prefix', 'entity_type'):
        if field in data and not isinstance(data[field], str):
            return jsonify({'error': 'Invalid request.'}), 400

    prefix = data.get('prefix', '').strip()
    entity_type = data.get('entity_type', '').strip()

    # 长度小于2时不进行查询
    if len(prefix) < 2:
        return jsonify([])

    if graph is None:
        return jsonify({'error': 'Failed to load graph schema.'}), 503
    
    query = ""
    # 使用 STARTS WITH 进行前缀匹配
    
    if entity_type:
        try:
            entity_types = current_entity_types()
        except Exception:
            app.logger.exception("Failed to load Neo4j entity types.")
            return jsonify({'error': 'Failed to load graph schema.'}), 503
        if entity_type not in entity_types:
            return jsonify({'error': 'Unknown entity type'}), 400
        entity_label = quote_cypher_identifier(entity_type)
        # 情况一：如果指定了实体类型，只在该类型中匹配
        query = f"""
        MATCH (n:{entity_label})
        WHERE (n.name IS NOT NULL AND toLower(n.name) CONTAINS toLower($prefix)) OR (n.synonyms IS NOT NULL AND toLower(n.synonyms) CONTAINS toLower($prefix))
        RETURN n.name AS name
        ORDER BY n.name 
        LIMIT 30
        """
    else:
        # 情况二：如果未指定实体类型，则在所有有 name 属性的节点中匹配
        # 修复 Cypher 语法: 使用 n.name IS NOT NULL 替换 EXISTS(n.name)
        query = """
        MATCH (n) 
        WHERE (n.name IS NOT NULL AND toLower(n.name) CONTAINS toLower($prefix)) OR (n.synonyms IS NOT NULL AND toLower(n.synonyms) CONTAINS toLower($prefix))
        RETURN labels(n) AS labels, n.name AS name
        ORDER BY n.name 
        LIMIT 40
        """
    
    try:
        # 执行查询，prefix作为参数传入
        results = graph.run(query, prefix=prefix).data()
        
        # 格式化结果
        if entity_type:
            suggestions = [
                {'name': record['name'], 'type': entity_type}
                for record in results
            ]
        else:
            suggestions = []
            for record in results:
                labels = sorted(record.get('labels') or [])
                suggestions.append({
                    'name': record['name'],
                    'type': labels[0] if labels else 'unlabeled',
                })
        
        return jsonify(suggestions)

    except Exception as e:
        import traceback
        print(f"❌ Neo4j suggestion error: {traceback.format_exc()}")
        return jsonify([])


if __name__ == '__main__':
    app.run(debug=True, port=5000)
