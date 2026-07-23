import os
from dotenv import load_dotenv

load_dotenv()

# Neo4j 配置
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

# 大模型配置
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
MODEL_NAME = "qwen3-max"
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

# 路径配置
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

NODES_PATH = f"{DATA_DIR}/nodes.jsonl"
EMBEDDINGS_PATH = f"{DATA_DIR}/embeddings.pt"
CANDIDATES_PATH = f"{DATA_DIR}/candidates.jsonl"
ALIGNMENT_RESULTS_PATH = f"{DATA_DIR}/alignment_results.jsonl"

# 处理参数
BATCH_SIZE = 500  # Neo4j批处理大小
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
SIMILARITY_THRESHOLD = 0.7  # 预过滤阈值
TOP_K = 10  # 候选实体数量
NAME_WEIGHT = 0.8
STRUCTURE_WEIGHT = 0.2