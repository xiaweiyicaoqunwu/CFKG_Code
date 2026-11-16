import ast
import logging
import os
import json
import time
from openai import OpenAI
from tqdm import tqdm
from config import *
from utils import load_processed_ids

logger = logging.getLogger(__name__)


def parse_alignment_response(response, candidate_names):
    """验证响应，并兼容旧提示词产生的单引号字典及 Markdown 代码块。"""
    if not isinstance(response, str) or not response.strip():
        raise ValueError("Empty model response")
    text = response.strip()
    if text == "None":  # 兼容旧版的无匹配响应
        return []
    if text.startswith("```") and text.endswith("```"):
        text = "\n".join(text.splitlines()[1:-1]).strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        try:
            # 仅解析字面量，禁止使用 eval 执行模型输出。
            parsed = ast.literal_eval(text)
        except (ValueError, SyntaxError) as error:
            raise ValueError("Model response is not a JSON object or literal dictionary") from error
    if not isinstance(parsed, dict) or "aligned_entities" not in parsed:
        raise ValueError("Response must contain aligned_entities")
    aligned_names = parsed["aligned_entities"]
    if not isinstance(aligned_names, list) or any(not isinstance(name, str) for name in aligned_names):
        raise ValueError("aligned_entities must be a list of candidate names")
    unknown = set(aligned_names) - set(candidate_names)
    if unknown:
        raise ValueError(f"Response contains names outside the candidate list: {sorted(unknown)!r}")
    return aligned_names


def call_llm(main_name, candidate_names):
    """调用大模型进行实体对齐判断"""
    client = OpenAI(
        api_key=DASHSCOPE_API_KEY,
        base_url=BASE_URL
    )
    
    system_prompt = (
        "You are an expert in citrus flavor compounds, specializing in compound nomenclature, citrus breeding, and flavor characteristics. "
        "Strictly determine whether the primary entity and each candidate represent the same research subject. "
        'Return one valid JSON object, using double quotes, such as {"aligned_entities": ["exact candidate name"]}. '
        "Copy selected names exactly from the candidate list, including case and punctuation. "
        'If none match, return {"aligned_entities": []}. '
        "Do not include Markdown fences or additional explanations."
    )
    
    user_prompt = (
        f"Main entity: '{main_name}'\n"
        f"Candidate entity list: {candidate_names}\n"
        "Please strictly return the result as required:"
    )
    
    for attempt in range(3):  # 重试机制
        try:
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,  # 降低随机性
                max_tokens=2048,
                response_format={"type": "json_object"},
                extra_body={"enable_thinking": False},
                timeout=30
            )
            choice = completion.choices[0]
            if choice.finish_reason != "stop":
                raise ValueError(f"Incomplete model response: finish_reason={choice.finish_reason!r}")
            response = choice.message.content
            try:
                parse_alignment_response(response, candidate_names)
            except ValueError as error:
                raise ValueError(f"{error}; response={response!r}") from error
            return response
        except Exception as e:
            logger.warning("LLM call failed for %r (attempt %s/3): %s", main_name, attempt + 1, e)
            time.sleep(2 ** attempt)  # 指数退避
    return None

def process_alignment():
    processed_ids = load_processed_ids(ALIGNMENT_RESULTS_PATH)
    results_count = 0
    no_match_count = 0
    failed_count = 0
    skipped_count = 0
    print(f"Candidates: {os.path.abspath(CANDIDATES_PATH)}")
    print(f"Results: {os.path.abspath(ALIGNMENT_RESULTS_PATH)}")
    
    with open(CANDIDATES_PATH, 'r', encoding='utf-8') as f_in, \
         open(ALIGNMENT_RESULTS_PATH, 'a', encoding='utf-8') as f_out:
        
        lines = f_in.readlines()
        with tqdm(lines, desc="LLM Alignment") as progress:
            for line in progress:
                try:
                    data = json.loads(line)
                    main_node = data["main_node"]
                    if main_node["id"] in processed_ids:
                        skipped_count += 1
                        continue

                    candidate_names = [c["name"] for c in data["candidates"]]
                    response = call_llm(main_node["name"], candidate_names)
                    try:
                        aligned_names = parse_alignment_response(response, candidate_names)
                    except ValueError as error:
                        failed_count += 1
                        logger.warning(
                            "Alignment failed: main_id=%r, main_name=%r, error=%s, response=%r",
                            main_node["id"], main_node["name"], error, response,
                        )
                        continue

                    aligned_ids = [
                        c["candidate_id"] for c in data["candidates"]
                        if c["name"] in aligned_names
                    ]
                    result = {
                        "main_id": main_node["id"],
                        "main_name": main_node["name"],
                        "aligned_entities": aligned_ids
                    }

                    # 保持输出文件仅含有效对齐关系，写入后立即同步。
                    if aligned_ids:
                        f_out.write(json.dumps(result, ensure_ascii=False) + '\n')
                        f_out.flush()
                        os.fsync(f_out.fileno())
                        processed_ids.add(main_node["id"])
                        results_count += 1
                    else:
                        no_match_count += 1
                    time.sleep(0.5)
                finally:
                    progress.set_postfix(
                        saved=results_count, no_match=no_match_count,
                        failed=failed_count, skipped=skipped_count, refresh=False,
                    )
    
    print(f"Alignment completed. {results_count} valid alignments saved to {ALIGNMENT_RESULTS_PATH}")
    print(f"No match: {no_match_count}; failed: {failed_count}; skipped: {skipped_count}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if not DASHSCOPE_API_KEY:
        raise ValueError("DASHSCOPE_API_KEY not set in environment variables")
    process_alignment()
