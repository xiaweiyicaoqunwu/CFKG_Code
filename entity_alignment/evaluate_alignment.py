"""在人工标注的实体对测试集上计算二分类指标，仅使用 Python 标准库。

默认输入为本脚本目录下的 data/alignment_results.jsonl 和
data/evalution/test_1500.jsonl，可用 --results / --test 指定其他文件。

label=1 表示同一实体，label=0 表示不同实体。默认按有序实体对匹配：
candidate_id 出现在 main_id 的 aligned_entities 中时预测为 1，否则为 0。
--undirected 允许反向记录也命中，但不推断传递关系。
只评估测试集中的实体对，测试集之外的预测不计入 FP；重复实体对去重。

accuracy  = (TP + TN) / (TP + TN + FP + FN)
precision = TP / (TP + FP)
recall    = TP / (TP + FN)
F1        = 2 * TP / (2 * TP + FP + FN)
precision、recall、F1 均针对正类 label=1；分母为 0 时返回 0。
"""

import argparse
import json
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent


def read_jsonl(path):
    """逐行读取 JSON 对象，兼容 UTF-8 BOM 和空行。"""
    with Path(path).open("r", encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            location = f"{path}:{line_number}"
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{location}: invalid JSON: {error.msg}") from error
            if not isinstance(record, dict):
                raise ValueError(f"{location}: expected a JSON object")
            yield location, record


def entity_id(record, field, location):
    """统一整数和字符串 ID 的表示；不使用实体名称匹配。"""
    value = record.get(field)
    if type(value) not in (int, str) or not str(value).strip():
        raise ValueError(f"{location}: {field} must be an integer or nonempty string")
    return str(value).strip()


def pair_key(main_id, candidate_id, undirected):
    return tuple(sorted((main_id, candidate_id))) if undirected else (main_id, candidate_id)


def load_predictions(path, undirected=False):
    """展开 aligned_entities，并合并重复结果行中的实体对。"""
    pairs = set()
    for location, record in read_jsonl(path):
        main_id = entity_id(record, "main_id", location)
        aligned = record.get("aligned_entities")
        if not isinstance(aligned, list):
            raise ValueError(f"{location}: aligned_entities must be a list")
        for value in aligned:
            candidate_id = entity_id({"candidate_id": value}, "candidate_id", location)
            pairs.add(pair_key(main_id, candidate_id, undirected))
    return pairs


def evaluate(results_path, test_path, undirected=False):
    """在全部标注实体对上计算指标，包括没有对齐结果的实体对。"""
    predictions = load_predictions(results_path, undirected)
    gold_labels = {}
    counts = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}

    for location, record in read_jsonl(test_path):
        main_id = entity_id(record, "main_id", location)
        candidate_id = entity_id(record, "candidate_id", location)
        label = record.get("label")
        if type(label) is not int or label not in (0, 1):
            raise ValueError(f"{location}: label must be the integer 0 or 1")
        pair = pair_key(main_id, candidate_id, undirected)
        if pair in gold_labels:
            if gold_labels[pair] != label:
                raise ValueError(f"{location}: conflicting labels for pair {pair}")
            continue
        gold_labels[pair] = label

        predicted = pair in predictions
        if label == 1:
            counts["tp" if predicted else "fn"] += 1
        else:
            counts["fp" if predicted else "tn"] += 1

    total = sum(counts.values())
    if total == 0:
        raise ValueError(f"{test_path}: test set is empty")

    tp, fp, tn, fn = (counts[key] for key in ("tp", "fp", "tn", "fn"))
    return {
        "mode": "undirected" if undirected else "directed",
        "num_samples": total,
        "positive_samples": tp + fn,
        "negative_samples": tn + fp,
        **counts,
        "accuracy": (tp + tn) / total,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate entity alignment on labeled entity pairs (positive label: 1)."
    )
    parser.add_argument(
        "--results", type=Path, default=BASE_DIR / "data" / "alignment_results.jsonl",
        help="Prediction JSONL containing main_id and aligned_entities.",
    )
    parser.add_argument(
        "--test", type=Path, default=BASE_DIR / "data" / "evalution" / "eval_2000.jsonl",
        help="Gold JSONL containing main_id, candidate_id and label (0 or 1).",
    )
    parser.add_argument(
        "--undirected", action="store_true",
        help="Treat (A, B) and (B, A) as the same alignment; default is directed.",
    )
    parser.add_argument("--output", type=Path, help="Optionally save metrics as JSON.")
    args = parser.parse_args()

    try:
        metrics = evaluate(args.results, args.test, args.undirected)
        if args.output:
            args.output.write_text(
                json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
    except (OSError, ValueError) as error:
        parser.error(str(error))

    print(f"Mode: {metrics['mode']}")
    print(
        f"Samples: {metrics['num_samples']} "
        f"(positive={metrics['positive_samples']}, negative={metrics['negative_samples']})"
    )
    print("  ".join(f"{key.upper()}={metrics[key]}" for key in ("tp", "fp", "tn", "fn")))
    for key in ("accuracy", "precision", "recall", "f1"):
        name = "F1" if key == "f1" else key
        print(f"{name:9s}: {metrics[key]:.6f} ({metrics[key]:.2%})")
    if args.output:
        print(f"Metrics saved to: {args.output.resolve()}")


if __name__ == "__main__":
    main()
