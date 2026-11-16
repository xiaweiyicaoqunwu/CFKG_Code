"""按测试集的 entity_type 分别评估实体对齐，仅依赖 Python 标准库。

运行示例：
    python evaluate_alignment_by_type.py
    python evaluate_alignment_by_type.py --test data/evalution/test_1500.jsonl
    python evaluate_alignment_by_type.py --output metrics_by_type.json
    python evaluate_alignment_by_type.py --directed

复用同目录 evaluate_alignment.py 的读取和实体对匹配函数，不修改原脚本。
默认允许反向命中（无向匹配），不推断传递关系；--directed 切换为有向匹配。
保留 --undirected 参数，可显式指定无向匹配。
仅评估人工标注实体对：未命中预测的样本视为负例，测试集外预测不计入 FP。
precision、recall、F1 针对正类 label=1，分母为零时返回 0。
同一实体对的重复标注去重，标签或类别冲突时报错。
overall 基于全部样本的混淆矩阵计算，不对类别指标求平均。
"""

import argparse
import json
from pathlib import Path

from evaluate_alignment import BASE_DIR, entity_id, load_predictions, pair_key, read_jsonl


COUNT_KEYS = ("tp", "fp", "tn", "fn")


def calculate_metrics(counts):
    """由一个类别（或总体）的混淆矩阵计算指标。"""
    tp, fp, tn, fn = (counts[key] for key in COUNT_KEYS)
    total = tp + fp + tn + fn
    return {
        "num_samples": total,
        "positive_samples": tp + fn,
        "negative_samples": tn + fp,
        **counts,
        "accuracy": (tp + tn) / total if total else 0.0,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
    }


def evaluate_by_type(results_path, test_path, undirected=True):
    """返回各类别指标和总体指标，分类依据为测试集 entity_type。"""
    predictions = load_predictions(results_path, undirected)
    seen = {}
    counts_by_type = {}

    for location, record in read_jsonl(test_path):
        main_id = entity_id(record, "main_id", location)
        candidate_id = entity_id(record, "candidate_id", location)
        label = record.get("label")
        if type(label) is not int or label not in (0, 1):
            raise ValueError(f"{location}: label must be the integer 0 or 1")
        entity_type = record.get("entity_type")
        if not isinstance(entity_type, str) or not entity_type.strip():
            raise ValueError(f"{location}: entity_type must be a nonempty string")
        entity_type = entity_type.strip()
        pair = pair_key(main_id, candidate_id, undirected)
        if pair in seen:
            if seen[pair] != (label, entity_type):
                raise ValueError(f"{location}: conflicting label or entity_type for pair {pair}")
            continue
        seen[pair] = (label, entity_type)

        counts = counts_by_type.setdefault(entity_type, dict.fromkeys(COUNT_KEYS, 0))
        predicted = pair in predictions
        if label == 1:
            counts["tp" if predicted else "fn"] += 1
        else:
            counts["fp" if predicted else "tn"] += 1

    if not seen:
        raise ValueError(f"{test_path}: test set is empty")

    overall_counts = {
        key: sum(counts[key] for counts in counts_by_type.values()) for key in COUNT_KEYS
    }
    return {
        "mode": "undirected" if undirected else "directed",
        "overall": calculate_metrics(overall_counts),
        "by_entity_type": {
            kind: calculate_metrics(counts_by_type[kind]) for kind in sorted(counts_by_type)
        },
    }


def print_report(report):
    """打印每类样本数、混淆矩阵及四项指标（范围 0 到 1）。"""
    grouped = report["by_entity_type"]
    type_width = max(len("entity_type"), len("overall"), *(len(kind) for kind in grouped))
    print(f"Mode: {report['mode']}")
    print(
        f"{'entity_type':<{type_width}} {'samples':>7} {'positive':>8} {'negative':>8} "
        f"{'TP':>6} {'FP':>6} {'TN':>6} {'FN':>6} "
        f"{'accuracy':>10} {'precision':>10} {'recall':>10} {'F1':>10}"
    )
    for kind, metrics in [*grouped.items(), ("overall", report["overall"])]:
        counts = " ".join(f"{metrics[key]:6d}" for key in COUNT_KEYS)
        scores = " ".join(
            f"{metrics[key]:10.6f}" for key in ("accuracy", "precision", "recall", "f1")
        )
        print(
            f"{kind:<{type_width}} {metrics['num_samples']:7d} "
            f"{metrics['positive_samples']:8d} {metrics['negative_samples']:8d} "
            f"{counts} {scores}"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate entity alignment separately for each entity_type."
    )
    parser.add_argument(
        "--results", type=Path, default=BASE_DIR / "data" / "alignment_results.jsonl",
        help="Prediction JSONL containing main_id and aligned_entities.",
    )
    parser.add_argument(
        "--test", type=Path,
        default=BASE_DIR / "data" / "evalution" / "entity_alignment_eval_2000.jsonl",
        help="Gold JSONL containing main_id, candidate_id, label and entity_type.",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--undirected", action="store_true",
        help="Treat (A, B) and (B, A) as the same alignment (default).",
    )
    mode.add_argument(
        "--directed", dest="undirected", action="store_false",
        help="Match only the exact (main_id, candidate_id) direction.",
    )
    parser.set_defaults(undirected=True)
    parser.add_argument("--output", type=Path, help="Optionally save grouped metrics as JSON.")
    args = parser.parse_args()

    try:
        report = evaluate_by_type(args.results, args.test, args.undirected)
        if args.output:
            args.output.write_text(
                json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
    except (OSError, ValueError) as error:
        parser.error(str(error))

    print_report(report)
    if args.output:
        print(f"Metrics saved to: {args.output.resolve()}")


if __name__ == "__main__":
    main()
