"""通用 XLSX → 实体 JSON／关系 JSON。只需修改下方配置后运行本文件。

依赖：python -m pip install openpyxl
输出：json_output/<SOURCE>/entities/<实体类型>.json 和 relations/<关系类型>.json。
输入应为一行表头、一行一条记录的矩形表，已完成必要的单位规范化。
复杂宽表先转成长表；本脚本不执行爬取、单位推断或 LLM 实体对齐。
"""
import glob
import hashlib
import json
import math
import re
from datetime import date, datetime, time
from decimal import Decimal, InvalidOperation
from pathlib import Path

from openpyxl import load_workbook


# ====================== 修改这里即可适配数据源 ======================
BASE_DIR = Path(__file__).resolve().parent
INPUT_FILES = ["database.xlsx"]  # 支持通配符，例如 foodb/*.xlsx
SOURCE = "Database"  # 数据来源名称，写入 JSON 的 source 属性
OUTPUT_DIR = BASE_DIR / "json_output" / SOURCE
SHEET = 0                                  # 工作表序号（从 0 开始）或名称
HEADER_ROW = 1                             # 表头所在行（从 1 开始）

# properties 中左侧是 JSON 属性名，右侧是 XLSX 列名。
# name 可映射到多个列，如 ["Primary descriptor", "Descriptor 2", "Descriptor 3"]。
# "@filename" 表示不含扩展名的文件名，适合一个文件对应一种柑橘。
# {"value": "固定值"} 表示常量。未配置的原表列不会导出。
# 可选列写成 {"column": "InChIKey", "optional": True}；列不存在时略过。
# 化合物主标识符为完整 InChIKey，其次为 PubChem CID（pubchem_id）。
ENTITY_RULES = {
    "compound": {
        "properties": {"name": "Name",
                       "inchikey": {"column": "InChIKey", "optional": True},
                       "pubchem_id": "PubChem CID", "smiles": "canonical SMILES"},
    },
    "taste": {
        "properties": {"name": "Taste"},
        "split": r"[;,，；]",               # 只拆分本类实体的名称
        "lower": True,                     # 多词描述 green tea 不按空格拆开
    },
}
RELATION_RULES = [
    {"type": "activate", "head": "compound", "tail": "taste", "properties": {}},
]
# 关系 properties 同样填写“输出属性名: 列名”，例如 {"threshold": "threshold"}。
# citrus→compound 用 hasCompound；flavor→odor/taste/mouthfeel 用 composedOf。
# 别对化合物名称按逗号拆分。两端均多值时会连接所有组合，配对数据应先展开成行。
# ===================================================================

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


def clean(value):
    if value is None or isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (date, datetime, time)):
        return value.isoformat()
    if isinstance(value, str):
        value = value.strip()
        if value.casefold() in {"", "none", "null", "nan", "n/a"}:
            return None
        if value.startswith("="):
            raise ValueError("选中的字段含公式，请先计算并另存为值")
    return value


def values_from(mapping, row, file, sheet):
    if isinstance(mapping, list):
        return [v for column in mapping for v in values_from(column, row, file, sheet)]
    if isinstance(mapping, dict):
        if "value" in mapping:
            value = mapping["value"]
        else:
            column = mapping["column"]
            if column not in row and not mapping.get("optional", False):
                raise ValueError(f"缺少列 {column!r}（{file.name}/{sheet}）")
            value = row.get(column)
    elif mapping == "@filename":
        value = file.stem
    elif mapping == "@sheet":
        value = sheet
    else:
        if mapping not in row:
            raise ValueError(f"缺少列 {mapping!r}（{file.name}/{sheet}）")
        value = row[mapping]
    value = clean(value)
    return [] if value is None else [value]


def properties_from(mapping, row, file, sheet):
    result = {}
    for key, column in mapping.items():
        values = list(dict.fromkeys(values_from(column, row, file, sheet)))
        if values:
            result[key] = values[0] if len(values) == 1 else values
    return result


def normalize_identifier(value):
    """保留编号末尾的有效 0，例如 1200.0 → '1200'。"""
    try:
        number = Decimal(str(value))
        if number.is_finite() and number > 0 and number == number.to_integral_value():
            return format(number, "f").split(".")[0]
    except InvalidOperation:
        pass
    raise ValueError(f"无效的正整数标识符: {value!r}")


def normalize_inchikey(value):
    """统一为大写并检查完整的 14-10-1 字母格式，不截断立体化学信息。

    格式依据：https://www.inchi-trust.org/technical-faq/#13.1
    此处仅作格式检查，不根据名称或其他标识符推断结构。
    """
    text = str(value).strip().upper()
    if not re.fullmatch(r"[A-Z]{14}-[A-Z]{10}-[A-Z]", text):
        raise ValueError(f"无效的完整 InChIKey 格式: {value!r}")
    return text


def make_id(*parts):
    return hashlib.sha256(json.dumps(parts, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def entity_id(label, props):
    # 优先完整 InChIKey，其次 CID；两者都缺失才使用名称。
    # 不将仅有 CID 的记录隐含地对齐到使用 InChIKey 的记录。
    key = next((k for k in ("inchikey", "pubchem_id") if label == "compound" and props.get(k)), "name")
    value = props[key] if key != "name" else " ".join(str(props[key]).split()).casefold()
    return label + ":" + make_id(key, value)


def convert(files, output, source, entity_rules=None, relation_rules=None, sheet=0, header_row=1):
    entity_rules = ENTITY_RULES if entity_rules is None else entity_rules
    relation_rules = RELATION_RULES if relation_rules is None else relation_rules
    files, output = list(files), Path(output)
    if not files or not source.strip() or header_row < 1:
        raise ValueError("请设置输入文件、来源名称及有效的表头行号")
    for label, rule in entity_rules.items():
        if label not in LABELS or "name" not in rule["properties"]:
            raise ValueError(f"实体类型或 name 映射不合法: {label}")
        if {"id", "source", "aliases"} & rule["properties"].keys():
            raise ValueError("id、source、aliases 是保留属性名")
    for rule in relation_rules:
        if (rule["head"], rule["tail"]) not in ONTOLOGY.get(rule["type"], set()):
            raise ValueError(f"不符合论文定义的关系: {rule}")
        if rule["head"] not in entity_rules or rule["tail"] not in entity_rules:
            raise ValueError("关系两端必须在 ENTITY_RULES 中定义")
        if (LABELS | {"id", "source", "head_id", "tail_id", "head_label", "tail_label"}) & rule.get("properties", {}).keys():
            raise ValueError("关系属性与端点信息重名")
    entities = {label: {} for label in entity_rules}
    relations = {r["type"]: {} for r in relation_rules}
    conflicts = invalid_ids = 0
    for file in map(Path, files):
        handle = file.open("rb")
        workbook, rownum = None, header_row
        try:
            workbook = load_workbook(handle, read_only=True, data_only=False)
            ws = workbook.worksheets[sheet] if isinstance(sheet, int) else workbook[sheet]
            rows = ws.iter_rows(min_row=header_row, values_only=True)
            headers = [str(v).strip() if v is not None else "" for v in next(rows, ())]
            if not any(headers) or len([h for h in headers if h]) != len({h for h in headers if h}):
                raise ValueError(f"空表头或重复列名: {file.name}/{ws.title}")
            for rownum, cells in enumerate(rows, header_row + 1):
                row = {h: v for h, v in zip(headers, cells) if h}
                if all(v is None or v == "" for v in row.values()):
                    continue
                endpoints = {}
                for label, rule in entity_rules.items():
                    props = properties_from(rule["properties"], row, file, ws.title)
                    names = props.pop("name", [])
                    names = names if isinstance(names, list) else [names]
                    if rule.get("split"):
                        names = [part for name in names for part in re.split(rule["split"], str(name))]
                    names = [str(n).strip().casefold() if rule.get("lower") else str(n).strip() for n in names]
                    if label == "compound":
                        for key, normalizer in [("inchikey", normalize_inchikey), ("pubchem_id", normalize_identifier)]:
                            if key in props:
                                try:
                                    props[key] = normalizer(props[key])
                                except ValueError:
                                    props[key + "_raw"] = str(props.pop(key))
                                    invalid_ids += 1
                    endpoints[label] = []
                    for name in dict.fromkeys(n for n in names if clean(n) is not None):
                        item = {"name": name, **props}
                        uid = entity_id(label, item)
                        stored = entities[label].setdefault(uid, {"id": uid, "source": [source]})
                        for key, value in item.items():
                            if key not in stored:
                                stored[key] = value
                            elif stored[key] != value:
                                if key == "name":
                                    aliases = stored.setdefault("aliases", [])
                                    if value not in aliases:
                                        aliases.append(value)
                                else:
                                    conflicts += 1  # 同 ID 的冲突属性保留首个非空值
                        endpoints[label].append(stored)
                for rule in relation_rules:
                    props = properties_from(rule.get("properties", {}), row, file, ws.title)
                    for head in endpoints[rule["head"]]:
                        for tail in endpoints[rule["tail"]]:
                            uid = make_id(rule["type"], head["id"], tail["id"], props)
                            relations[rule["type"]][uid] = {
                                "id": uid, rule["head"]: head["name"], rule["tail"]: tail["name"],
                                "head_id": head["id"], "tail_id": tail["id"],
                                "head_label": rule["head"], "tail_label": rule["tail"],
                                **props, "source": [source],
                            }
        except (ValueError, KeyError) as exc:
            raise ValueError(f"{file.name} 第 {rownum} 行附近: {exc}") from exc
        finally:
            if workbook is not None:
                workbook.close()
            handle.close()
    for folder, groups in [("entities", entities), ("relations", relations)]:
        target = output / folder
        stale = {p.stem for p in target.glob("*.json")} - groups.keys()
        if stale:
            raise ValueError(f"{target} 含其他映射的文件 {stale}，请换一个输出目录")
    for folder, groups in [("entities", entities), ("relations", relations)]:
        (output / folder).mkdir(parents=True, exist_ok=True)
        for label, records in groups.items():
            path = output / folder / f"{label}.json"
            path.write_text(json.dumps(list(records.values()), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    result = {"entities": sum(map(len, entities.values())), "relations": sum(map(len, relations.values())),
              "invalid_identifiers_kept_as_raw": invalid_ids, "property_conflicts_kept_first": conflicts}
    print(json.dumps(result, ensure_ascii=False))
    return result


if __name__ == "__main__":
    paths = []
    for pattern in INPUT_FILES:
        matches = [Path(p) for p in sorted(glob.glob(str(BASE_DIR / pattern)))
                   if Path(p).suffix.lower() == ".xlsx" and not Path(p).name.startswith("~$")]
        if not matches:
            raise SystemExit(f"未找到 XLSX: {pattern}")
        paths.extend(matches)
    convert(dict.fromkeys(paths), OUTPUT_DIR, SOURCE, sheet=SHEET, header_row=HEADER_ROW)
    print(f"JSON 已写入：{OUTPUT_DIR}")
