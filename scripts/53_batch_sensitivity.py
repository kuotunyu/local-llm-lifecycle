"""量測「換一組 batch 組成，結果會變多少」——評估儀器本身的誤差。

為什麼需要這支：`51_eval_tmmlu.py` 全程 greedy decoding（`do_sample=False`），直覺上
應該完全決定性。實測確認**同一組 batch 組成**下逐位元可重現（同一份考卷跑兩次，輸出檔
SHA-256 相同）。但 batch 組成一改，結果就會動——bf16 矩陣乘法的規約順序隨 padding 與
batch 形狀改變，邊界題目的 logit 差距小到會被翻轉。

這不是 bug，是浮點運算的性質。重點是它會在**抽樣誤差之外**再貢獻一份不確定度：
n=200 的先導實測裡，光是把固定 batch=16 換成長度排序的動態 batch，就讓 Δ macro
從 −12.25pp 變成 −10.73pp（1.5pp），而翻轉的題目只有 base 2 題 / FT 4 題。

所以報告不能只講抽樣 CI。這支腳本把兩次不同 batch 設定的全量結果拿來逐題比對，
量出儀器誤差的量級，讓「Δ = −X.X pp」這句話能同時交代兩種誤差來源。

用法：
  # 用已進 git 的資料重算（clone 後即可跑，不需要 GPU）
  python3 53_batch_sensitivity.py \
      --compact-b results/eval_perm/batch6144_predictions.jsonl

  # 或指定兩個原始結果目錄
  python3 53_batch_sensitivity.py --dir-a results/eval_raw --dir-b ~/tmmlu_batch_b/eval_raw

對照組（`--max-batch-tokens 6144`）的逐題預測以緊湊格式入庫：一行一題
`{"qid":...,"base":"B","ft":"D"}`，約 0.9 MB 取代 5.6 MB 的原始輸出。
gold_answer 與 subject 跟 A 組完全相同（同一份考卷），correct 由兩者比對得到，
所以只需要記「每題兩個模型各答什麼」。A 組就是主實驗的結果，本來就在 git 裡。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(path: Path) -> dict[str, dict]:
    rows = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            if r["qid"] in rows:
                raise ValueError(f"{path.name} 有重複 qid：{r['qid']}（斷點續跑重複寫入？）")
            rows[r["qid"]] = r
    return rows


def load_compact_b(path: Path, ref: dict[str, dict[str, dict]]) -> dict[str, dict[str, dict]]:
    """讀對照組的緊湊逐題預測，補回 gold/subject/correct 後回傳跟 load() 同樣的結構。

    對照組只需要記「每題兩個模型各答什麼」——gold_answer 與 subject 跟 A 組完全相同
    （同一份考卷），correct 由兩者比對得到。所以緊湊格式一行一題：
        {"qid":"accounting-0","base":"B","ft":"D"}
    約 0.9 MB，取代 5.6 MB 的原始輸出。ref 是已進 git 的 A 組結果，用來取 gold 與 subject。
    """
    out: dict[str, dict[str, dict]] = {g: {} for g in ref}
    seen = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        qid = r["qid"]
        if qid in seen:
            raise ValueError(f"{path.name} 有重複 qid：{qid}")
        seen.add(qid)
        for group, key in (("tmmlu_base", "base"), ("tmmlu_ft", "ft")):
            a = ref[group].get(qid)
            if a is None:
                raise ValueError(f"{qid} 不在 A 組結果裡，兩組考卷不一致？")
            pred = r[key] or None
            if pred == "?":
                pred = None
            out[group][qid] = {
                "qid": qid,
                "subject": a["subject"],
                "gold_answer": a["gold_answer"],
                "pred_answer": pred,
                "correct": pred == a["gold_answer"],
            }
    return out


def macro_micro(rows: dict[str, dict]) -> tuple[float, float]:
    by_subject: dict[str, list[bool]] = {}
    for r in rows.values():
        by_subject.setdefault(r["subject"], []).append(bool(r["correct"]))
    per_subject = [sum(v) / len(v) for v in by_subject.values()]
    macro = sum(per_subject) / len(per_subject)
    micro = sum(bool(r["correct"]) for r in rows.values()) / len(rows)
    return macro, micro


def compare_group(a: dict, b: dict, group: str) -> dict:
    if set(a) != set(b):
        raise ValueError(f"{group}：兩邊 qid 集合不同，無法逐題比對"
                         f"（A 獨有 {len(set(a)-set(b))}、B 獨有 {len(set(b)-set(a))}）")
    n = len(a)
    pred_diff = [q for q in a if a[q]["pred_answer"] != b[q]["pred_answer"]]
    corr_diff = [q for q in a if bool(a[q]["correct"]) != bool(b[q]["correct"])]
    macro_a, micro_a = macro_micro(a)
    macro_b, micro_b = macro_micro(b)
    return {
        "n": n,
        "pred_changed": len(pred_diff),
        "pred_changed_frac": len(pred_diff) / n,
        "correct_changed": len(corr_diff),
        "correct_changed_frac": len(corr_diff) / n,
        "macro_a": macro_a, "macro_b": macro_b, "macro_shift": macro_b - macro_a,
        "micro_a": micro_a, "micro_b": micro_b, "micro_shift": micro_b - micro_a,
        "examples": [
            {"qid": q, "gold": a[q]["gold_answer"],
             "pred_a": a[q]["pred_answer"], "pred_b": b[q]["pred_answer"]}
            for q in pred_diff[:10]
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir-a", default="results/eval_raw",
                        help="第一組 batch 設定的結果目錄（預設用已進 git 的主實驗結果）")
    parser.add_argument("--dir-b", default=None, help="第二組 batch 設定的結果目錄")
    parser.add_argument("--compact-b", default=None,
                        help="第二組的緊湊逐題預測（已進 git："
                             "results/eval_perm/batch6144_predictions.jsonl）；與 --dir-b 擇一")
    parser.add_argument("--label-a", default="max-batch-tokens 12288")
    parser.add_argument("--label-b", default="max-batch-tokens 6144")
    parser.add_argument("--out", default=None, help="輸出 JSON 路徑（預設不寫檔）")
    args = parser.parse_args()

    if not args.dir_b and not args.compact_b:
        parser.error("要給 --dir-b 或 --compact-b")

    a_dir = Path(args.dir_a).expanduser()
    a = {g: load(a_dir / f"{g}.jsonl") for g in ("tmmlu_base", "tmmlu_ft")}

    if args.compact_b:
        b = load_compact_b(Path(args.compact_b).expanduser(), a)
        b_desc = str(args.compact_b)
    else:
        b_dir = Path(args.dir_b).expanduser()
        b = {g: load(b_dir / f"{g}.jsonl") for g in ("tmmlu_base", "tmmlu_ft")}
        b_desc = str(b_dir)

    result = {"dir_a": str(a_dir), "dir_b": b_desc,
              "label_a": args.label_a, "label_b": args.label_b, "groups": {}}

    for group in ("tmmlu_base", "tmmlu_ft"):
        result["groups"][group] = compare_group(a[group], b[group], group)

    print(f"\n===== batch 組成敏感度（{args.label_a} vs {args.label_b}）=====")
    for group, g in result["groups"].items():
        print(f"\n[{group}] n = {g['n']}")
        print(f"  預測字母改變 : {g['pred_changed']:>6}  ({g['pred_changed_frac']*100:.2f}%)")
        print(f"  對錯改變     : {g['correct_changed']:>6}  ({g['correct_changed_frac']*100:.2f}%)")
        print(f"  macro accuracy: {g['macro_a']:.4f} -> {g['macro_b']:.4f}"
              f"  ({g['macro_shift']*100:+.2f} pp)")
        print(f"  micro accuracy: {g['micro_a']:.4f} -> {g['micro_b']:.4f}"
              f"  ({g['micro_shift']*100:+.2f} pp)")

    gb, gf = result["groups"]["tmmlu_base"], result["groups"]["tmmlu_ft"]
    delta_a = gf["macro_a"] - gb["macro_a"]
    delta_b = gf["macro_b"] - gb["macro_b"]
    result["delta_macro_a"] = delta_a
    result["delta_macro_b"] = delta_b
    result["delta_macro_shift"] = delta_b - delta_a
    # ASCII 連字號，理由同 51_eval_tmmlu.py：U+2212 在 cp950 console 會讓腳本中斷。
    # 這支是 README 標為「clone 後可直接跑」的其中一條，Windows 使用者一跑就撞到。
    print(f"\nΔ macro（FT - base）：{delta_a*100:+.2f} pp -> {delta_b*100:+.2f} pp"
          f"　儀器誤差 = {abs(delta_b-delta_a)*100:.2f} pp")
    print("（這是抽樣誤差之外的另一份不確定度，報告的 CI 不涵蓋它）")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n已寫入 {args.out}")


if __name__ == "__main__":
    main()
