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
  python3 53_batch_sensitivity.py --dir-a results/eval_raw --dir-b results/eval_batch_b
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
            rows[r["qid"]] = r
    return rows


def macro_micro(rows: dict[str, dict]) -> tuple[float, float]:
    by_subject: dict[str, list[bool]] = {}
    for r in rows.values():
        by_subject.setdefault(r["subject"], []).append(bool(r["correct"]))
    per_subject = [sum(v) / len(v) for v in by_subject.values()]
    macro = sum(per_subject) / len(per_subject)
    micro = sum(bool(r["correct"]) for r in rows.values()) / len(rows)
    return macro, micro


def compare_group(a_path: Path, b_path: Path, group: str) -> dict:
    a, b = load(a_path), load(b_path)
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
    parser.add_argument("--dir-a", required=True, help="第一組 batch 設定的結果目錄")
    parser.add_argument("--dir-b", required=True, help="第二組 batch 設定的結果目錄")
    parser.add_argument("--label-a", default="A")
    parser.add_argument("--label-b", default="B")
    parser.add_argument("--out", default=None, help="輸出 JSON 路徑（預設不寫檔）")
    args = parser.parse_args()

    a_dir, b_dir = Path(args.dir_a), Path(args.dir_b)
    result = {"dir_a": str(a_dir), "dir_b": str(b_dir),
              "label_a": args.label_a, "label_b": args.label_b, "groups": {}}

    for group in ("tmmlu_base", "tmmlu_ft"):
        result["groups"][group] = compare_group(
            a_dir / f"{group}.jsonl", b_dir / f"{group}.jsonl", group
        )

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
    print(f"\nΔ macro（FT − base）：{delta_a*100:+.2f} pp -> {delta_b*100:+.2f} pp"
          f"　儀器誤差 = {abs(delta_b-delta_a)*100:.2f} pp")
    print("（這是抽樣誤差之外的另一份不確定度，報告的 CI 不涵蓋它）")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n已寫入 {args.out}")


if __name__ == "__main__":
    main()
