"""檢定 FT 的選項偏誤是「位置」造成的，還是「選項內容」造成的。

為什麼需要這支：`54_tmmlu_option_permutation.py` 量到選項順序隨機化投票能回收 45% 的退步，
但那個實驗**只隔離位置**。循環位移把選項內容整組搬到別的位置，所以任何綁在**內容本身**
的偏好都會跟著一起搬、量不到。最容易想到的內容型偏誤是**長度**——DRCD 抽取式微調把模型
往「短、精確的原文片段」推，如果 FT 學到的是「偏好比較短的選項」，那它看起來會像選項偏誤，
但性質完全不同（而且循環位移救不回來）。

在這支沒跑之前，EVAL_REPORT 只能說「剩下的退步**不是**位置偏誤」，不能說它是什麼。

做法：位移實驗的資料剛好可以把兩者拆開。每題的四個選項內容會輪流出現在四個位置，所以
  - 若偏誤綁在**位置**：依位置字母統計會偏斜，依長度名次統計會接近均勻
  - 若偏誤綁在**長度**：依長度名次統計會偏斜，依位置字母統計會接近均勻
兩張分佈同時看，就能判斷是哪一種——不需要 GPU，資料都已經在磁碟上。

用法：
  # 只用主實驗（固定順序）的資料
  python3 55_content_bias_check.py --eval-dir results/eval_raw

  # 加上位移實驗的資料（決定性的那一半）。兩種來源結果相同：
  #   --compact  已進 git 的緊湊逐題預測（1.3 MB，推薦）
  #   --perm-dir 原始的 22 MB 輸出目錄（未進 git）
  python3 55_content_bias_check.py --eval-dir results/eval_raw \
      --compact results/eval_perm/permutation_predictions.jsonl

位移後的選項長度是**推導**出來的，不需要那份 35 MB 的位移考卷：位移是決定性的，
位移 k 時位置 i 放的是原始第 (i-k) mod 4 個選項，所以有原始考卷就夠了。
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from math import erfc, exp, pi, sqrt
from pathlib import Path

LETTERS = ("A", "B", "C", "D")


def chi2_sf_df3(x: float) -> float:
    """卡方分佈 df=3 的上尾機率（閉式解，不引入 scipy）。"""
    if x <= 0:
        return 1.0
    return erfc(sqrt(x / 2.0)) + sqrt(2.0 * x / pi) * exp(-x / 2.0)


def chi2_uniform(counts: Counter, keys) -> tuple[float, float]:
    n = sum(counts[k] for k in keys)
    if n == 0:
        return 0.0, 1.0
    e = n / len(keys)
    x = sum((counts[k] - e) ** 2 / e for k in keys)
    return x, chi2_sf_df3(x)


def load_jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def length_rank(opts: list[str], chosen_idx: int) -> int:
    """被選中的選項在四個選項裡的長度名次（1 = 最短，4 = 最長）。"""
    order = sorted(range(len(opts)), key=lambda i: (len(opts[i]), i))
    return order.index(chosen_idx) + 1


def profile_from_compact(sample: dict, compact_path: Path, key: str) -> dict:
    """用「原始考卷 + 緊湊預測」重建位移實驗的統計，不需要 35 MB 的位移後考卷。

    位移是決定性的：位移 k 時，位置 i 放的是原始第 (i-k) mod 4 個選項。
    所以只要有原始選項內容與 k，就能算出模型選中的那個選項有多長，
    不必把 4 倍大的置換考卷實體化出來（也就不必進 git）。
    """
    by_pos: Counter = Counter()
    by_len: Counter = Counter()
    chosen_lens: list[int] = []

    for line in compact_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        row = sample.get(r["qid"])
        if row is None:
            continue
        orig = [str(row[x]) for x in LETTERS]
        for k, ch in enumerate(r[key]):
            if ch == "?":
                continue
            pos = LETTERS.index(ch)
            # 位移 k 下，位置 pos 放的是原始第 (pos-k)%4 個選項
            opts_at_positions = [orig[(i - k) % len(LETTERS)] for i in range(len(LETTERS))]
            by_pos[ch] += 1
            by_len[length_rank(opts_at_positions, pos)] += 1
            chosen_lens.append(len(opts_at_positions[pos]))

    return {
        "n": len(chosen_lens),
        "by_position": {k: by_pos[k] for k in LETTERS},
        "by_length_rank": {k: by_len[k] for k in (1, 2, 3, 4)},
        "mean_chosen_length": sum(chosen_lens) / len(chosen_lens) if chosen_lens else 0.0,
    }


def profile(sample: dict, preds: list[dict]) -> dict:
    by_pos: Counter = Counter()
    by_len: Counter = Counter()
    chosen_lens: list[int] = []
    for r in preds:
        p = r["pred_answer"]
        if p is None:
            continue
        row = sample.get(r["qid"])
        if row is None:
            continue
        opts = [str(row[x]) for x in LETTERS]
        idx = LETTERS.index(p)
        by_pos[p] += 1
        by_len[length_rank(opts, idx)] += 1
        chosen_lens.append(len(opts[idx]))
    return {
        "n": len(chosen_lens),
        "by_position": {k: by_pos[k] for k in LETTERS},
        "by_length_rank": {k: by_len[k] for k in (1, 2, 3, 4)},
        "mean_chosen_length": sum(chosen_lens) / len(chosen_lens) if chosen_lens else 0.0,
    }


def spread(counts: dict) -> float:
    """最大與最小佔比的差（百分點）。當成偏斜程度的直觀指標。"""
    n = sum(counts.values())
    if not n:
        return 0.0
    fr = [v / n * 100 for v in counts.values()]
    return max(fr) - min(fr)


def report(label: str, prof: dict) -> dict:
    n = prof["n"]
    pos_chi, pos_p = chi2_uniform(Counter(prof["by_position"]), LETTERS)
    len_chi, len_p = chi2_uniform(Counter(prof["by_length_rank"]), (1, 2, 3, 4))
    pos_s, len_s = spread(prof["by_position"]), spread(prof["by_length_rank"])
    print(f"  {label}（n={n:,}）")
    print("    依位置字母 : " + "  ".join(
        f"{k} {prof['by_position'][k]/n*100:.1f}%" for k in LETTERS)
        + f"   全距 {pos_s:.1f} pp   chi2(3)={pos_chi:,.0f}  p={pos_p:.2e}")
    print("    依長度名次 : " + "  ".join(
        f"{k} {prof['by_length_rank'][k]/n*100:.1f}%" for k in (1, 2, 3, 4))
        + f"   全距 {len_s:.1f} pp   chi2(3)={len_chi:,.0f}  p={len_p:.2e}")
    print(f"    被選中選項的平均長度 {prof['mean_chosen_length']:.3f} 字元")
    return {**prof, "position_chi2": pos_chi, "position_p": pos_p,
            "length_chi2": len_chi, "length_p": len_p,
            "position_spread_pp": pos_s, "length_spread_pp": len_s}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-dir", default="results/eval_raw")
    ap.add_argument("--perm-dir", default=None,
                    help="位移實驗的原始結果目錄（22 MB，未進 git）")
    ap.add_argument("--compact", default=None,
                    help="位移實驗的緊湊逐題預測（已進 git，1.3 MB）"
                         "；與 --perm-dir 擇一，結果相同")
    ap.add_argument("--out", default="results/tmmlu_content_bias.json")
    args = ap.parse_args()

    out: dict = {}
    ev = Path(args.eval_dir).expanduser()
    sample_path = ev / "tmmlu_sample.json"
    if not sample_path.exists():
        raise FileNotFoundError(
            f"缺少 {sample_path}。考卷沒有進 git，用 "
            f"`51_eval_tmmlu.py --full --groups none --out-dir {ev}` 重建（決定性）。")
    sample = {r["qid"]: r for r in json.loads(sample_path.read_text(encoding="utf-8"))}

    print("=" * 74)
    print("A. 主實驗（固定選項順序）")
    print("=" * 74)
    print("  位置與長度在這裡是綁在一起的，無法拆開；只當背景參考。")
    out["fixed_order"] = {}
    for g in ("tmmlu_base", "tmmlu_ft"):
        out["fixed_order"][g] = report(g, profile(sample, load_jsonl(ev / f"{g}.jsonl")))

    if args.compact or args.perm_dir:
        print()
        print("=" * 74)
        print("B. 位移實驗（決定性）：位置與長度已被拆開")
        print("=" * 74)
        out["permuted"] = {}
        if args.compact:
            cp = Path(args.compact).expanduser()
            print(f"  資料來源：{cp}（緊湊格式；位移後的選項長度由原始考卷推導）")
            for g, key in (("tmmlu_base", "base"), ("tmmlu_ft", "ft")):
                out["permuted"][g] = report(g, profile_from_compact(sample, cp, key))
        else:
            pd = Path(args.perm_dir).expanduser()
            psample = {r["qid"]: r for r in json.loads(
                (pd / "tmmlu_sample.json").read_text(encoding="utf-8"))}
            for g in ("tmmlu_base", "tmmlu_ft"):
                out["permuted"][g] = report(g, profile(psample, load_jsonl(pd / f"{g}.jsonl")))

        ft = out["permuted"]["tmmlu_ft"]
        ratio = ft["position_spread_pp"] / ft["length_spread_pp"] if ft["length_spread_pp"] else float("inf")
        out["ft_position_over_length_spread_ratio"] = ratio
        print()
        print(f"  FT 的位置偏斜 / 長度偏斜 = {ft['position_spread_pp']:.1f} pp / "
              f"{ft['length_spread_pp']:.1f} pp = {ratio:.1f} 倍")
        verdict = ("偏誤主要綁在**位置**，不是選項長度"
                   if ratio > 2 else
                   "位置與長度的偏斜量級相近，無法乾淨歸因，需要更細的內容特徵")
        out["verdict"] = verdict
        print(f"  判定：{verdict}")
        print("\n  注意：這只排除了『長度』這一個內容維度。其他內容特徵"
              "（語意合理度、是否含數字、與題幹的詞彙重疊等）沒有被檢定。")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已寫入 {args.out}")


if __name__ == "__main__":
    main()
