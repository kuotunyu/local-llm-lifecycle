"""第 3 節「量化幾乎不吃微調效果」的配對檢定與等價界。

為什麼需要這支：第 4 節（TMMLU+）有配對 bootstrap CI、McNemar、分層重抽，但**扛整個論點
的第 3 節只有點估計**加一句「在雜訊範圍內」——而那個雜訊從來沒有被量過。

這是個 null result（「量化沒有造成損耗」），而 null result 最容易被攻擊的方式就是
「你只是沒有檢定力去偵測它」。要讓它站得住，得給**等價界**：不是說差異等於 0，
而是說「真實損耗最多不超過 X，因為 CI 就到那裡」。

五組跑的是同一份 4,699 題考卷、同一組 qid，逐題 em/f1 都已落盤，所以是配對資料，
零 GPU 就能算。

分層方式：DRCD 這份考卷是「3,524 題可回答 + 1,175 題 unanswerable」的固定設計，
所以 bootstrap 分層在 answerable 上重抽，維持兩層的比例。

用法：
  python3 56_drcd_paired_stats.py --eval-dir results/eval_raw
"""

from __future__ import annotations

import argparse
import json
from math import comb
from pathlib import Path

import numpy as np

BOOTSTRAP_N = 10_000
BOOTSTRAP_SEED = 20260731

GROUPS = ("base_zeroshot", "base_fewshot", "ft_unquantized", "ft_q8", "ft_q4")
LABEL = {
    "base_zeroshot": "組1 base zero-shot",
    "base_fewshot": "組2 base few-shot",
    "ft_unquantized": "組3 FT bf16",
    "ft_q8": "組4 FT Q8_0",
    "ft_q4": "組5 FT Q4_K_M",
}

# (較新/被測, 基準, 說明)
CONTRASTS = [
    ("ft_q4", "ft_unquantized", "Q4_K_M vs 未量化（量化總損耗）"),
    ("ft_q8", "ft_unquantized", "Q8_0 vs 未量化（GGUF 轉檔損耗）"),
    ("ft_q4", "ft_q8", "Q4_K_M vs Q8_0（Q8→Q4 量化損耗）"),
    ("ft_unquantized", "base_zeroshot", "微調增益（上限）"),
    ("ft_q4", "base_zeroshot", "微調增益（Q4 部署後）"),
]


def load_group(path: Path) -> dict[str, dict]:
    rows = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r["qid"] in rows:
            raise ValueError(f"{path.name} 有重複 qid：{r['qid']}")
        rows[r["qid"]] = r
    return rows


def mcnemar_exact(a: np.ndarray, b: np.ndarray) -> dict:
    """a、b 是 0/1 的逐題正確與否（EM==1 視為對）。"""
    b_only = int(np.sum((a == 1) & (b == 0)))
    c_only = int(np.sum((a == 0) & (b == 1)))
    n = b_only + c_only
    if n == 0:
        p = 1.0
    else:
        k = min(b_only, c_only)
        tail = sum(comb(n, i) for i in range(k + 1)) / (2 ** n)
        p = min(1.0, 2 * tail)
    return {"a_right_b_wrong": b_only, "a_wrong_b_right": c_only,
            "n_discordant": n, "p_value_two_sided": p}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-dir", default="results/eval_raw")
    ap.add_argument("--n-bootstrap", type=int, default=BOOTSTRAP_N)
    ap.add_argument("--seed", type=int, default=BOOTSTRAP_SEED)
    ap.add_argument("--out", default="results/drcd_paired_stats.json")
    args = ap.parse_args()

    ev = Path(args.eval_dir)
    data = {g: load_group(ev / f"{g}.jsonl") for g in GROUPS}

    qid_sets = {g: set(d) for g, d in data.items()}
    common = set.intersection(*qid_sets.values())
    for g, s in qid_sets.items():
        if s != common:
            raise ValueError(f"{g} 的 qid 集合與其他組不同（獨有 {len(s - common)} 題），無法配對")
    qids = sorted(common)
    n = len(qids)

    em = {g: np.array([float(data[g][q]["em"]) for q in qids]) for g in GROUPS}
    f1 = {g: np.array([float(data[g][q]["f1"]) for q in qids]) for g in GROUPS}
    answerable = np.array([bool(data[GROUPS[0]][q]["answerable"]) for q in qids])

    # 分層：可回答 / unanswerable 兩層固定比例（考卷本來就是這樣組的）
    strata = [np.flatnonzero(answerable), np.flatnonzero(~answerable)]
    rng = np.random.default_rng(args.seed)
    picks = np.concatenate(
        [idx[rng.integers(0, len(idx), size=(args.n_bootstrap, len(idx)))] for idx in strata],
        axis=1,
    )

    print(f"===== DRCD 五組配對檢定（n = {n:,}，"
          f"可回答 {int(answerable.sum()):,} / unanswerable {int((~answerable).sum()):,}）=====")
    print(f"配對分層 bootstrap {args.n_bootstrap:,} 次，seed={args.seed}\n")

    boot = {g: {"em": em[g][picks].mean(axis=1), "f1": f1[g][picks].mean(axis=1)} for g in GROUPS}

    print(f"{'組別':<22}{'EM':>9}{'F1':>9}")
    for g in GROUPS:
        print(f"{LABEL[g]:<22}{em[g].mean():>9.4f}{f1[g].mean():>9.4f}")

    out = {"n_items": n, "n_bootstrap": args.n_bootstrap, "seed": args.seed,
           "point": {g: {"em": float(em[g].mean()), "f1": float(f1[g].mean())} for g in GROUPS},
           "contrasts": {}}

    print(f"\n{'對照':<34}{'ΔEM (pp)':>10}{'95% CI (pp)':>20}{'ΔF1 (pp)':>10}{'95% CI (pp)':>20}")
    for new, base, desc in CONTRASTS:
        rec = {}
        cells = []
        for metric, arr in (("em", em), ("f1", f1)):
            d = boot[new][metric] - boot[base][metric]
            lo, hi = np.percentile(d, [2.5, 97.5])
            point = arr[new].mean() - arr[base].mean()
            rec[metric] = {"delta": float(point), "ci95_low": float(lo), "ci95_high": float(hi),
                           "se": float(d.std(ddof=1))}
            cells.append((point * 100, f"[{lo*100:+.2f}, {hi*100:+.2f}]"))
        out["contrasts"][f"{new}__vs__{base}"] = {"desc": desc, **rec}
        print(f"{desc:<34}{cells[0][0]:>+9.3f}{cells[0][1]:>20}"
              f"{cells[1][0]:>+9.3f}{cells[1][1]:>20}")

    # ---- 等價界：量化最多吃掉微調增益的幾 % ----
    gain_em = out["contrasts"]["ft_unquantized__vs__base_zeroshot"]["em"]["delta"]
    gain_f1 = out["contrasts"]["ft_unquantized__vs__base_zeroshot"]["f1"]["delta"]
    q = out["contrasts"]["ft_q4__vs__ft_unquantized"]
    # CI 下界（最負）＝ 資料仍相容的最壞情況損耗
    worst_em = -min(q["em"]["ci95_low"], 0.0)
    worst_f1 = -min(q["f1"]["ci95_low"], 0.0)
    out["equivalence"] = {
        "finetuning_gain_em": gain_em,
        "finetuning_gain_f1": gain_f1,
        "worst_case_quantization_loss_em": worst_em,
        "worst_case_quantization_loss_f1": worst_f1,
        "worst_case_pct_of_gain_em": worst_em / gain_em * 100 if gain_em else float("nan"),
        "worst_case_pct_of_gain_f1": worst_f1 / gain_f1 * 100 if gain_f1 else float("nan"),
    }
    print(f"\n等價界（null result 要講的話）：")
    print(f"  微調增益 EM {gain_em*100:+.2f} pp、F1 {gain_f1*100:+.2f} pp")
    print(f"  Q4 相對未量化的 95% CI 下界 = {q['em']['ci95_low']*100:+.3f} pp（EM）、"
          f"{q['f1']['ci95_low']*100:+.3f} pp（F1）")
    print(f"  → **量化最多吃掉微調增益的 {out['equivalence']['worst_case_pct_of_gain_em']:.2f}%"
          f"（EM）／{out['equivalence']['worst_case_pct_of_gain_f1']:.2f}%（F1）**"
          f"，這是 95% CI 允許的最壞情況，不是點估計。")

    # ---- 量化不是 no-op：逐題其實有變 ----
    print(f"\n量化不是 no-op（逐題比對，抵銷後才看起來像沒差）：")
    out["item_level_changes"] = {}
    for new, base in (("ft_q8", "ft_unquantized"), ("ft_q4", "ft_unquantized"), ("ft_q4", "ft_q8")):
        txt_diff = sum(1 for q_ in qids
                       if data[new][q_]["pred_answer"] != data[base][q_]["pred_answer"])
        em_diff = int(np.sum(em[new] != em[base]))
        a = (em[new] == 1.0).astype(int)
        b = (em[base] == 1.0).astype(int)
        mc = mcnemar_exact(a, b)
        out["item_level_changes"][f"{new}__vs__{base}"] = {
            "answer_text_differs": txt_diff, "em_differs": em_diff, "mcnemar": mc}
        print(f"  {LABEL[new]} vs {LABEL[base]}：答案文字不同 {txt_diff:,} 題"
              f"（{txt_diff/n*100:.2f}%），EM 改變 {em_diff:,} 題"
              f"；變好 {mc['a_wrong_b_right']} / 變壞 {mc['a_right_b_wrong']}"
              f"，McNemar p = {mc['p_value_two_sided']:.3f}")
    print("  → 量化確實改變了個別答案，只是變好與變壞的題數相當、互相抵銷，"
          "所以整體指標看起來沒動。")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已寫入 {args.out}")


if __name__ == "__main__":
    main()
