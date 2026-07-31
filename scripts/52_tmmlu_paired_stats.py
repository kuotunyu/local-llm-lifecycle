"""Phase 5 附帶：TMMLU+ base vs FT 的配對統計檢定。

`51_eval_tmmlu.py` 只給點估計（Δ macro accuracy）。點估計本身不足以支撐
「存在 catastrophic forgetting」這種因果宣稱——要先知道這個 Δ 的不確定度有多大。

base 與 FT 跑的是**完全同一份題目**（同一個 `tmmlu_sample.json`，qid 集合相同），
所以逐題結果是配對資料，可以用配對檢定，比兩組各自算獨立 CI 再目測有沒有重疊
精確得多（配對消掉了「題目難度」這個共同變異來源）。

本腳本輸出三種量：

1. **McNemar 精確檢定**（micro 層級）：只看兩邊答案不一致的題目。
   b = base 對 / FT 錯，c = base 錯 / FT 對。虛無假設 b、c 同分佈，
   在 b+c 次不一致中 b 服從 Binomial(b+c, 0.5)，用雙尾精確二項檢定算 p。

2. **配對分層 bootstrap 的 95% CI**（macro 與 micro 都算）：
   抽樣設計是「每科目抽固定題數」，所以 bootstrap 也要**分層在科目內重抽**，
   而不是把 20,118 題當成一坨 i.i.d.。每一輪 bootstrap 對每個科目抽出同一組
   index 給 base 與 FT（維持配對），重算 per-subject accuracy → macro → Δ。
   66 個科目是 TMMLU+ 的**全體**、不是抽樣得來的，所以不對科目層重抽——
   不確定性完全來自「抽到哪些題」。

3. **量化解析度診斷**：每科目 n 題時，per-subject accuracy 只能落在 {0, 1/n, ...,
   1} 這 n+1 個值上，Δ macro 的最小非零刻度是 1/(66n)。n 小的時候「某科目掉了
   0.67」根本不是效果量，是 3 題錯 2 題的量化假象。這一節就是用來把它講清楚。

4. **選項偏誤診斷**：accuracy 掉下來有兩種很不一樣的原因——(a) 真的不知道答案，
   (b) 模型對某個選項字母產生偏好，把機率質量從正確字母挪走。後者在指令微調後
   相當常見，而且是**可校正**的（校正 prior 就能救回大半），跟「知識被洗掉」
   在實務上是完全不同的結論。這裡比較 base 與 FT 的**預測字母邊際分佈**，以及
   **依 gold 字母分層的 accuracy**：若退步集中在某幾個 gold 字母上、且預測邊際
   明顯偏離均勻，就是選項偏誤而非單純的知識遺忘。

用法：
  python3 52_tmmlu_paired_stats.py --out-dir results/eval_raw
  python3 52_tmmlu_paired_stats.py --out-dir results/eval_repro_check --no-write
"""

from __future__ import annotations

import argparse
import json
from math import comb
from pathlib import Path

import numpy as np

BOOTSTRAP_N = 10_000
BOOTSTRAP_SEED = 20260730


# ---------------------------------------------------------------------------
# 載入配對資料
# ---------------------------------------------------------------------------

def load_group(path: Path) -> dict[str, dict]:
    rows = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r["qid"] in rows:
            raise ValueError(f"{path.name} 有重複 qid：{r['qid']}（斷點續跑重複寫入？）")
        rows[r["qid"]] = r
    return rows


class Paired:
    """base 與 FT 在同一份題目上的逐題結果，所有陣列同序對齊。"""

    def __init__(self, qids, subjects, gold, base_pred, ft_pred, base_correct, ft_correct):
        self.qids = qids
        self.subjects = subjects
        self.gold = gold
        self.base_pred = base_pred
        self.ft_pred = ft_pred
        self.b = base_correct
        self.f = ft_correct


def build_paired(base_path: Path, ft_path: Path) -> Paired:
    base = load_group(base_path)
    ft = load_group(ft_path)

    only_base = set(base) - set(ft)
    only_ft = set(ft) - set(base)
    if only_base or only_ft:
        raise ValueError(
            f"qid 集合不一致，無法配對：base 獨有 {len(only_base)} 題、FT 獨有 {len(only_ft)} 題。"
            f"（範例：{sorted(only_base)[:3]} / {sorted(only_ft)[:3]}）"
        )

    qids = sorted(base)  # 固定排序，讓 bootstrap 可重現
    for q in qids:
        if base[q]["subject"] != ft[q]["subject"]:
            raise ValueError(f"{q} 兩邊 subject 不一致")
        if base[q]["gold_answer"] != ft[q]["gold_answer"]:
            raise ValueError(f"{q} 兩邊 gold_answer 不一致")

    # pred_answer 可能是 None（模型沒吐出可解析的 A-D）；統一成 "?" 方便統計
    def preds(g):
        return np.array([(g[q]["pred_answer"] or "?") for q in qids])

    return Paired(
        qids=qids,
        subjects=np.array([base[q]["subject"] for q in qids]),
        gold=np.array([base[q]["gold_answer"] for q in qids]),
        base_pred=preds(base),
        ft_pred=preds(ft),
        base_correct=np.array([bool(base[q]["correct"]) for q in qids], dtype=np.int8),
        ft_correct=np.array([bool(ft[q]["correct"]) for q in qids], dtype=np.int8),
    )


# ---------------------------------------------------------------------------
# 指標
# ---------------------------------------------------------------------------

def subject_index(subjects: np.ndarray) -> list[tuple[str, np.ndarray]]:
    """回傳 [(subject, 該科目在陣列中的 index), ...]，依科目名排序（可重現）。"""
    return [(s, np.flatnonzero(subjects == s)) for s in sorted(set(subjects.tolist()))]


def macro_micro(correct: np.ndarray, groups: list[tuple[str, np.ndarray]]) -> tuple[float, float]:
    per_subject = np.array([correct[idx].mean() for _, idx in groups])
    return float(per_subject.mean()), float(correct.mean())


# ---------------------------------------------------------------------------
# 1. McNemar 精確檢定
# ---------------------------------------------------------------------------

def mcnemar_exact(b: np.ndarray, f: np.ndarray) -> dict:
    b_only = int(np.sum((b == 1) & (f == 0)))  # base 對、FT 錯（退步）
    c_only = int(np.sum((b == 0) & (f == 1)))  # base 錯、FT 對（進步）
    both_correct = int(np.sum((b == 1) & (f == 1)))
    both_wrong = int(np.sum((b == 0) & (f == 0)))
    n_discordant = b_only + c_only

    if n_discordant == 0:
        p = 1.0
    else:
        # 雙尾精確二項檢定：P(X <= min) + P(X >= max)，X ~ Bin(n_discordant, 0.5)
        k = min(b_only, c_only)
        tail = sum(comb(n_discordant, i) for i in range(0, k + 1)) / (2 ** n_discordant)
        p = min(1.0, 2 * tail)

    return {
        "both_correct": both_correct,
        "both_wrong": both_wrong,
        "base_correct_ft_wrong": b_only,
        "base_wrong_ft_correct": c_only,
        "n_discordant": n_discordant,
        "p_value_two_sided": p,
    }


# ---------------------------------------------------------------------------
# 2. 配對分層 bootstrap
# ---------------------------------------------------------------------------

def paired_stratified_bootstrap(
    b: np.ndarray, f: np.ndarray, groups: list[tuple[str, np.ndarray]],
    n_boot: int = BOOTSTRAP_N, seed: int = BOOTSTRAP_SEED,
) -> dict:
    rng = np.random.default_rng(seed)
    n_subj = len(groups)
    n_items = len(b)

    macro_b = np.empty((n_boot, n_subj), dtype=np.float64)
    macro_f = np.empty((n_boot, n_subj), dtype=np.float64)
    micro_hits_b = np.zeros(n_boot, dtype=np.float64)
    micro_hits_f = np.zeros(n_boot, dtype=np.float64)

    for si, (_, idx) in enumerate(groups):
        n_s = len(idx)
        # 同一組 resample index 同時餵給 base 與 FT —— 這就是「配對」的體現
        picks = idx[rng.integers(0, n_s, size=(n_boot, n_s))]
        bs = b[picks]
        fs = f[picks]
        macro_b[:, si] = bs.mean(axis=1)
        macro_f[:, si] = fs.mean(axis=1)
        micro_hits_b += bs.sum(axis=1)
        micro_hits_f += fs.sum(axis=1)

    delta_macro = macro_f.mean(axis=1) - macro_b.mean(axis=1)
    delta_micro = (micro_hits_f - micro_hits_b) / n_items

    def summarize(d: np.ndarray) -> dict:
        lo, hi = np.percentile(d, [2.5, 97.5])
        return {
            "ci95_low": float(lo),
            "ci95_high": float(hi),
            "bootstrap_mean": float(d.mean()),
            "bootstrap_se": float(d.std(ddof=1)),
            # 「Δ >= 0（沒有退步）」在 bootstrap 分佈中出現的比例，粗略當單尾 p 讀
            "frac_bootstrap_ge_zero": float((d >= 0).mean()),
        }

    # 逐科目的 Δ 也一併給 CI——重抽已經做完了，這裡只是換個維度取百分位，不用再算一次。
    # 有了 CI，「退步最明顯的科目」才是排名而不是量化雜訊。
    per_subject = {}
    delta_subj = macro_f - macro_b  # (n_boot, n_subj)
    lo_all, hi_all = np.percentile(delta_subj, [2.5, 97.5], axis=0)
    for si, (name, idx) in enumerate(groups):
        per_subject[name] = {
            "n": int(len(idx)),
            "base_accuracy": float(b[idx].mean()),
            "ft_accuracy": float(f[idx].mean()),
            "delta": float(f[idx].mean() - b[idx].mean()),
            "ci95_low": float(lo_all[si]),
            "ci95_high": float(hi_all[si]),
        }

    return {
        "n_bootstrap": n_boot,
        "seed": seed,
        "delta_macro_accuracy": summarize(delta_macro),
        "delta_micro_accuracy": summarize(delta_micro),
        "per_subject_delta": per_subject,
    }


# ---------------------------------------------------------------------------
# 3. 量化解析度診斷
# ---------------------------------------------------------------------------

def quantization_diagnostics(b: np.ndarray, f: np.ndarray, groups: list[tuple[str, np.ndarray]]) -> dict:
    sizes = [len(idx) for _, idx in groups]
    n_min, n_max = min(sizes), max(sizes)
    per_subject_delta = {
        s: float(f[idx].mean() - b[idx].mean()) for s, idx in groups
    }
    # 每科目 n 題 → per-subject accuracy 的刻度是 1/n；Δ macro 的刻度是 1/(66n)
    return {
        "n_subjects": len(groups),
        "items_per_subject_min": n_min,
        "items_per_subject_max": n_max,
        "per_subject_accuracy_resolution_max": 1.0 / n_min,
        "delta_macro_resolution_max": 1.0 / (len(groups) * n_min),
        # 單題翻轉造成的 per-subject Δ；n 很小的時候這個值大得離譜
        "one_item_flip_moves_subject_accuracy_by": 1.0 / n_min,
        "per_subject_delta": per_subject_delta,
    }


# ---------------------------------------------------------------------------
# 4. 選項偏誤診斷
# ---------------------------------------------------------------------------

LETTERS = ("A", "B", "C", "D")


def _chi2_sf_df3(x: float) -> float:
    """卡方分佈 df=3 的上尾機率，閉式解，避免為了一個 p 值把 scipy 拉進相依。
    df=3：SF(x) = erfc(sqrt(x/2)) + sqrt(2x/pi) * exp(-x/2)"""
    from math import erfc, exp, pi, sqrt

    return erfc(sqrt(x / 2.0)) + sqrt(2.0 * x / pi) * exp(-x / 2.0)


def option_bias(p: Paired) -> dict:
    n = len(p.qids)
    gold_marginal = {g: int(np.sum(p.gold == g)) for g in LETTERS}
    base_marginal = {g: int(np.sum(p.base_pred == g)) for g in LETTERS}
    ft_marginal = {g: int(np.sum(p.ft_pred == g)) for g in LETTERS}
    base_unparsed = int(np.sum(p.base_pred == "?"))
    ft_unparsed = int(np.sum(p.ft_pred == "?"))

    # 依 gold 字母分層的 accuracy：知識遺忘會讓四個字母大致等幅下降，
    # 選項偏誤則會集中打擊「FT 不愛預測的那個字母」對應的 gold
    by_gold = {}
    for g in LETTERS:
        idx = np.flatnonzero(p.gold == g)
        if len(idx) == 0:
            continue
        ab, af = float(p.b[idx].mean()), float(p.f[idx].mean())
        by_gold[g] = {
            "n": int(len(idx)),
            "base_accuracy": ab,
            "ft_accuracy": af,
            "delta": af - ab,
        }

    # base 預測邊際 vs FT 預測邊際的 2×4 卡方獨立性檢定（df=3）
    tot_b = sum(base_marginal.values())
    tot_f = sum(ft_marginal.values())
    chi2 = 0.0
    if tot_b and tot_f:
        for g in LETTERS:
            col = base_marginal[g] + ft_marginal[g]
            if col == 0:
                continue
            for cnt, tot in ((base_marginal[g], tot_b), (ft_marginal[g], tot_f)):
                exp_ = col * tot / (tot_b + tot_f)
                chi2 += (cnt - exp_) ** 2 / exp_

    # 預測邊際偏離「gold 邊際」的總變異距離（0 = 完全吻合，1 = 完全錯開）
    def tvd(marg: dict, tot: int) -> float:
        if tot == 0:
            return float("nan")
        return 0.5 * sum(abs(marg[g] / tot - gold_marginal[g] / n) for g in LETTERS)

    deltas = [v["delta"] for v in by_gold.values()]
    # 依 gold 字母平均的 accuracy（= 各字母 recall 的平均）。gold 分佈不均時，
    # 這個值不會被「剛好偏好多數字母」灌水，可以用來檢查退步是不是只是 gold 分佈的假象。
    bal_base = sum(v["base_accuracy"] for v in by_gold.values()) / len(by_gold)
    bal_ft = sum(v["ft_accuracy"] for v in by_gold.values()) / len(by_gold)
    return {
        "balanced_accuracy_base": bal_base,
        "balanced_accuracy_ft": bal_ft,
        "balanced_accuracy_delta": bal_ft - bal_base,
        "gold_marginal": gold_marginal,
        "base_pred_marginal": base_marginal,
        "ft_pred_marginal": ft_marginal,
        "base_unparsed": base_unparsed,
        "ft_unparsed": ft_unparsed,
        "accuracy_by_gold_letter": by_gold,
        # 四個 gold 字母的退步幅度全距：知識遺忘應該小，選項偏誤會很大
        "delta_spread_across_gold_letters": (max(deltas) - min(deltas)) if deltas else float("nan"),
        "base_pred_vs_gold_tvd": tvd(base_marginal, tot_b),
        "ft_pred_vs_gold_tvd": tvd(ft_marginal, tot_f),
        "chi2_base_vs_ft_pred_marginal": chi2,
        "chi2_df": 3,
        "chi2_p_value": _chi2_sf_df3(chi2) if chi2 > 0 else 1.0,
    }


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="results/eval_raw")
    parser.add_argument("--stats-path", default=None,
                        help="輸出 JSON 路徑，預設 <out-dir>/../tmmlu_paired_stats.json")
    parser.add_argument("--n-bootstrap", type=int, default=BOOTSTRAP_N)
    parser.add_argument("--seed", type=int, default=BOOTSTRAP_SEED)
    parser.add_argument("--no-write", action="store_true", help="只印不寫檔")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    p = build_paired(out_dir / "tmmlu_base.jsonl", out_dir / "tmmlu_ft.jsonl")
    b, f = p.b, p.f
    groups = subject_index(p.subjects)

    macro_base, micro_base = macro_micro(b, groups)
    macro_ft, micro_ft = macro_micro(f, groups)

    stats = {
        "source_dir": str(out_dir),
        "n_items": len(p.qids),
        "n_subjects": len(groups),
        "point_estimate": {
            "base_macro_accuracy": macro_base,
            "ft_macro_accuracy": macro_ft,
            "delta_macro_accuracy": macro_ft - macro_base,
            "base_micro_accuracy": micro_base,
            "ft_micro_accuracy": micro_ft,
            "delta_micro_accuracy": micro_ft - micro_base,
        },
        "mcnemar_exact": mcnemar_exact(b, f),
        "paired_stratified_bootstrap": paired_stratified_bootstrap(
            b, f, groups, n_boot=args.n_bootstrap, seed=args.seed
        ),
        "quantization": quantization_diagnostics(b, f, groups),
        "option_bias": option_bias(p),
    }

    print(f"\n===== TMMLU+ 配對統計（{out_dir}）=====")
    print(f"n = {stats['n_items']} 題，{stats['n_subjects']} 科目"
          f"（每科 {stats['quantization']['items_per_subject_min']}"
          f"-{stats['quantization']['items_per_subject_max']} 題）")

    pe = stats["point_estimate"]
    print(f"\nmacro accuracy：base {pe['base_macro_accuracy']:.4f} → FT {pe['ft_macro_accuracy']:.4f}"
          f"  Δ = {pe['delta_macro_accuracy']*100:+.2f} pp")
    print(f"micro accuracy：base {pe['base_micro_accuracy']:.4f} → FT {pe['ft_micro_accuracy']:.4f}"
          f"  Δ = {pe['delta_micro_accuracy']*100:+.2f} pp")

    bs = stats["paired_stratified_bootstrap"]
    for key, label in (("delta_macro_accuracy", "Δ macro"), ("delta_micro_accuracy", "Δ micro")):
        d = bs[key]
        print(f"\n{label}  95% CI = [{d['ci95_low']*100:+.2f}, {d['ci95_high']*100:+.2f}] pp"
              f"（配對分層 bootstrap，{bs['n_bootstrap']} 次，SE = {d['bootstrap_se']*100:.2f} pp）")
        crosses = d["ci95_low"] <= 0 <= d["ci95_high"]
        print(f"  CI {'包含' if crosses else '不含'} 0"
              f"　→ {'無法排除「沒有退步」' if crosses else '退步在統計上可辨識'}")

    mc = stats["mcnemar_exact"]
    print(f"\nMcNemar 精確檢定：base 對/FT 錯 = {mc['base_correct_ft_wrong']}，"
          f"base 錯/FT 對 = {mc['base_wrong_ft_correct']}，"
          f"不一致題數 = {mc['n_discordant']}，p = {mc['p_value_two_sided']:.4g}")

    q = stats["quantization"]
    print(f"\n量化解析度：每科目最少 {q['items_per_subject_min']} 題 → "
          f"單題翻轉就讓該科目 accuracy 跳 {q['one_item_flip_moves_subject_accuracy_by']*100:.1f} pp；"
          f"Δ macro 最小刻度 {q['delta_macro_resolution_max']*100:.3f} pp")

    ps = stats["paired_stratified_bootstrap"]["per_subject_delta"]
    ranked = sorted(ps.items(), key=lambda kv: kv[1]["delta"])
    n_sig_down = sum(1 for _, v in ps.items() if v["ci95_high"] < 0)
    n_sig_up = sum(1 for _, v in ps.items() if v["ci95_low"] > 0)
    print(f"\n逐科目：{n_sig_down} 科顯著退步（CI 完全在 0 以下）、{n_sig_up} 科顯著進步、"
          f"{len(ps) - n_sig_down - n_sig_up} 科無法區分")
    print(f"\n退步最明顯的 8 科：")
    print(f"{'科目':<50}{'n':>6}{'base':>8}{'FT':>8}{'Δ pp':>9}{'95% CI (pp)':>20}")
    for name, v in ranked[:8]:
        ci = f"[{v['ci95_low']*100:+.1f}, {v['ci95_high']*100:+.1f}]"
        print(f"{name:<50}{v['n']:>6}{v['base_accuracy']:>8.3f}{v['ft_accuracy']:>8.3f}"
              f"{v['delta']*100:>+8.1f}{ci:>20}")
    print(f"\n進步最明顯的 3 科：")
    for name, v in ranked[-3:][::-1]:
        ci = f"[{v['ci95_low']*100:+.1f}, {v['ci95_high']*100:+.1f}]"
        print(f"{name:<50}{v['n']:>6}{v['base_accuracy']:>8.3f}{v['ft_accuracy']:>8.3f}"
              f"{v['delta']*100:>+8.1f}{ci:>20}")

    ob = stats["option_bias"]
    print(f"\n選項偏誤：無法解析的輸出 base {ob['base_unparsed']} 題 / FT {ob['ft_unparsed']} 題")
    print(f"{'字母':<6}{'gold':>7}{'base 預測':>11}{'FT 預測':>10}"
          f"{'base acc':>10}{'FT acc':>9}{'Δ':>9}")
    for g in LETTERS:
        row = ob["accuracy_by_gold_letter"].get(g)
        if row is None:
            continue
        print(f"{g:<6}{ob['gold_marginal'][g]:>7}{ob['base_pred_marginal'][g]:>11}"
              f"{ob['ft_pred_marginal'][g]:>10}{row['base_accuracy']:>10.3f}"
              f"{row['ft_accuracy']:>9.3f}{row['delta']*100:>+8.1f}pp")
    print(f"預測邊際偏離 gold 的 TVD：base {ob['base_pred_vs_gold_tvd']:.4f} → FT {ob['ft_pred_vs_gold_tvd']:.4f}"
          f"（越大越偏）")
    print(f"base vs FT 預測邊際卡方（df=3）= {ob['chi2_base_vs_ft_pred_marginal']:.2f}，"
          f"p = {ob['chi2_p_value']:.4g}")
    print(f"四個 gold 字母的退步幅度全距 = {ob['delta_spread_across_gold_letters']*100:.1f} pp"
          f"（純知識遺忘應該偏小；偏大代表退步集中在特定字母，是選項偏誤）")

    if not args.no_write:
        stats_path = Path(args.stats_path) if args.stats_path else out_dir.parent / "tmmlu_paired_stats.json"
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n已寫入 {stats_path}")


if __name__ == "__main__":
    main()
