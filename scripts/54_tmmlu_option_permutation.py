"""量化「TMMLU+ 的退步有多少是可校正的選項偏誤」。

背景：`52_tmmlu_paired_stats.py` 在全量 20,118 題上量到 FT 相對 base 的退步高度集中在
特定 gold 字母上——FT 選 B 的次數少 1,946 次、選 D 多 1,889 次，於是 gold=B 掉 16.9 pp
而 **gold=D 反而進步 7.7 pp**。這是作答分佈位移的特徵，不是知識消失。

但那只是**診斷**。「相當一部分退步可以靠 decoding 端校正」在被實測之前只是推論。
這支腳本把它變成數字。

做法：**選項內容循環位移 + 位置層級聚合**
  對每一題產生 4 個版本，位移 k=0..3。位移 k 時，原本第 j 個選項移到第 (j+k) mod 4 個
  位置，gold 也跟著移到 (gold+k) mod 4。k=0 是恆等式，所以會原封不動復現既有結果，
  可以當一致性檢查。
  模型在位移 k 的版本上答了位置 p，換算回原始選項編號就是 (p-k) mod 4。

  這樣同一個「正確內容」會輪流出現在 A、B、C、D 四個位置。如果模型只是偏好某個字母，
  四個版本平均之後這份偏好會相互抵銷；如果是真的不知道答案，平均後仍然答錯。

三個指標：
  1. **單一順序 accuracy**（k=0）：現況，會受字母偏好污染
  2. **位移平均 accuracy**：四個位移的平均
  3. **多數決 accuracy**：四票投回同一個原始選項，取眾數

**主要指標是多數決，不是位移平均。** 這一點違反直覺，但可以直接證明：

  位移平均**量不到可校正的選項偏誤**。考慮一個「知道答案、但從不說 B」的模型。
  循環位移讓每一題的 gold 剛好造訪 A/B/C/D 各一次，所以**每一題**都恰好在其中
  一個位移上答錯 → 位移平均 = 3/4，固定不變。而單一順序 = 1 − P(gold=B)，
  在 gold 邊際均勻時同樣是 3/4。**兩者相等，回收量恆為 0。**
  實測（`54_test_option_permutation.py`，gold 完全平衡的 4,800 題子集）：
    單一順序 0.750000、位移平均 0.750000、回收 **+0.0000 pp**
    同一個偏誤，多數決回收 **+25.00 pp**
  把 gold 改成偏斜（80% 是 B）之後，位移平均的回收量才跳到 +55 pp。

  所以位移平均量的是「**對 gold 邊際不均的穩健度**」，不是選項偏誤。
  TMMLU+ test split 的 gold 邊際本來就均勻（5054/5029/5012/5023），
  位移平均**注定**回報接近 0 的回收量——那是代數恆等式，不是實驗結果。
  若把它讀成「攤平位置效應後退步還在，所以是真的知識損失」，那個結論是被公式
  逼出來的。位移平均在本實驗的正確定位是**設計健全性檢查**：確認退步不是 gold
  分佈造成的假象。預期回收量 0.00 ± 0.02 pp，明顯偏離才代表「選項內容 × 位置」
  有交互作用（那會是另一個發現）。

  多數決才會動：上面那個模型四票裡三票投對，眾數就是對的 → 完全回收。

關於多數決的兩個但書：
  - 它每題用 4 次推論，單一順序只用 1 次，所以增益裡混有「集成」成分。但本專案
    全程 `do_sample=False`，**同一個順序重跑是決定性的**，變異完全來自位移本身；
    而且純集成效應在這個評分規則下是**負的**（比較弱的模型票更散、更常平手、
    被罰更重），所以集成不會製造假的正回收——回收量若為正，方向是可信的。
  - 它是**評分協定**的性質，不是模型本身的性質。使用者實際拿到的仍是單一順序的行為。

用法（三階段，中間那段跑 GPU）：
  # 1. 產生置換後的考卷（會寫成 51_eval_tmmlu.py 的 sample 快取，讓它直接沿用）
  python3 54_tmmlu_option_permutation.py --build --src results/eval_raw/tmmlu_sample.json \
      --out-dir ~/tmmlu_perm/eval_raw

  # 2. 用既有的、已驗證過的推論路徑跑（不重寫推論邏輯）
  python3 51_eval_tmmlu.py --full --groups all --out-dir ~/tmmlu_perm/eval_raw \
      --base-model unsloth/Qwen3-8B --merged-dir <merged>

  # 3. 聚合
  python3 54_tmmlu_option_permutation.py --analyze --out-dir ~/tmmlu_perm/eval_raw \
      --stats-path results/tmmlu_option_permutation.json

**--out-dir 不要指到 results/ 底下。** `51_eval_tmmlu.py` 會把彙總寫到
`out_dir.parent / "tmmlu_summary.json"`，若 out-dir 是 `results/eval_perm`，
就會用一份「對 80,440 列彙總、看起來很合理但定義完全不同」的檔案覆蓋掉
`results/tmmlu_summary.json`（第 4 節的正式結果）。用 WSL 原生路徑跑，
只把最後的統計 JSON 用 --stats-path 寫回 repo。
（順帶：results 在 /mnt/c 上，逐 batch flush 走 9p 會慢很多，原生路徑也比較快。）
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

LETTERS = ("A", "B", "C", "D")
N_PERM = len(LETTERS)
PERM_SEP = "#p"


# ---------------------------------------------------------------------------
# 1. 產生置換後的考卷
# ---------------------------------------------------------------------------

def permute_row(row: dict, k: int) -> dict:
    """位移 k：原本第 j 個選項移到第 (j+k) mod 4 個位置。"""
    orig_opts = [row[L] for L in LETTERS]
    gold_idx = LETTERS.index(row["gold_answer"])

    new_opts = [None] * N_PERM
    for j, content in enumerate(orig_opts):
        new_opts[(j + k) % N_PERM] = content
    assert all(o is not None for o in new_opts)

    new_gold_idx = (gold_idx + k) % N_PERM
    # 內容守恆：置換只能換位置，不能改動或遺失任何選項
    assert sorted(map(str, new_opts)) == sorted(map(str, orig_opts)), "置換改動了選項內容"
    # gold 指向的內容必須不變
    assert new_opts[new_gold_idx] == orig_opts[gold_idx], "gold 對應的選項內容跑掉了"

    out = {
        "qid": f"{row['qid']}{PERM_SEP}{k}",
        "subject": row["subject"],
        "question": row["question"],
        "gold_answer": LETTERS[new_gold_idx],
    }
    for i, L in enumerate(LETTERS):
        out[L] = new_opts[i]
    return out


def build(src: Path, out_dir: Path) -> None:
    rows = json.loads(src.read_text(encoding="utf-8"))
    print(f"原始考卷 {len(rows)} 題 <- {src}")

    permuted, mapping = [], {}
    skipped = []
    for row in rows:
        opts = [row[L] for L in LETTERS]
        # 選項內容重複的題目不能做位移聚合：模型答「哪個位置」無法唯一對回原始選項，
        # 會把票投錯地方。這種題直接排除並記錄，不要靜靜地混進去。
        if len(set(map(str, opts))) != N_PERM:
            skipped.append(row["qid"])
            continue
        for k in range(N_PERM):
            permuted.append(permute_row(row, k))
        mapping[row["qid"]] = {
            "subject": row["subject"],
            "gold_answer": row["gold_answer"],
            "gold_index": LETTERS.index(row["gold_answer"]),
        }

    out_dir.mkdir(parents=True, exist_ok=True)
    # 直接寫成 51_eval_tmmlu.py 的 sample 快取檔名，讓它 load_or_build_sample() 沿用
    (out_dir / "tmmlu_sample.json").write_text(
        json.dumps(permuted, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out_dir / "perm_map.json").write_text(
        json.dumps({"n_original": len(rows), "n_skipped": len(skipped),
                    "skipped_qids": skipped, "items": mapping},
                   ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"排除選項內容重複的題目 {len(skipped)} 題"
          + (f"（例：{skipped[:3]}）" if skipped else ""))
    print(f"產生 {len(permuted)} 列（{len(mapping)} 題 x {N_PERM} 個位移）-> {out_dir}")
    print(f"下一步：51_eval_tmmlu.py --full --groups all --out-dir {out_dir}")


# ---------------------------------------------------------------------------
# 2. 聚合
# ---------------------------------------------------------------------------

def load_predictions(path: Path) -> dict[tuple[str, int], str | None]:
    """回傳 {(原始 qid, k): 模型答的位置字母}。

    重複 key 直接報錯：51_eval_tmmlu.py 的輸出是 append 模式（斷點續跑用），
    若同一個 (qid, k) 被寫了兩次，dict 會靜靜地留下最後一筆。跟 52 一樣要擋。
    """
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        qid, _, k = r["qid"].rpartition(PERM_SEP)
        key = (qid, int(k))
        if key in out:
            raise ValueError(f"{path.name} 有重複的 (qid, 位移)：{key}（斷點續跑重複寫入？）")
        out[key] = r["pred_answer"]
    return out


def score_group(preds: dict, mapping: dict) -> dict:
    """把每題的 4 個位移結果聚合成三個指標。"""
    per_subject_single: dict[str, list[int]] = {}
    per_subject_avg: dict[str, list[float]] = {}
    per_subject_vote: dict[str, list[float]] = {}
    per_subject_vote_strict: dict[str, list[int]] = {}
    per_item: dict[str, dict] = {}   # 給 bootstrap 用的逐題分數
    # 位移後模型仍偏好哪些位置字母（若偏好還在，代表是位置偏誤而非內容偏誤）
    letter_counts: Counter = Counter()
    n_missing = 0
    n_tie = 0

    for qid, meta in mapping.items():
        gold_idx = meta["gold_index"]
        subject = meta["subject"]

        votes = []          # 每個位移換算回來的「原始選項編號」
        per_perm_correct = []
        for k in range(N_PERM):
            pred_letter = preds.get((qid, k), "__MISSING__")
            if pred_letter == "__MISSING__":
                n_missing += 1
                per_perm_correct.append(0)
                continue
            if pred_letter is None:          # 沒解析出 A-D
                per_perm_correct.append(0)
                continue
            letter_counts[pred_letter] += 1
            pred_pos = LETTERS.index(pred_letter)
            orig_choice = (pred_pos - k) % N_PERM   # 換算回原始選項編號
            votes.append(orig_choice)
            per_perm_correct.append(1 if orig_choice == gold_idx else 0)

        # 1. 單一順序（k=0，恆等式）
        per_subject_single.setdefault(subject, []).append(per_perm_correct[0])
        # 2. 位移平均
        per_subject_avg.setdefault(subject, []).append(sum(per_perm_correct) / N_PERM)
        # 3. 多數決。平手怎麼算很關鍵：4 票投 4 個選項，模型一沒把握就容易平手
        #    （實測 base 有 12.1% 的題目平手）。把平手一律算錯會**系統性偏袒偏誤較小的
        #    模型**——偏誤越大的模型票越分散、平手越多、被罰越重，那就不是在量校正效果，
        #    是在量平手率。所以主指標用「隨機打破平手的期望值」（gold 在並列者之中就得
        #    1/並列數），這是無偏的；嚴格版（平手算錯）留著當保守下界。
        if votes:
            tally = Counter(votes).most_common()
            top = tally[0][1]
            winners = [c for c, n in tally if n == top]
            if len(winners) > 1:
                n_tie += 1
            vote_expected = (1.0 / len(winners)) if gold_idx in winners else 0.0
            vote_strict = 1 if (len(winners) == 1 and winners[0] == gold_idx) else 0
        else:
            vote_expected, vote_strict = 0.0, 0
        per_subject_vote.setdefault(subject, []).append(vote_expected)
        per_subject_vote_strict.setdefault(subject, []).append(vote_strict)

        per_item[qid] = {
            "subject": subject,
            "single": per_perm_correct[0],
            "avg": sum(per_perm_correct) / N_PERM,
            "vote": vote_expected,
            "vote_strict": vote_strict,
        }

    def macro(d):
        return sum(sum(v) / len(v) for v in d.values()) / len(d)

    def micro(d):
        tot = sum(len(v) for v in d.values())
        return sum(sum(v) for v in d.values()) / tot

    return {
        "n_items": len(mapping),
        "n_missing_predictions": n_missing,
        "n_ties_in_vote": n_tie,
        "single_order": {"macro": macro(per_subject_single), "micro": micro(per_subject_single)},
        "perm_averaged": {"macro": macro(per_subject_avg), "micro": micro(per_subject_avg)},
        "majority_vote": {"macro": macro(per_subject_vote), "micro": micro(per_subject_vote)},
        "majority_vote_strict": {"macro": macro(per_subject_vote_strict),
                                 "micro": micro(per_subject_vote_strict)},
        "position_letter_counts": dict(sorted(letter_counts.items())),
        "tie_rate": n_tie / len(mapping) if mapping else 0.0,
        "_per_subject": {
            "single": {k: sum(v) / len(v) for k, v in per_subject_single.items()},
            "avg": {k: sum(v) / len(v) for k, v in per_subject_avg.items()},
            "vote": {k: sum(v) / len(v) for k, v in per_subject_vote.items()},
        },
        "_per_item": per_item,
    }


# ---------------------------------------------------------------------------
# 配對分層 bootstrap（跟 52_tmmlu_paired_stats.py 同一套設計）
# ---------------------------------------------------------------------------

BOOTSTRAP_N = 10_000
BOOTSTRAP_SEED = 20260731


def bootstrap_deltas(base_items: dict, ft_items: dict,
                     n_boot: int = BOOTSTRAP_N, seed: int = BOOTSTRAP_SEED) -> dict:
    """三個指標的 Δ macro 各給一個 95% CI。

    跟 52 一樣：分層在科目內重抽（抽樣設計是每科固定題數），base 與 FT 用**同一組**
    重抽 index 以維持配對。66 個科目是 TMMLU+ 全體、不是抽樣，所以不對科目層重抽。
    """
    import numpy as np

    qids = sorted(base_items)
    subjects = sorted({base_items[q]["subject"] for q in qids})
    by_subject = {s: [q for q in qids if base_items[q]["subject"] == s] for s in subjects}

    metrics = ("single", "avg", "vote", "vote_strict")
    arrays = {}
    for m_ in metrics:
        arrays[m_] = {
            "b": {s: np.array([base_items[q][m_] for q in by_subject[s]], dtype=np.float64)
                  for s in subjects},
            "f": {s: np.array([ft_items[q][m_] for q in by_subject[s]], dtype=np.float64)
                  for s in subjects},
        }

    rng = np.random.default_rng(seed)
    n_subj = len(subjects)
    acc = {m_: {"b": np.empty((n_boot, n_subj)), "f": np.empty((n_boot, n_subj))}
           for m_ in metrics}

    for si, s in enumerate(subjects):
        n_s = len(by_subject[s])
        idx = rng.integers(0, n_s, size=(n_boot, n_s))   # 同一組 index 餵給所有指標與兩組
        for m_ in metrics:
            acc[m_]["b"][:, si] = arrays[m_]["b"][s][idx].mean(axis=1)
            acc[m_]["f"][:, si] = arrays[m_]["f"][s][idx].mean(axis=1)

    out = {"n_bootstrap": n_boot, "seed": seed}
    dist = {}
    for m_ in metrics:
        d = acc[m_]["f"].mean(axis=1) - acc[m_]["b"].mean(axis=1)
        dist[m_] = d
        lo, hi = np.percentile(d, [2.5, 97.5])
        out[m_] = {"delta_macro": float(d.mean()), "ci95_low": float(lo),
                   "ci95_high": float(hi), "se": float(d.std(ddof=1))}

    # 「校正回收了多少」本身也要有 CI——它是兩個相關統計量的差，
    # 必須在每一輪 bootstrap 內相減，不能拿兩個獨立 CI 目測。
    #
    # 正負號：兩個 Δ 都是負的（FT 比 base 差）。「回收」= 退步變小 = Δ 往 0 靠 =
    # Δ_metric **大於** Δ_single。所以是 metric − single，不是 single − metric。
    # 寫反的話：single=-3.33、vote=-0.30 會算出 -3.03（負值），但同一行的百分比
    # 因為分子分母同時變號反而是對的 91%，於是印出「回收 -3.03 pp（原退步的 91%）」
    # 這種自相矛盾的句子；更糟的是下面的守門檢查會在**校正真的有效時**才觸發警告。
    for m_ in ("avg", "vote", "vote_strict"):
        rec = dist[m_] - dist["single"]        # 正值 = 校正後退步變小
        lo, hi = np.percentile(rec, [2.5, 97.5])
        out[f"recovered_{m_}"] = {
            "pp": float(rec.mean()), "ci95_low": float(lo), "ci95_high": float(hi),
            "frac_bootstrap_le_zero": float((rec <= 0).mean()),
        }
    return out


def analyze(out_dir: Path, stats_path: Path | None, allow_incomplete: bool = False) -> None:
    mapping = json.loads((out_dir / "perm_map.json").read_text(encoding="utf-8"))["items"]
    result = {"n_original_items": len(mapping), "groups": {}}

    loaded = {}
    for group in ("tmmlu_base", "tmmlu_ft"):
        p = out_dir / f"{group}.jsonl"
        if not p.exists():
            raise FileNotFoundError(f"缺少 {p}，先跑 51_eval_tmmlu.py")
        loaded[group] = load_predictions(p)

    # 只保留「兩組都有完整 4 個位移」的題目。
    #
    # 缺漏的 (qid, k) 在 score_group 裡會被當成答錯。若某一組缺得比較多（實務上就是這樣：
    # WDDM/TDR 重置專打最長的 prompt，而長題目在兩組是同一批），那一組會被系統性壓低，
    # 直接污染 base 與 FT 的差。丟掉覆蓋不全的題目才能維持配對與公平；丟掉多少要講出來。
    complete = {
        qid for qid in mapping
        if all((qid, k) in loaded[g] for k in range(N_PERM) for g in loaded)
    }
    dropped = len(mapping) - len(complete)
    if dropped and not allow_incomplete:
        frac = dropped / len(mapping)
        if frac > 0.02:
            raise ValueError(
                f"有 {dropped} 題（{frac*100:.1f}%）在某一組缺少位移結果，超過 2% 的容許值。"
                f"推論尚未跑完就分析會得到有偏誤的比較；要強制分析請加 --allow-incomplete。"
            )
        print(f"[注意] {dropped} 題（{frac*100:.2f}%）因為某一組缺少位移結果而排除，"
              f"兩組同時排除以維持配對。實際分析 {len(complete)} 題。")
    mapping = {q: v for q, v in mapping.items() if q in complete}
    result["n_original_items"] = len(mapping)
    result["n_dropped_incomplete"] = dropped

    for group in ("tmmlu_base", "tmmlu_ft"):
        result["groups"][group] = score_group(loaded[group], mapping)

    b, f = result["groups"]["tmmlu_base"], result["groups"]["tmmlu_ft"]

    boot = bootstrap_deltas(b.pop("_per_item"), f.pop("_per_item"))
    result["bootstrap"] = boot

    print(f"\n===== 選項位移校正（n = {len(mapping)} 題 x {N_PERM} 個位移）=====")
    print(f"{'指標':<20}{'base':>9}{'FT':>9}{'Δ (pp)':>10}{'95% CI (pp)':>20}")
    deltas = {}
    for key, label, bkey in (("single_order", "單一順序", "single"),
                             ("perm_averaged", "位移平均 [主]", "avg"),
                             ("majority_vote", "多數決(期望)", "vote"),
                             ("majority_vote_strict", "多數決(平手算錯)", "vote_strict")):
        d = f[key]["macro"] - b[key]["macro"]
        deltas[key] = d
        ci = (f"[{boot[bkey]['ci95_low']*100:+.2f}, {boot[bkey]['ci95_high']*100:+.2f}]"
              if bkey in boot else "-")
        print(f"{label:<20}{b[key]['macro']:>9.4f}{f[key]['macro']:>9.4f}"
              f"{d*100:>+9.2f}{ci:>20}")
    print(f"{'多數決平手率':<20}{b['tie_rate']*100:>8.1f}%{f['tie_rate']*100:>8.1f}%")

    base_d = deltas["single_order"]
    print(f"\n退步的分解（以單一順序的 Δ = {base_d*100:+.2f} pp 為基準）：")
    for key, label, bkey in (("perm_averaged", "位移平均", "avg"),
                             ("majority_vote", "多數決(期望)", "vote"),
                             ("majority_vote_strict", "多數決(平手算錯)", "vote_strict")):
        recovered = deltas[key] - base_d          # 正值 = 退步變小
        pct = (recovered / abs(base_d) * 100) if base_d else float("nan")
        r = boot[f"recovered_{bkey}"]
        print(f"  {label}：Δ = {deltas[key]*100:+.2f} pp，回收 {recovered*100:+.2f} pp"
              f"（原退步的 {pct:.0f}%），回收量 95% CI "
              f"[{r['ci95_low']*100:+.2f}, {r['ci95_high']*100:+.2f}] pp")
        if r["frac_bootstrap_le_zero"] > 0.025:
            print(f"    [注意] 有 {r['frac_bootstrap_le_zero']*100:.1f}% 的 bootstrap 樣本回收量 <= 0，"
                  f"「校正有效」在統計上不夠穩固")

    # 位移平均的預期回收量在 gold 邊際均勻時就是 0（見 docstring）。把它印出來當設計檢查。
    gold_note = boot["recovered_avg"]
    if abs(gold_note["pp"]) > 0.005:
        print(f"\n  [設計檢查] 位移平均的回收量是 {gold_note['pp']*100:+.2f} pp。"
              f"gold 邊際接近均勻時這個值理論上該接近 0；明顯偏離代表"
              f"「選項內容 x 位置」有交互作用，值得單獨追。")

    print(f"\n位移後模型偏好的位置字母（若偏好仍在，代表是位置偏誤）：")
    for g, lbl in ((b, "base"), (f, "FT")):
        tot = sum(g["position_letter_counts"].values())
        dist = "  ".join(f"{k} {v/tot*100:.1f}%" for k, v in g["position_letter_counts"].items())
        print(f"  {lbl:<5} {dist}")

    print(f"\n多數決平手（保守算錯）：base {b['n_ties_in_vote']} 題 / FT {f['n_ties_in_vote']} 題")
    if b["n_missing_predictions"] or f["n_missing_predictions"]:
        print(f"缺漏預測：base {b['n_missing_predictions']} / FT {f['n_missing_predictions']}")

    result["delta_macro"] = {k: deltas[k] for k in deltas}
    result["recovered_pp"] = {k: (deltas[k] - base_d) for k in deltas}   # 正值 = 退步變小
    if stats_path:
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        stats_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n已寫入 {stats_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--analyze", action="store_true")
    parser.add_argument("--src", default="results/eval_raw/tmmlu_sample.json")
    # 預設值刻意放在 repo 外：51_eval_tmmlu.py 會把彙總寫到 out_dir.parent/"tmmlu_summary.json"，
    # 所以任何指向 results/ 底下的 out-dir 都會覆蓋掉 results/tmmlu_summary.json——
    # 也就是 README、EVAL_REPORT §4、PLAN 與兩張 HF model card 背後的那份正式結果。
    parser.add_argument("--out-dir", default="~/tmmlu_perm/eval_raw")
    parser.add_argument("--stats-path", default="results/tmmlu_option_permutation.json")
    parser.add_argument("--allow-incomplete", action="store_true",
                        help="容許推論尚未跑完就分析（結果不可信，只供中途觀察）")
    args = parser.parse_args()

    out_dir = Path(args.out_dir).expanduser()

    # 硬性守衛：out-dir 指到 results/ 底下會讓 51_eval_tmmlu.py 用一份「對 80,440 列彙總、
    # 看起來很合理但定義完全不同」的檔案覆蓋掉 results/tmmlu_summary.json。那是招牌結果檔，
    # 被覆蓋不會報錯、也不容易察覺，所以直接擋掉而不是只在 docstring 提醒。
    results_dir = (Path(__file__).resolve().parent.parent / "results").resolve()
    try:
        inside = out_dir.resolve().is_relative_to(results_dir)
    except (AttributeError, ValueError):      # Python < 3.9 沒有 is_relative_to
        inside = str(out_dir.resolve()).startswith(str(results_dir))
    if inside:
        raise SystemExit(
            f"--out-dir 不可指向 {results_dir} 底下（給的是 {out_dir}）。\n"
            f"51_eval_tmmlu.py 會寫 out_dir.parent/'tmmlu_summary.json'，"
            f"會覆蓋掉正式結果檔。請用 repo 外的路徑，例如 ~/tmmlu_perm/eval_raw，"
            f"再用 --stats-path 把統計寫回 results/。"
        )

    if args.build:
        build(Path(args.src), out_dir)
    elif args.analyze:
        analyze(out_dir, Path(args.stats_path) if args.stats_path else None,
                allow_incomplete=args.allow_incomplete)
    else:
        parser.error("要 --build 或 --analyze")


if __name__ == "__main__":
    main()
