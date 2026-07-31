"""`54_tmmlu_option_permutation.py` 的驗證測試（不需要 GPU，用真實 TMMLU+ 題目跑）。

分兩part：

**A. 置換與換算的正確性。** 用行為已知的假模型當 oracle，斷言指標算出來的值等於
   手算的理論值。置換的 off-by-one 很難用肉眼看出來，但會靜靜地產出「看起來很合理」
   的錯誤數字——這正是本專案 §4.3 記錄過的失敗形狀。

**B. 位移平均的盲點。** 證明「位移平均」在 gold 邊際均勻時**量不到**可校正的選項偏誤，
   回收量恆為 0。這是為什麼主要指標是多數決而不是位移平均。這個測試存在的意義是：
   如果哪天有人把主要指標改回位移平均，它會失敗。

用法：
  python3 54_test_option_permutation.py
"""

from __future__ import annotations

import importlib.util
import json
import random
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
# 全量考卷（8.9 MB）沒有進 git；`tmmlu_testset.json` 是為了讓這支測試在乾淨 clone 下
# 也跑得起來而提交的平衡子集（2,000 題，四個 gold 字母各 500，取自同一份 test split）。
FULL_SAMPLE = REPO / "results" / "eval_raw" / "tmmlu_sample.json"
TEST_SAMPLE = REPO / "results" / "eval_raw" / "tmmlu_testset.json"

spec = importlib.util.spec_from_file_location(
    "perm", HERE / "54_tmmlu_option_permutation.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
L = m.LETTERS


def load_rows(force_testset: bool = False):
    src = TEST_SAMPLE if force_testset else (
        FULL_SAMPLE if FULL_SAMPLE.exists() else TEST_SAMPLE)
    if not src.exists():
        print(f"找不到 {FULL_SAMPLE} 或 {TEST_SAMPLE}。"
              f"跑 `51_eval_tmmlu.py --full --groups none --out-dir results/eval_raw` 重建考卷。")
        sys.exit(2)
    print(f"資料來源：{src.name}")
    rows = json.loads(src.read_text(encoding="utf-8"))
    # 選項內容重複的題目本來就會被 build() 排除，測試也要排除
    return [r for r in rows if len(set(str(r[x]) for x in L)) == 4]


def make_case(rows):
    mapping = {r["qid"]: {"subject": r["subject"], "gold_answer": r["gold_answer"],
                          "gold_index": L.index(r["gold_answer"])} for r in rows}
    perms = {(r["qid"], k): m.permute_row(r, k) for r in rows for k in range(m.N_PERM)}
    return mapping, perms


# ---------------------------------------------------------------------------
# A. 置換與換算的正確性
# ---------------------------------------------------------------------------

def test_structure(rows):
    for r in rows:
        orig = [str(r[x]) for x in L]
        for k in range(m.N_PERM):
            p = m.permute_row(r, k)
            assert sorted(str(p[x]) for x in L) == sorted(orig), "置換改動了選項內容"
            assert str(p[p["gold_answer"]]) == str(r[r["gold_answer"]]), "gold 對應的內容跑掉"
            assert p["question"] == r["question"], "題幹被動到"
        p0 = m.permute_row(r, 0)
        assert [str(p0[x]) for x in L] == orig, "k=0 不是恆等式"
        assert p0["gold_answer"] == r["gold_answer"], "k=0 動到 gold 字母"
    print("[PASS] 內容守恆、gold 跟著內容走、k=0 是恆等式")


def test_oracles(mapping, perms):
    # 全對
    s = m.score_group({key: p["gold_answer"] for key, p in perms.items()}, mapping)
    for k in ("single_order", "perm_averaged", "majority_vote"):
        assert abs(s[k]["micro"] - 1.0) < 1e-12, f"完美 oracle 的 {k} 不是 1.0"
    print("[PASS] 完美 oracle -> 三個指標都是 1.0000")

    # 全錯
    s = m.score_group(
        {key: next(x for x in L if x != p["gold_answer"]) for key, p in perms.items()}, mapping)
    for k in ("single_order", "perm_averaged", "majority_vote"):
        assert s[k]["micro"] == 0.0, f"反 oracle 的 {k} 不是 0.0"
    print("[PASS] 反 oracle -> 三個指標都是 0.0000")

    # 只會答固定字母：位移平均與多數決(期望) 都必須剛好等於亂猜的 0.25。
    # 這是整套換算的核心性質——若字母偏執沒有塌到 chance，代表換算是錯的。
    for fixed in L:
        s = m.score_group({key: fixed for key in perms}, mapping)
        avg, vote = s["perm_averaged"]["micro"], s["majority_vote"]["micro"]
        strict = s["majority_vote_strict"]["micro"]
        assert abs(avg - 0.25) < 1e-12, f"always-{fixed} 的位移平均 {avg} != 0.25"
        assert abs(vote - 0.25) < 1e-12, f"always-{fixed} 的多數決(期望) {vote} != 0.25"
        assert strict == 0.0, f"always-{fixed} 的多數決(平手算錯) {strict} != 0.0"
    print("[PASS] 只答固定字母 -> 位移平均 = 多數決(期望) = 0.2500（chance）")


# ---------------------------------------------------------------------------
# B. 位移平均的盲點（本實驗最重要的一條）
# ---------------------------------------------------------------------------

def test_perm_averaged_blindspot(rows):
    by_gold = {g: [r for r in rows if r["gold_answer"] == g] for g in L}
    n = min(min(len(v) for v in by_gold.values()), 1200)
    rng = random.Random(7)
    balanced = [r for g in L for r in rng.sample(by_gold[g], n)]
    marg = Counter(r["gold_answer"] for r in balanced)
    assert len(set(marg.values())) == 1, "子集的 gold 邊際不是完全均勻"

    mapping, perms = make_case(balanced)
    # 「知道答案、但從不說 B」——一個純粹的字母偏誤
    preds = {key: ("D" if p["gold_answer"] == "B" else p["gold_answer"])
             for key, p in perms.items()}
    s = m.score_group(preds, mapping)
    single = s["single_order"]["micro"]
    avg = s["perm_averaged"]["micro"]
    vote = s["majority_vote"]["micro"]

    assert abs(single - 0.75) < 1e-12, f"gold 均勻時單一順序應該剛好 0.75，得到 {single}"
    assert abs(avg - 0.75) < 1e-12, f"位移平均應該剛好 0.75，得到 {avg}"
    assert abs(avg - single) < 1e-12, "位移平均的回收量應該恆為 0"
    assert vote == 1.0, f"多數決應該完全回收這個偏誤，得到 {vote}"
    print(f"[PASS] gold 均勻 + 純字母偏誤：單一順序 {single:.4f}、位移平均 {avg:.4f}"
          f"（回收 {(avg-single)*100:+.2f} pp）、多數決 {vote:.4f}"
          f"（回收 {(vote-single)*100:+.2f} pp）")
    print("       -> 位移平均量不到選項偏誤；主要指標必須是多數決")

    # 對照：gold 偏斜時位移平均才會動，證明它量的是「對 gold 邊際不均的穩健度」
    # gold 偏斜的對照組：B 佔約 80%。切片大小依資料量調整，
    # 這樣全量考卷與 committed 子集都適用。
    nb = min(len(by_gold["B"]), 800)
    skewed = by_gold["B"][:nb] + by_gold["A"][:nb // 4]
    mapping2, perms2 = make_case(skewed)
    preds2 = {key: ("D" if p["gold_answer"] == "B" else p["gold_answer"])
              for key, p in perms2.items()}
    s2 = m.score_group(preds2, mapping2)
    rec = s2["perm_averaged"]["micro"] - s2["single_order"]["micro"]
    assert rec > 0.4, f"gold 偏斜時位移平均應該大幅回收，只得到 {rec}"
    print(f"[PASS] 同一個偏誤、gold 偏斜（80% 是 B）：位移平均回收 {rec*100:+.1f} pp")
    print("       -> 它真正量的是對 gold 邊際不均的穩健度")


def main() -> None:
    # --testset 強制走 committed 子集，用來驗證乾淨 clone 的路徑真的跑得起來
    rows = load_rows(force_testset="--testset" in sys.argv)
    rng = random.Random(0)
    subset = rng.sample(rows, min(400, len(rows)))
    print(f"用 {len(subset)} 題真實 TMMLU+ 題目測試\n")

    test_structure(subset)
    mapping, perms = make_case(subset)
    test_oracles(mapping, perms)
    print()
    test_perm_averaged_blindspot(rows)
    print("\n全部通過")


if __name__ == "__main__":
    main()
