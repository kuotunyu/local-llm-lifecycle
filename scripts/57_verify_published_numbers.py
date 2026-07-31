#!/usr/bin/env python3
"""從 committed 的逐題結果重算四個章節，逐欄比對已發表的 JSON。

存在理由：這個 repo 的宣稱是「clone 下來就能重算出 EVAL_REPORT 裡的數字」。
那句話唯一有意義的驗證方式，是真的重算一次再逐欄比對——而不是確認腳本沒有當掉。
README 第 178 行那張表列了五條「clone 後可直接跑」，這支腳本把其中四條自動化
（第五條 54_test_option_permutation.py 本身就是測試，CI 另外單獨跑）。

用法（在 repo 根目錄）：

    python3 scripts/57_verify_published_numbers.py

任何一欄對不上就 exit 1。只需要 numpy，不需要 GPU、不需要下載模型或資料集。
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# (章節標籤, 重算指令, 輸出檔參數, 已發表的 JSON, 比對時要忽略的頂層 key)
#
# 忽略的欄位都是「執行環境」而非「結果」：source_dir / dir_a / dir_b 記的是當初跑的
# 目錄路徑，換一台機器必然不同，拿它判定不一致沒有意義。其餘每一欄都要完全相同。
CASES = [
    (
        "§4.3 TMMLU+ 配對檢定與 CI",
        ["scripts/52_tmmlu_paired_stats.py", "--out-dir", "results/eval_raw"],
        "--stats-path",
        "results/tmmlu_paired_stats.json",
        {"source_dir"},
    ),
    (
        "§4.4 選項位移校正",
        ["scripts/54_tmmlu_option_permutation.py", "--analyze",
         "--compact", "results/eval_perm/permutation_predictions.jsonl"],
        "--stats-path",
        "results/tmmlu_option_permutation.json",
        set(),
    ),
    (
        "§6 batch 組成敏感度",
        ["scripts/53_batch_sensitivity.py",
         "--compact-b", "results/eval_perm/batch6144_predictions.jsonl"],
        "--out",
        "results/tmmlu_batch_sensitivity.json",
        {"dir_a", "dir_b"},
    ),
    (
        "§3.1-3.2 DRCD 等價界",
        ["scripts/56_drcd_paired_stats.py", "--eval-dir", "results/eval_raw"],
        "--out",
        "results/drcd_paired_stats.json",
        set(),
    ),
]

FLOAT_TOL = 1e-12


def diff(published, recomputed, path: str = "") -> list[str]:
    """回傳所有對不上的欄位路徑。浮點數用 1e-12 容差，其餘要求完全相等。"""
    out: list[str] = []
    if isinstance(published, dict):
        if not isinstance(recomputed, dict) or set(published) != set(recomputed):
            return [f"{path}（key 集合不同）"]
        for k in published:
            out += diff(published[k], recomputed[k], f"{path}.{k}")
    elif isinstance(published, list):
        if not isinstance(recomputed, list) or len(published) != len(recomputed):
            return [f"{path}（長度不同）"]
        for i, (a, b) in enumerate(zip(published, recomputed)):
            out += diff(a, b, f"{path}[{i}]")
    elif isinstance(published, float) or isinstance(recomputed, float):
        try:
            if abs(float(published) - float(recomputed)) > FLOAT_TOL:
                out.append(f"{path}: 已發表 {published} vs 重算 {recomputed}")
        except (TypeError, ValueError):
            out.append(f"{path}: 型別不符 {published!r} vs {recomputed!r}")
    elif published != recomputed:
        out.append(f"{path}: 已發表 {published!r} vs 重算 {recomputed!r}")
    return out


def check_readme() -> list[str]:
    """README 首屏宣稱表的每個數字，都要能從 results/ 的 JSON 算出來。

    這一關與上面的重算是同一個問題的兩面：重算驗的是「JSON 能不能從逐題資料復現」，
    這裡驗的是「README 寫的數字有沒有 JSON 撐著」。兩邊都過，那張宣稱表才不是空話。
    最容易發生的失誤是改了 results/ 卻忘了同步 README。
    """
    lines = (ROOT / "README.md").read_text(encoding="utf-8").splitlines()

    # 只看首屏那張宣稱表，不是整份 README。
    # 差別很重要：同一個數字在 README 裡出現好幾次（表格一次、後面敘述又一次），
    # 若只問「有沒有出現在檔案裡」，改壞表格裡那一處也驗得過——實測確認過會漏。
    start = next((i for i, ln in enumerate(lines) if ln.startswith("| 問題")), None)
    if start is None:
        return ["找不到首屏的宣稱表（開頭為 `| 問題` 的那一列）"]
    table = []
    for ln in lines[start:]:
        if not ln.startswith("|"):
            break
        table.append(ln)
    # README 用排版減號 U+2212，JSON 算出來是 ASCII 負號，比對前先統一
    readme = "\n".join(table).replace("−", "-")

    ps = json.loads((ROOT / "results/tmmlu_paired_stats.json").read_text(encoding="utf-8"))
    op = json.loads((ROOT / "results/tmmlu_option_permutation.json").read_text(encoding="utf-8"))

    delta = ps["point_estimate"]["delta_macro_accuracy"] * 100
    ci = ps["paired_stratified_bootstrap"]["delta_macro_accuracy"]
    recovered = op["recovered_pp"]["majority_vote"] * 100
    single = op["delta_macro"]["single_order"] * 100
    pct = round(recovered / abs(single) * 100)

    expect = [
        ("Δ macro 點估計", f"{delta:.2f}"),
        ("95% CI 下界", f"{ci['ci95_low'] * 100:.2f}"),
        ("95% CI 上界", f"{ci['ci95_high'] * 100:.2f}"),
        ("投票回收的百分點", f"{recovered:.2f}"),
        ("回收比例", f"{pct}%"),
    ]
    return [f"{label}：README 找不到 {value}" for label, value in expect if value not in readme]


def main() -> int:
    failed = 0
    with tempfile.TemporaryDirectory() as tmp:
        for i, (label, cmd, out_flag, published_path, ignore) in enumerate(CASES):
            pub = ROOT / published_path
            if not pub.exists():
                print(f"[未通過] {label}：找不到已發表的 {published_path}")
                failed += 1
                continue

            dest = Path(tmp) / f"{i}.json"
            # encoding 要指名 utf-8：Windows 上 text=True 預設用 cp950 解子行程的輸出，
            # 而這些腳本印的是中文 UTF-8，會在 subprocess 的 reader thread 裡拋
            # UnicodeDecodeError——那個例外不會讓 run() 失敗，只會讓 stderr 靜默遺失。
            proc = subprocess.run(
                [sys.executable, *cmd, out_flag, str(dest)],
                cwd=ROOT, capture_output=True, text=True,
                encoding="utf-8", errors="replace",
            )
            if proc.returncode != 0:
                print(f"[未通過] {label}：重算腳本 exit {proc.returncode}")
                print("         " + (proc.stderr.strip().splitlines() or ["(無 stderr)"])[-1])
                failed += 1
                continue
            if not dest.exists():
                print(f"[未通過] {label}：腳本沒有寫出 {dest.name}")
                failed += 1
                continue

            a = json.loads(pub.read_text(encoding="utf-8"))
            b = json.loads(dest.read_text(encoding="utf-8"))
            for k in ignore:
                a.pop(k, None)
                b.pop(k, None)

            d = diff(a, b)
            if d:
                print(f"[未通過] {label}：{len(d)} 欄對不上")
                for line in d[:5]:
                    print(f"         {line}")
                if len(d) > 5:
                    print(f"         …另外 {len(d) - 5} 欄")
                failed += 1
            else:
                n = len(json.dumps(a))
                print(f"[通過]   {label}（{n:,} 字元的 JSON 逐欄相同）")

    print()
    readme_problems = check_readme()
    if readme_problems:
        print(f"[未通過] README 首屏宣稱表：{len(readme_problems)} 個數字對不回 results/")
        for line in readme_problems:
            print(f"         {line}")
        failed += 1
    else:
        print("[通過]   README 首屏宣稱表的每個數字都對得回 results/ 的 JSON")

    print()
    total = len(CASES) + 1
    if failed:
        print(f"{total - failed}/{total} 關通過。")
        print("有對不上的——若這是預期中的更新，請一併更新 results/ 的 JSON 與 README。")
        return 1
    print(f"{total}/{total} 關通過：四個章節可從 committed 資料重算出已發表的數字，"
          "且 README 的宣稱數字全部對得回 results/。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
