"""Phase 5：TMMLU+ catastrophic forgetting 檢查（PLAN.md §2.3 / §4.5）。

66 科目均勻抽樣共 200 題，base（unsloth/Qwen3-8B）vs FT（合併後 bf16）**同在 transformers
bf16 下比**（隔離微調單一變因，不牽扯量化），報 macro-average accuracy 變化——
用來確認 DRCD QLoRA 微調有沒有明顯犧牲模型的一般知識能力。

TMMLU+ 透過 HF datasets-server REST API 直接抓（`requests`，不裝 `datasets` 套件，
專案 venv 目前沒有它，用 REST API 避免多一輪 pip install）。抽樣結果快取到
results/eval_raw/tmmlu_sample.json，重跑沿用同一份題目（base/FT 才能在完全相同的
200 題上比較）。

踩雷：66 科目 × 2 次呼叫（probe + fetch）連續打會撞到 datasets-server 的 429 rate
limit（第一次跑就中）。已修：`fetch_subject_rows` 加重試+指數退避，`sample_tmmlu`
科目間插 0.5s 固定延遲。

用法：
  python3 51_eval_tmmlu.py --base-model unsloth/Qwen3-8B --merged-dir work/merged \
      --out-dir results/eval_raw
  python3 51_eval_tmmlu.py --summarize --out-dir results/eval_raw
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import random
import re
import sys
import time
from pathlib import Path

# 必須在 torch 第一次 import（CUDA 配置器初始化）之前設定。同一支 process 內連續
# 載入兩顆模型（base、FT）跑生成，沒設這個會有顯存碎片化問題——50_eval_qa.py 的
# base_fewshot 事故就是這樣壞的；這裡兩顆模型間也觀察到同樣症狀（第二顆模型
# tmmlu_ft 慢了 200 倍，但輸出本身沒壞，純粹是配置器碎片化拖速度）。
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import requests

SUBJECTS = [
    "accounting", "administrative_law", "advance_chemistry", "agriculture",
    "anti_money_laundering", "auditing", "basic_medical_science", "business_management",
    "chinese_language_and_literature", "clinical_psychology", "computer_science",
    "culinary_skills", "dentistry", "economics", "education", "education_(profession_level)",
    "educational_psychology", "engineering_math", "finance_banking", "financial_analysis",
    "fire_science", "general_principles_of_law", "geography_of_taiwan", "human_behavior",
    "insurance_studies", "introduction_to_law", "jce_humanities", "junior_chemistry",
    "junior_chinese_exam", "junior_math_exam", "junior_science_exam", "junior_social_studies",
    "logic_reasoning", "macroeconomics", "management_accounting", "marketing_management",
    "mechanical", "music", "national_protection", "nautical_science",
    "occupational_therapy_for_psychological_disorders", "official_document_management",
    "optometry", "organic_chemistry", "pharmacology", "pharmacy", "physical_education",
    "physics", "politic_science", "real_estate", "secondary_physics",
    "statistics_and_machine_learning", "taiwanese_hokkien", "taxation", "technical",
    "three_principles_of_people", "trade", "traditional_chinese_medicine_clinical_medicine",
    "trust_practice", "ttqav2", "tve_chinese_language", "tve_design", "tve_mathematics",
    "tve_natural_sciences", "veterinary_pathology", "veterinary_pharmacology",
]
assert len(SUBJECTS) == 66

DATASETS_SERVER = "https://datasets-server.huggingface.co/rows"

SYSTEM_PROMPT_MC = (
    "你是知識淵博的助手。請閱讀題目和四個選項，僅回答正確選項的字母（A、B、C 或 D），"
    "不要輸出其他文字或說明。"
)


# ---------------------------------------------------------------------------
# TMMLU+ 抽樣（HF datasets-server REST API）
# ---------------------------------------------------------------------------

def fetch_subject_rows(subject: str, offset: int, length: int, max_retries: int = 10) -> list[dict]:
    """實測：datasets-server 對這支 IP 有滾動配額（觀察到約 60-65 次請求後開始
    連續 429，不是針對特定科目封鎖——前面科目幾乎都一次成功，卡在最後幾科），
    指數退避加大到 cap 60s、重試次數拉到 10 次，讓它有機會撐過配額重置窗口。"""
    for attempt in range(max_retries):
        resp = requests.get(
            DATASETS_SERVER,
            params={"dataset": "ikala/tmmluplus", "config": subject, "split": "test", "offset": offset, "length": length},
            timeout=30,
        )
        if resp.status_code == 429:
            wait = min(2 ** attempt, 60)
            print(f"  429 rate limit（{subject}），等 {wait}s 重試（第 {attempt+1}/{max_retries} 次）...", flush=True)
            time.sleep(wait)
            continue
        resp.raise_for_status()
        data = resp.json()
        return data["rows"], data["num_rows_total"]
    raise RuntimeError(f"{subject} 連續 {max_retries} 次撞 429，放棄")


def sample_tmmlu(n_total: int, seed: int, progress_path: Path | None = None) -> list[dict]:
    rng = random.Random(seed)
    base_k = n_total // len(SUBJECTS)
    remainder = n_total - base_k * len(SUBJECTS)
    bonus_subjects = set(rng.sample(SUBJECTS, remainder))

    # 逐科目落盤（JSONL，一科一行）：datasets-server 有滾動 rate limit，撐過
    # 60 幾科後容易卡住，若又失敗，重跑只需要補齊剩下的科目，不用整個重抓。
    done_subjects: dict[str, list[dict]] = {}
    if progress_path is not None and progress_path.exists():
        for line in progress_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                done_subjects[rec["subject"]] = rec["samples"]
        if done_subjects:
            print(f"沿用已抓取的 {len(done_subjects)} 個科目（{progress_path}）")

    progress_f = None
    if progress_path is not None:
        progress_path.parent.mkdir(parents=True, exist_ok=True)
        progress_f = open(progress_path, "a", encoding="utf-8")

    samples = []
    try:
        subjects_to_fetch = [s for s in SUBJECTS if s not in done_subjects]
        for subj_samples in done_subjects.values():
            samples.extend(subj_samples)

        for si, subject in enumerate(subjects_to_fetch):
            k = base_k + (1 if subject in bonus_subjects else 0)
            if k == 0:
                continue
            if si > 0:
                time.sleep(1.5)  # 每科目間隔，避免連續打爆 datasets-server 的 rate limit
            # 探測跟抓題合併成一次呼叫（原本 probe+fetch 兩次）：直接用固定
            # offset=0 抓 length=k，回應本身就帶 num_rows_total，不用再多打一次
            # ——66 科目直接砍半成 66 次呼叫。犧牲的是「科目內隨機 offset」，
            # 改成固定抓 test split 前 k 題；每科目樣本數本來就只有 3-4 題，
            # 隨機 offset 對代表性影響可忽略。
            offset = 0
            rows, total = fetch_subject_rows(subject, offset, k)
            if total == 0:
                print(f"  警告：{subject} test split 為空，略過")
                continue
            k = min(k, total, len(rows))
            subj_samples = []
            for i, r in enumerate(rows):
                row = r["row"]
                subj_samples.append(
                    {
                        "qid": f"{subject}-{offset + i}",
                        "subject": subject,
                        "question": row["question"],
                        "A": row["A"], "B": row["B"], "C": row["C"], "D": row["D"],
                        "gold_answer": row["answer"].strip().upper(),
                    }
                )
            samples.extend(subj_samples)
            if progress_f is not None:
                progress_f.write(json.dumps({"subject": subject, "samples": subj_samples}, ensure_ascii=False) + "\n")
                progress_f.flush()
            print(f"  [{si+1+len(done_subjects)}/{len(SUBJECTS)}] {subject}：{len(subj_samples)} 題", flush=True)
    finally:
        if progress_f is not None:
            progress_f.close()

    rng.shuffle(samples)
    return samples


def load_or_build_sample(cache_path: Path, n_total: int, seed: int) -> list[dict]:
    if cache_path.exists():
        print(f"沿用既有抽樣：{cache_path}")
        return json.loads(cache_path.read_text(encoding="utf-8"))
    print(f"從 datasets-server 抽樣 {n_total} 題（66 科目均勻抽樣，seed={seed}）...")
    progress_path = cache_path.with_suffix(".progress.jsonl")
    samples = sample_tmmlu(n_total, seed, progress_path=progress_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(samples, ensure_ascii=False, indent=2), encoding="utf-8")
    progress_path.unlink(missing_ok=True)
    print(f"抽樣完成，共 {len(samples)} 題，已快取至 {cache_path}")
    return samples


# ---------------------------------------------------------------------------
# prompt 建構 / 輸出解析
# ---------------------------------------------------------------------------

def build_mc_messages(row: dict) -> list[dict]:
    user_content = (
        f"題目：{row['question']}\n"
        f"A. {row['A']}\nB. {row['B']}\nC. {row['C']}\nD. {row['D']}\n"
        "請回答正確選項的字母："
    )
    return [{"role": "system", "content": SYSTEM_PROMPT_MC}, {"role": "user", "content": user_content}]


_LETTER_RE = re.compile(r"\b([ABCD])\b")
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def parse_letter(raw_text: str) -> str | None:
    text = _THINK_RE.sub("", raw_text).strip().upper()
    m = _LETTER_RE.search(text)
    return m.group(1) if m else None


# ---------------------------------------------------------------------------
# 斷點續跑 + transformers 批次推論
# ---------------------------------------------------------------------------

def load_done_qids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    qids = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            qids.add(json.loads(line)["qid"])
        except Exception:
            continue
    return qids


def run_mc_group(model, tokenizer, rows: list[dict], batch_size: int, max_new_tokens: int, out_path: Path, group: str) -> None:
    import torch

    done = load_done_qids(out_path)
    pending = [r for r in rows if r["qid"] not in done]
    print(f"[{group}] 待跑 {len(pending)} / 共 {len(rows)}（已完成 {len(done)}）")
    if not pending:
        return

    t0 = time.time()
    with open(out_path, "a", encoding="utf-8") as f:
        for i in range(0, len(pending), batch_size):
            batch = pending[i : i + batch_size]
            prompts = [
                tokenizer.apply_chat_template(
                    build_mc_messages(r), tokenize=False, add_generation_prompt=True, enable_thinking=False
                )
                for r in batch
            ]
            enc = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
            with torch.no_grad():
                out = model.generate(
                    **enc, max_new_tokens=max_new_tokens, do_sample=False,
                    temperature=None, top_p=None, top_k=None, pad_token_id=tokenizer.pad_token_id,
                )
            input_len = enc["input_ids"].shape[1]
            for j, row in enumerate(batch):
                text = tokenizer.decode(out[j][input_len:], skip_special_tokens=True)
                pred = parse_letter(text)
                rec = {
                    "qid": row["qid"], "subject": row["subject"], "gold_answer": row["gold_answer"],
                    "raw_output": text, "pred_answer": pred, "correct": pred == row["gold_answer"],
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            del enc, out
            if (i // batch_size + 1) % 8 == 0:
                torch.cuda.empty_cache()
            done_n = min(i + batch_size, len(pending))
            elapsed = time.time() - t0
            rate = done_n / elapsed if elapsed > 0 else 0
            print(f"  [{group}] {done_n}/{len(pending)}  {rate:.2f} 題/秒", flush=True)


def load_transformers_model(model_path: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"載入 transformers 模型：{model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    return model, tokenizer


def unload_transformers_model(model) -> None:
    import torch

    del model
    gc.collect()
    torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# 彙整
# ---------------------------------------------------------------------------

def summarize_group(path: Path) -> dict:
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    n = len(rows)
    if n == 0:
        return {"n": 0}
    by_subject: dict[str, list[bool]] = {}
    for r in rows:
        by_subject.setdefault(r["subject"], []).append(r["correct"])
    per_subject_acc = {s: sum(vs) / len(vs) for s, vs in by_subject.items()}
    macro_accuracy = sum(per_subject_acc.values()) / len(per_subject_acc)
    micro_accuracy = sum(r["correct"] for r in rows) / n
    return {
        "n": n,
        "n_subjects": len(by_subject),
        "macro_accuracy": macro_accuracy,
        "micro_accuracy": micro_accuracy,
        "per_subject_accuracy": per_subject_acc,
    }


def summarize_all(out_dir: Path) -> dict:
    summary = {}
    for group in ("tmmlu_base", "tmmlu_ft"):
        p = out_dir / f"{group}.jsonl"
        if p.exists():
            summary[group] = summarize_group(p)
    if "tmmlu_base" in summary and "tmmlu_ft" in summary:
        summary["delta_macro_accuracy"] = (
            summary["tmmlu_ft"]["macro_accuracy"] - summary["tmmlu_base"]["macro_accuracy"]
        )
        summary["delta_micro_accuracy"] = (
            summary["tmmlu_ft"]["micro_accuracy"] - summary["tmmlu_base"]["micro_accuracy"]
        )
    return summary


def print_summary(summary: dict) -> None:
    print("\n===== TMMLU+ forgetting check 彙整 =====")
    for group in ("tmmlu_base", "tmmlu_ft"):
        s = summary.get(group)
        if not s or s.get("n", 0) == 0:
            continue
        print(f"{group}: n={s['n']}  科目數={s['n_subjects']}  macro_acc={s['macro_accuracy']:.4f}  micro_acc={s['micro_accuracy']:.4f}")
    if "delta_macro_accuracy" in summary:
        print(f"\nΔ macro accuracy (FT − base) = {summary['delta_macro_accuracy']:+.4f}")
        print(f"Δ micro accuracy (FT − base) = {summary['delta_micro_accuracy']:+.4f}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default="unsloth/Qwen3-8B")
    parser.add_argument("--merged-dir")
    parser.add_argument("--out-dir", default="results/eval_raw")
    parser.add_argument("--n-total", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--groups", default="all", help="逗號分隔：tmmlu_base,tmmlu_ft 或 'all'")
    parser.add_argument("--summarize", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.summarize:
        summary = summarize_all(out_dir)
        (out_dir.parent / "tmmlu_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print_summary(summary)
        return

    groups = ["tmmlu_base", "tmmlu_ft"] if args.groups == "all" else [g.strip() for g in args.groups.split(",")]

    sample_cache = out_dir / "tmmlu_sample.json"
    rows = load_or_build_sample(sample_cache, args.n_total, args.seed)

    model_for_group = {"tmmlu_base": args.base_model, "tmmlu_ft": args.merged_dir}
    for g in groups:
        model_path = model_for_group.get(g)
        if not model_path:
            print(f"缺少模型路徑，略過：{g}")
            continue
        model, tokenizer = load_transformers_model(model_path)
        try:
            run_mc_group(model, tokenizer, rows, args.batch_size, args.max_new_tokens, out_dir / f"{g}.jsonl", g)
        finally:
            unload_transformers_model(model)

    summary = summarize_all(out_dir)
    (out_dir.parent / "tmmlu_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print_summary(summary)


if __name__ == "__main__":
    main()
