"""Phase 5：TMMLU+ 微調後的通用能力退步檢查（完整方法見 EVAL_REPORT）。

TMMLU+ 66 科目 test split **全量 20,118 題**，base（unsloth/Qwen3-8B）vs FT（合併後
bf16）**同在 transformers bf16 下比**（隔離微調單一變因，不牽扯量化），報 macro-average
accuracy 變化——用來確認 DRCD QLoRA 微調有沒有明顯犧牲模型的一般知識能力。

資料來源用 `datasets` 套件在本機載入（`load_dataset("ikala/tmmluplus", subject,
split="test")`）。**原本走 HF datasets-server REST API，已於 2026-07-30 換掉**：REST API
單次 `length` 上限 100，全量要分頁 200 次以上，而實測約 60-65 次請求後就開始連續撞 429；
本機載入沒有這個限制，而且跟姊妹專案「TMMLU+ 評測擂台」的讀法一致。
換資料來源時做過控制：用新路徑重建同一份 200 題抽樣，與舊 REST API 產出的
`tmmlu_sample.json` 逐位元相同（200/200 題全對，含題幹、四個選項與 gold answer）。

注意 `n=22,690` 是 TMMLU+ **三個 split 的總和**（train 330 + validation 2,242 +
test 20,118）。評測只用 test split，所以全量是 20,118 題。

抽樣結果快取到 results/eval_raw/tmmlu_sample.json，重跑沿用同一份題目（base/FT 才能在
完全相同的題目上比較）。**改變規模時要先刪掉這個快取**，否則會沿用舊考卷。

推論用**動態 batch（token 預算制）**，不是固定 batch size——理由與實測見 `run_mc_group`。

可重現性的邊界（2026-07-30 實測，值得記住）：greedy decoding 在**同一組 batch 組成**下
逐位元可重現（同一份考卷跑兩次結果 SHA-256 相同），但**換 batch 組成會動到結果**：
把固定 batch=16 換成長度排序的動態 batch，200 題裡 base 有 2 題、FT 有 4 題（1-2%）
預測字母改變，Δ macro 因此從 −12.25pp 變成 −10.73pp。原因是 bf16 矩陣乘法的規約順序
隨 padding/batch 形狀改變，邊界題目的 logit 差距小到足以被翻轉。
所以「greedy 就是決定性的」只在固定 batch 組成下成立；報告數字時要連 batch 設定一起講，
而且這正是**不能只報點估計**的另一個理由。

用法：
  # 全量 test split（20,118 題 × 2 個權重）
  python3 51_eval_tmmlu.py --base-model unsloth/Qwen3-8B --merged-dir work/merged \
      --out-dir results/eval_raw --full
  # 抽樣模式（舊行為，每科均勻抽樣至指定總題數）
  python3 51_eval_tmmlu.py --merged-dir work/merged --n-total 200
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

DATASET_ID = "ikala/tmmluplus"

SYSTEM_PROMPT_MC = (
    "你是知識淵博的助手。請閱讀題目和四個選項，僅回答正確選項的字母（A、B、C 或 D），"
    "不要輸出其他文字或說明。"
)


# ---------------------------------------------------------------------------
# TMMLU+ 取題（datasets 套件，本機載入）
# ---------------------------------------------------------------------------

def load_subject_test_split(subject: str):
    """載入單一科目的 test split。第一次會下載並轉成 arrow 快取到
    ~/.cache/huggingface/datasets，之後重跑都是本機讀取，沒有 rate limit。"""
    from datasets import load_dataset

    return load_dataset(DATASET_ID, subject, split="test")


def rows_to_samples(subject: str, ds, indices) -> list[dict]:
    """qid 用 `<subject>-<test split 的列序>`。這個編號規則跟舊的 REST API 版本一致
    （舊版抓 offset=0、length=k，qid 就是 0..k-1），所以新舊產出的題目可以直接對齊比對。"""
    out = []
    for i in indices:
        row = ds[int(i)]
        out.append(
            {
                "qid": f"{subject}-{i}",
                "subject": subject,
                "question": row["question"],
                "A": row["A"], "B": row["B"], "C": row["C"], "D": row["D"],
                "gold_answer": row["answer"].strip().upper(),
            }
        )
    return out


def sample_tmmlu(n_total: int, seed: int, full: bool = False) -> list[dict]:
    """full=True 取 66 科 test split 全量（20,118 題）；否則每科均勻抽樣至 n_total 題。

    抽樣模式取每科 test split 的**前 k 題**（不是隨機 offset）。這是舊 REST API 版本
    留下的行為，保留是為了讓 200 題的歷史結果能被逐位元復現；全量模式沒有這個問題。
    """
    rng = random.Random(seed)

    if full:
        samples = []
        for si, subject in enumerate(SUBJECTS):
            ds = load_subject_test_split(subject)
            samples.extend(rows_to_samples(subject, ds, range(len(ds))))
            print(f"  [{si+1}/{len(SUBJECTS)}] {subject}：{len(ds)} 題（累計 {len(samples)}）", flush=True)
        rng.shuffle(samples)
        return samples

    base_k = n_total // len(SUBJECTS)
    remainder = n_total - base_k * len(SUBJECTS)
    # rng 的呼叫順序會影響最後的 shuffle 結果，不要動：先 sample 再 shuffle
    bonus_subjects = set(rng.sample(SUBJECTS, remainder))

    samples = []
    for si, subject in enumerate(SUBJECTS):
        k = base_k + (1 if subject in bonus_subjects else 0)
        if k == 0:
            continue
        ds = load_subject_test_split(subject)
        if len(ds) == 0:
            print(f"  警告：{subject} test split 為空，略過")
            continue
        k = min(k, len(ds))
        samples.extend(rows_to_samples(subject, ds, range(k)))
        print(f"  [{si+1}/{len(SUBJECTS)}] {subject}：{k} 題", flush=True)

    rng.shuffle(samples)
    return samples


def load_or_build_sample(cache_path: Path, n_total: int, seed: int, full: bool = False) -> list[dict]:
    if cache_path.exists():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        print(f"沿用既有抽樣：{cache_path}（{len(cached)} 題）")
        # 換規模卻忘了刪快取是最容易犯的錯，會安靜地拿舊考卷跑新實驗
        if not full and len(cached) != n_total:
            print(f"  警告：快取有 {len(cached)} 題，但 --n-total 是 {n_total}。"
                  f"要換規模請先刪掉 {cache_path}。")
        return cached
    if full:
        print(f"從 {DATASET_ID} 載入 66 科目 test split 全量...")
    else:
        print(f"從 {DATASET_ID} 抽樣 {n_total} 題（66 科目均勻抽樣，seed={seed}）...")
    samples = sample_tmmlu(n_total, seed, full=full)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(samples, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"取題完成，共 {len(samples)} 題，已快取至 {cache_path}")
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


def run_mc_group(model, tokenizer, rows: list[dict], batch_size: int, max_new_tokens: int,
                 out_path: Path, group: str, max_batch_tokens: int = 12288) -> None:
    import torch

    done = load_done_qids(out_path)
    pending = [r for r in rows if r["qid"] not in done]
    print(f"[{group}] 待跑 {len(pending)} / 共 {len(rows)}（已完成 {len(done)}）")
    if not pending:
        return

    # 動態 batch（token 預算制），跟 50_eval_qa.py 的 `run_transformers_group` 同一套。
    # 這裡原本是固定 batch=16，200 題抽樣時每科都取 test split 前 3 題、剛好都很短，
    # 所以一直沒事；換成全量 20,118 題之後撞進長題目（TMMLU+ 有帶長文與表格的題型），
    # 固定 batch=16 的 activation/KV 需求直接把 24 GB 打爆而 OOM。
    # 這是同一種病的第三次發作（前兩次：50_eval_qa.py 的 base_fewshot、本腳本的
    # 顯存碎片化），根因都是「用固定 batch 去餵長度分佈很寬的 prompt」。
    print(f"[{group}] 渲染 prompt 並量測 token 長度（動態 batch 預算 {max_batch_tokens} tokens）...", flush=True)
    rendered = []
    for r in pending:
        prompt = tokenizer.apply_chat_template(
            build_mc_messages(r), tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        rendered.append((r, prompt, len(tokenizer(prompt)["input_ids"])))
    # 依 token 長度排序讓同 batch 長度相近，減少 padding 浪費。結果逐題落盤、各自帶
    # qid，處理順序不影響評分。
    rendered.sort(key=lambda x: x[2])
    lens = [n for _, _, n in rendered]
    print(f"[{group}] prompt token 長度：min {min(lens)} / 中位 {lens[len(lens)//2]} / max {max(lens)}", flush=True)

    batches: list[list[tuple[dict, str, int]]] = []
    cur: list[tuple[dict, str, int]] = []
    cur_max_len = 0
    for item in rendered:
        need_len = max(cur_max_len, item[2] + max_new_tokens)
        if cur and ((len(cur) + 1) * need_len > max_batch_tokens or len(cur) >= batch_size):
            batches.append(cur)
            cur, cur_max_len = [item], item[2] + max_new_tokens
        else:
            cur.append(item)
            cur_max_len = need_len
    if cur:
        batches.append(cur)
    print(f"[{group}] 共 {len(batches)} 個 batch（大小 {min(len(b) for b in batches)}-{max(len(b) for b in batches)}）", flush=True)

    t0 = time.time()
    done_n = 0
    with open(out_path, "a", encoding="utf-8") as f:
        for bi, batch in enumerate(batches):
            prompts = [p for _, p, _ in batch]
            enc = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
            with torch.no_grad():
                out = model.generate(
                    **enc, max_new_tokens=max_new_tokens, do_sample=False,
                    temperature=None, top_p=None, top_k=None, pad_token_id=tokenizer.pad_token_id,
                )
            input_len = enc["input_ids"].shape[1]
            for j, (row, _, _) in enumerate(batch):
                text = tokenizer.decode(out[j][input_len:], skip_special_tokens=True)
                pred = parse_letter(text)
                rec = {
                    "qid": row["qid"], "subject": row["subject"], "gold_answer": row["gold_answer"],
                    "raw_output": text, "pred_answer": pred, "correct": pred == row["gold_answer"],
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            done_n += len(batch)
            del enc, out
            if (bi + 1) % 8 == 0:
                torch.cuda.empty_cache()
            if (bi + 1) % 20 == 0 or done_n == len(pending):
                elapsed = time.time() - t0
                rate = done_n / elapsed if elapsed > 0 else 0
                eta = (len(pending) - done_n) / rate if rate > 0 else float("inf")
                print(f"  [{group}] {done_n}/{len(pending)}  {rate:.2f} 題/秒  ETA {eta/60:.1f} 分",
                      flush=True)


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


def free_cuda_memory() -> None:
    """把顯存還給配置器。

    這裡原本是 `unload_transformers_model(model)`，內部只做 `del model`——那只解除
    **函式自己的區域綁定**，呼叫端的名字還指著同一顆模型，所以什麼都沒釋放。後果是
    同一支 process 連續跑 base 與 FT 時，兩顆 16.4 GB bf16 會同時待在 VRAM
    （32.8 GB > 24 GB）而必定 OOM；當初是靠分兩次執行（`--groups tmmlu_base` 再
    `--groups tmmlu_ft`）繞過去的。

    正解是讓**呼叫端**自己 `del`，再叫這支把配置器清乾淨。
    """
    import torch

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
        # 這裡用 ASCII 連字號而不是 U+2212：Windows console 預設 cp950 編不出 U+2212，
        # 印出來會直接 UnicodeEncodeError 中斷腳本。註解與 HF card 裡的 U+2212 不受影響，
        # 那些不經過 stdout 編碼。
        print(f"\nΔ macro accuracy (FT - base) = {summary['delta_macro_accuracy']:+.4f}")
        print(f"Δ micro accuracy (FT - base) = {summary['delta_micro_accuracy']:+.4f}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default="unsloth/Qwen3-8B")
    parser.add_argument("--merged-dir")
    parser.add_argument("--out-dir", default="results/eval_raw")
    parser.add_argument("--n-total", type=int, default=200,
                        help="抽樣模式的總題數（每科均勻分配）。全量請改用 --full")
    parser.add_argument("--full", action="store_true",
                        help="跑 66 科 test split 全量（20,118 題），忽略 --n-total")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=16, help="動態 batch 的大小上限")
    parser.add_argument("--max-batch-tokens", type=int, default=12288,
                        help="動態 batch 的 token 預算上限（(prompt+max_new)×batch ≤ 此值）")
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
    rows = load_or_build_sample(sample_cache, args.n_total, args.seed, full=args.full)

    model_for_group = {"tmmlu_base": args.base_model, "tmmlu_ft": args.merged_dir}
    for g in groups:
        model_path = model_for_group.get(g)
        if not model_path:
            print(f"缺少模型路徑，略過：{g}")
            continue
        model, tokenizer = load_transformers_model(model_path)
        try:
            run_mc_group(model, tokenizer, rows, args.batch_size, args.max_new_tokens,
                         out_dir / f"{g}.jsonl", g, max_batch_tokens=args.max_batch_tokens)
        finally:
            # `del` 一定要在呼叫端做——這裡是最後一個指向模型的參照，放掉之後
            # free_cuda_memory() 才有東西可以回收（見該函式的說明）。
            del model, tokenizer
            free_cuda_memory()

    summary = summarize_all(out_dir)
    (out_dir.parent / "tmmlu_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print_summary(summary)


if __name__ == "__main__":
    main()
