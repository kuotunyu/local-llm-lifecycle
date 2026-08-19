"""Phase 5：DRCD dev 五組對照評估（完整方法見 EVAL_REPORT）。

五組：
  1. base_zeroshot    — 原廠 base（未微調），transformers bf16，zero-shot
  2. base_fewshot     — 原廠 base，transformers bf16，3-shot（2 正 1 負）
  3. ft_unquantized   — 合併後 bf16（未量化），transformers bf16 —— 微調增益「上限」基準
  4. ft_q8            — GGUF Q8_0，llama-server —— 轉檔損耗探針（近無損）
  5. ft_q4            — GGUF Q4_K_M，llama-server —— 部署實態

核心問題：「Q4 吃掉多少微調增益」=（組3−組1）vs（組5−組1）；組4 用來切分損耗
是來自「轉檔（f16/Q8）」還是「Q8→Q4 量化」本身。

base 組（1、2）刻意用 unsloth/Qwen3-8B（跟 20_merge_lora.py 的 MERGE_BASE_MODEL 同一顆），
不用官方 Qwen/Qwen3-8B——因為那才是 LoRA 實際合併時的起點權重，同一份 base 才能讓
（組3−組1）乾淨地只反映微調本身，不會混入 unsloth tokenizer 修正版 vs 官方版的差異。

指標：CMRC2018 風格字元級 EM/F1（中文逐字、英數整詞切分，多參考答案取 max）、
JSON 合法率、answerable flag 準確率（含單獨的 NoAns 準確率／拒答召回率）。

組4/5 用 llama-server 的 continuous batching（-np N）+ 用戶端並發請求（ThreadPoolExecutor
＋ requests，效果等價 async client，省去額外裝 httpx/aiohttp）；不用 llama-cpp-python。

支援斷點續跑：每組結果逐題落盤到 results/eval_raw/<group>.jsonl，重跑會跳過已完成的 qid。

用法：
  # 先跑一個小 pilot 量測真實吞吐量
  python3 50_eval_qa.py --groups ft_q4 --n-samples 30 ...

  # 全量跑單一組
  python3 50_eval_qa.py --groups base_zeroshot --dev-answerable ... --dev-unanswerable ... \
      --few-shot-examples ... --out-dir results/eval_raw

  # 只彙整已完成的結果，不重新推論
  python3 50_eval_qa.py --summarize --out-dir results/eval_raw
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import re
import subprocess
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# 必須在 torch 第一次 import（CUDA 配置器初始化）之前設定。
# 降低長時間跑批次生成時的顯存碎片化（base_fewshot 效能事故的緩解之一）。
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import requests

SYSTEM_PROMPT = (
    "你是精確的閱讀理解助手。根據「文章」回答「問題」：\n"
    "- 答案必須是文章中的連續原文片段，一字不改\n"
    "- 若文章中找不到答案，answer 填空字串、answerable 填 false\n"
    "- 只輸出 JSON：{\"answer\": \"...\", \"answerable\": true|false}"
)

BASE_MODEL_DEFAULT = "unsloth/Qwen3-8B"
SERVER_PORT = 8712
SERVER_URL = f"http://127.0.0.1:{SERVER_PORT}"

# 固定順序（EVAL_REPORT 的組別順序）；用 set 只拿來做成員測試，不能拿來排序——
# Python 的 set 疊代順序受字串 hash 隨機化影響，每次執行可能不同，之前拿它排過
# 執行順序，害進度回報跟實際不一致（同一個模型的組別還是會排在一起執行，只是
# 順序不可預期），改成明確固定的 list。
ALL_GROUPS = ["base_zeroshot", "base_fewshot", "ft_unquantized", "ft_q8", "ft_q4"]
TRANSFORMERS_GROUPS = set(ALL_GROUPS[:3])
LLAMA_SERVER_GROUPS = set(ALL_GROUPS[3:])


# ---------------------------------------------------------------------------
# 資料載入
# ---------------------------------------------------------------------------

def load_dev_rows(dev_answerable: Path, dev_unanswerable: Path) -> list[dict]:
    rows = []
    for p in (dev_answerable, dev_unanswerable):
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return rows


def stratified_sample(rows: list[dict], n: int | None, seed: int) -> list[dict]:
    if n is None or n >= len(rows):
        return rows
    import random

    rng = random.Random(seed)
    pos = [r for r in rows if r["answerable"]]
    neg = [r for r in rows if not r["answerable"]]
    ratio = len(pos) / len(rows)
    n_pos = min(len(pos), round(n * ratio))
    n_neg = min(len(neg), n - n_pos)
    sampled = rng.sample(pos, n_pos) + rng.sample(neg, n_neg)
    rng.shuffle(sampled)
    return sampled


def build_fewshot_turns(examples: list[dict]) -> list[dict]:
    turns = []
    for ex in examples:
        user_content = f"文章：{ex['context']}\n\n問題：{ex['question']}"
        assistant_content = json.dumps(
            {"answer": ex["answer"], "answerable": ex["answerable"]}, ensure_ascii=False
        )
        turns.append({"role": "user", "content": user_content})
        turns.append({"role": "assistant", "content": assistant_content})
    return turns


def build_messages(row: dict, few_shot_turns: list[dict] | None) -> list[dict]:
    sys_msg, user_msg = row["messages"][0], row["messages"][1]
    msgs = [sys_msg]
    if few_shot_turns:
        msgs.extend(few_shot_turns)
    msgs.append(user_msg)
    return msgs


# ---------------------------------------------------------------------------
# 輸出解析
# ---------------------------------------------------------------------------

_JSON_RE = re.compile(r"\{.*?\}", re.DOTALL)
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def parse_prediction(raw_text: str) -> tuple[str, bool, bool]:
    """回傳 (answer, answerable, json_valid)。解析失敗一律視為錯誤答案。"""
    text = _THINK_RE.sub("", raw_text).strip()
    obj = None
    try:
        obj = json.loads(text)
    except Exception:
        m = _JSON_RE.search(text)
        if m:
            try:
                obj = json.loads(m.group(0))
            except Exception:
                obj = None
    if not isinstance(obj, dict) or "answer" not in obj or "answerable" not in obj:
        return "", False, False
    return str(obj.get("answer", "")), bool(obj.get("answerable", False)), True


# ---------------------------------------------------------------------------
# CMRC2018 風格字元級 EM/F1
# ---------------------------------------------------------------------------

_PUNCT = set(
    "-:_*^/\\~`+=，。：？！“”；’《》……·、「」（）－～『』,.:;!?()[]{}\"'"
)


def _mixed_segmentation(text: str) -> list[str]:
    segs, buf = [], ""
    for ch in text:
        if re.match(r"[一-鿿]", ch) or ch in _PUNCT:
            if buf:
                segs.extend(list(buf))
                buf = ""
            segs.append(ch)
        else:
            buf += ch
    if buf:
        segs.extend(list(buf))
    return segs


def _normalize(s: str) -> str:
    s = s.strip().lower()
    return "".join(ch for ch in s if ch not in _PUNCT)


def _drop_punct(tokens: list[str]) -> list[str]:
    return [t for t in tokens if t not in _PUNCT and t.strip() != ""]


def em_score(pred: str, gold: str) -> float:
    return 1.0 if _normalize(pred) == _normalize(gold) else 0.0


def f1_score(pred: str, gold: str) -> float:
    pred_tokens = _drop_punct(_mixed_segmentation(pred.strip().lower()))
    gold_tokens = _drop_punct(_mixed_segmentation(gold.strip().lower()))
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def _max_over_gold(pred: str, golds: list[str], fn) -> float:
    if not golds:
        golds = [""]
    return max(fn(pred, g) for g in golds)


def score_row(row: dict, pred_answer: str, pred_answerable: bool, json_valid: bool) -> dict:
    gold_answerable = row["answerable"]
    if gold_answerable:
        golds = row["gold_answers"]
        em = _max_over_gold(pred_answer, golds, em_score) if json_valid else 0.0
        f1 = _max_over_gold(pred_answer, golds, f1_score) if json_valid else 0.0
    else:
        # SQuAD2.0 慣例：gold 是 NoAns 時，只有 prediction 也給空字串才算 EM=F1=1
        is_empty_pred = pred_answer.strip() == ""
        em = 1.0 if (json_valid and is_empty_pred) else 0.0
        f1 = em
    answerable_correct = json_valid and (pred_answerable == gold_answerable)
    return {"em": em, "f1": f1, "answerable_correct": answerable_correct}


# ---------------------------------------------------------------------------
# 斷點續跑
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


def write_result(f, lock: threading.Lock, row: dict, raw_text: str) -> None:
    pred_answer, pred_answerable, json_valid = parse_prediction(raw_text)
    scores = score_row(row, pred_answer, pred_answerable, json_valid)
    rec = {
        "qid": row["qid"],
        "answerable": row["answerable"],
        "gold_answers": row["gold_answers"],
        "raw_output": raw_text,
        "pred_answer": pred_answer,
        "pred_answerable": pred_answerable,
        "json_valid": json_valid,
        **scores,
    }
    line = json.dumps(rec, ensure_ascii=False) + "\n"
    with lock:
        f.write(line)
        f.flush()


# ---------------------------------------------------------------------------
# transformers 引擎（組 1/2/3）
# ---------------------------------------------------------------------------

def run_transformers_group(
    model, tokenizer, rows: list[dict], few_shot_turns: list[dict] | None,
    batch_size: int, max_new_tokens: int, out_path: Path, group: str,
    max_batch_tokens: int = 12288,
) -> None:
    import torch

    done = load_done_qids(out_path)
    pending = [r for r in rows if r["qid"] not in done]
    print(f"[{group}] 待跑 {len(pending)} / 共 {len(rows)}（已完成 {len(done)}）")
    if not pending:
        return

    # 動態 batch：以「(prompt tokens + max_new_tokens) × batch 大小」為預算上限。
    # 固定 batch=16 對 few-shot 這種 ~1900 token 的長 prompt，KV cache 會把 24GB
    # 顯存撐爆，觸發 Windows WDDM 的 sysmem fallback（顯存溢到系統 RAM），速度
    # 掉 30 倍以上——base_fewshot 第一次跑就是這樣壞的。短 prompt（zero-shot）
    # 在這個預算下仍然能組到 batch_size 上限，速度不受影響。
    # 依 token 長度排序讓同 batch 長度相近，減少 padding 浪費（結果逐題落盤、
    # 各自帶 qid，處理順序不影響評分）。
    print(f"[{group}] 渲染 prompt 並量測 token 長度（動態 batch 預算 {max_batch_tokens} tokens）...")
    rendered = []
    for r in pending:
        prompt = tokenizer.apply_chat_template(
            build_messages(r, few_shot_turns),
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        rendered.append((r, prompt, len(tokenizer(prompt)["input_ids"])))
    rendered.sort(key=lambda x: x[2])

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
    print(f"[{group}] 共 {len(batches)} 個 batch（大小 {min(len(b) for b in batches)}-{max(len(b) for b in batches)}）")

    lock = threading.Lock()
    t0 = time.time()
    done_n = 0
    with open(out_path, "a", encoding="utf-8") as f:
        for bi, batch in enumerate(batches):
            prompts = [p for _, p, _ in batch]
            enc = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
            with torch.no_grad():
                out = model.generate(
                    **enc,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                    top_k=None,
                    pad_token_id=tokenizer.pad_token_id,
                )
            input_len = enc["input_ids"].shape[1]
            for j, (row, _, _) in enumerate(batch):
                gen_ids = out[j][input_len:]
                text = tokenizer.decode(gen_ids, skip_special_tokens=True)
                write_result(f, lock, row, text)
            done_n += len(batch)
            del enc, out
            if (bi + 1) % 8 == 0:
                torch.cuda.empty_cache()
            elapsed = time.time() - t0
            rate = done_n / elapsed if elapsed > 0 else 0
            eta = (len(pending) - done_n) / rate if rate > 0 else float("inf")
            print(
                f"  [{group}] {done_n}/{len(pending)}  {rate:.2f} 題/秒  "
                f"預估剩餘 {eta/60:.1f} 分鐘",
                flush=True,
            )


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
# llama-server 引擎（組 4/5）
# ---------------------------------------------------------------------------

def start_llama_server(llama_server_bin: str, gguf_path: Path, concurrency: int, per_request_ctx: int) -> tuple[subprocess.Popen, Path]:
    log_path = gguf_path.parent / f"eval-server-{gguf_path.stem}.log"
    log_f = open(log_path, "w", encoding="utf-8")
    ctx_total = concurrency * per_request_ctx
    proc = subprocess.Popen(
        [
            llama_server_bin,
            "-m", str(gguf_path),
            "-ngl", "99",
            "-c", str(ctx_total),
            "-np", str(concurrency),
            "--port", str(SERVER_PORT),
            "--jinja",
        ],
        stdout=log_f,
        stderr=subprocess.STDOUT,
    )
    ready = False
    for attempt in range(180):
        if proc.poll() is not None:
            print(f"llama-server 行程已結束（exit code {proc.returncode}），見 {log_path}")
            break
        try:
            r = requests.get(f"{SERVER_URL}/health", timeout=2)
            if r.status_code == 200:
                print(f"llama-server 就緒（等了 {attempt}s，-np {concurrency}，-c {ctx_total}）")
                ready = True
                break
        except requests.exceptions.RequestException:
            pass
        time.sleep(1)
    if not ready:
        log_f.flush()
        print(f"llama-server 未就緒，日誌：\n{log_path.read_text(encoding='utf-8')[-2000:]}")
        proc.kill()
        sys.exit(1)
    return proc, log_path


def stop_llama_server(proc: subprocess.Popen) -> None:
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=10)


def run_llama_server_group(
    gguf_path: Path, llama_server_bin: str, rows: list[dict],
    concurrency: int, max_new_tokens: int, per_request_ctx: int, out_path: Path, group: str,
) -> None:
    done = load_done_qids(out_path)
    pending = [r for r in rows if r["qid"] not in done]
    print(f"[{group}] 待跑 {len(pending)} / 共 {len(rows)}（已完成 {len(done)}）")
    if not pending:
        return

    proc, _ = start_llama_server(llama_server_bin, gguf_path, concurrency, per_request_ctx)
    lock = threading.Lock()
    t0 = time.time()
    n_done = [0]
    try:
        with open(out_path, "a", encoding="utf-8") as f:

            def worker(row: dict) -> None:
                messages = build_messages(row, None)
                resp = requests.post(
                    f"{SERVER_URL}/v1/chat/completions",
                    json={
                        "messages": messages,
                        "temperature": 0,
                        "max_tokens": max_new_tokens,
                        "seed": 42,
                        "chat_template_kwargs": {"enable_thinking": False},
                    },
                    timeout=120,
                )
                resp.raise_for_status()
                text = resp.json()["choices"][0]["message"]["content"]
                write_result(f, lock, row, text)

            with ThreadPoolExecutor(max_workers=concurrency) as ex:
                futures = {ex.submit(worker, r): r for r in pending}
                for fut in as_completed(futures):
                    fut.result()
                    n_done[0] += 1
                    if n_done[0] % 50 == 0 or n_done[0] == len(pending):
                        elapsed = time.time() - t0
                        rate = n_done[0] / elapsed if elapsed > 0 else 0
                        eta = (len(pending) - n_done[0]) / rate if rate > 0 else float("inf")
                        print(
                            f"  [{group}] {n_done[0]}/{len(pending)}  {rate:.2f} 題/秒  "
                            f"預估剩餘 {eta/60:.1f} 分鐘",
                            flush=True,
                        )
    finally:
        stop_llama_server(proc)


# ---------------------------------------------------------------------------
# 彙整
# ---------------------------------------------------------------------------

def summarize_group(path: Path) -> dict:
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    n = len(rows)
    if n == 0:
        return {"n": 0}
    hasans = [r for r in rows if r["answerable"]]
    noans = [r for r in rows if not r["answerable"]]
    return {
        "n": n,
        "n_hasans": len(hasans),
        "n_noans": len(noans),
        "json_valid_rate": sum(r["json_valid"] for r in rows) / n,
        "overall_em": sum(r["em"] for r in rows) / n,
        "overall_f1": sum(r["f1"] for r in rows) / n,
        "hasans_em": (sum(r["em"] for r in hasans) / len(hasans)) if hasans else None,
        "hasans_f1": (sum(r["f1"] for r in hasans) / len(hasans)) if hasans else None,
        "noans_accuracy": (sum(r["em"] for r in noans) / len(noans)) if noans else None,
        "answerable_flag_accuracy": sum(r["answerable_correct"] for r in rows) / n,
    }


def summarize_all(out_dir: Path) -> dict:
    summary = {}
    for group in ALL_GROUPS:
        p = out_dir / f"{group}.jsonl"
        if p.exists():
            summary[group] = summarize_group(p)
    return summary


def print_summary_table(summary: dict) -> None:
    print("\n===== 五組對照彙整 =====")
    header = f"{'group':<16}{'n':>6}{'json_valid':>11}{'overall_EM':>11}{'overall_F1':>11}{'hasans_EM':>11}{'hasans_F1':>11}{'noans_acc':>11}{'flag_acc':>10}"
    print(header)
    for group, s in summary.items():
        if s.get("n", 0) == 0:
            continue

        def fmt(v):
            return f"{v:.4f}" if isinstance(v, float) else "  -   "

        print(
            f"{group:<16}{s['n']:>6}{fmt(s['json_valid_rate']):>11}{fmt(s['overall_em']):>11}"
            f"{fmt(s['overall_f1']):>11}{fmt(s['hasans_em']):>11}{fmt(s['hasans_f1']):>11}"
            f"{fmt(s['noans_accuracy']):>11}{fmt(s['answerable_flag_accuracy']):>10}"
        )


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--groups", default="all", help="逗號分隔，或 'all'")
    parser.add_argument("--dev-answerable")
    parser.add_argument("--dev-unanswerable")
    parser.add_argument("--few-shot-examples")
    parser.add_argument("--base-model", default=BASE_MODEL_DEFAULT)
    parser.add_argument("--merged-dir")
    parser.add_argument("--gguf-q8")
    parser.add_argument("--gguf-q4")
    parser.add_argument("--llama-server-bin")
    parser.add_argument("--out-dir", default="results/eval_raw")
    parser.add_argument("--n-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-batch-tokens", type=int, default=12288,
                        help="transformers 組動態 batch 的 token 預算上限（(prompt+max_new)×batch ≤ 此值）")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--per-request-ctx", type=int, default=2048)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--summarize", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.summarize:
        summary = summarize_all(out_dir)
        (out_dir.parent / "eval_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print_summary_table(summary)
        return

    groups = ALL_GROUPS if args.groups == "all" else [g.strip() for g in args.groups.split(",")]
    for g in groups:
        if g not in ALL_GROUPS:
            print(f"未知的 group: {g}（可用：{ALL_GROUPS}）")
            sys.exit(1)

    rows = load_dev_rows(Path(args.dev_answerable), Path(args.dev_unanswerable))
    rows = stratified_sample(rows, args.n_samples, args.seed)
    print(f"評估集大小：{len(rows)}（answerable={sum(r['answerable'] for r in rows)}）")

    few_shot_turns = None
    if "base_fewshot" in groups:
        examples = json.loads(Path(args.few_shot_examples).read_text(encoding="utf-8"))
        few_shot_turns = build_fewshot_turns(examples)

    # --- transformers 組：同一顆模型的組別一起跑完再換模型，避免兩顆 8B 同時佔 VRAM ---
    transformer_groups = [g for g in groups if g in TRANSFORMERS_GROUPS]
    model_for_group = {
        "base_zeroshot": args.base_model,
        "base_fewshot": args.base_model,
        "ft_unquantized": args.merged_dir,
    }
    models_needed = {}
    for g in transformer_groups:
        models_needed.setdefault(model_for_group[g], []).append(g)

    for model_path, gs in models_needed.items():
        if not model_path:
            print(f"缺少模型路徑，略過：{gs}")
            continue
        model, tokenizer = load_transformers_model(model_path)
        try:
            for g in gs:
                fs = few_shot_turns if g == "base_fewshot" else None
                run_transformers_group(
                    model, tokenizer, rows, fs, args.batch_size, args.max_new_tokens,
                    out_dir / f"{g}.jsonl", g, max_batch_tokens=args.max_batch_tokens,
                )
        finally:
            unload_transformers_model(model)

    # --- llama-server 組：依序啟停 server，避免搶 port/GPU ---
    server_groups = [g for g in groups if g in LLAMA_SERVER_GROUPS]
    gguf_for_group = {"ft_q8": args.gguf_q8, "ft_q4": args.gguf_q4}
    for g in server_groups:
        gguf_path = gguf_for_group[g]
        if not gguf_path or not args.llama_server_bin:
            print(f"缺少 --gguf-* 或 --llama-server-bin，略過：{g}")
            continue
        run_llama_server_group(
            Path(gguf_path), args.llama_server_bin, rows, args.concurrency,
            args.max_new_tokens, args.per_request_ctx, out_dir / f"{g}.jsonl", g,
        )

    summary = summarize_all(out_dir)
    (out_dir.parent / "eval_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print_summary_table(summary)


if __name__ == "__main__":
    main()
