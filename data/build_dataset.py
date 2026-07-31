"""
Phase 1: DRCD -> SFT 資料集建構

流程：下載官方 DRCD 三個 JSON -> 統計 -> 合成 unanswerable 負例（同文章跨段落為主、
跨文章隨機為輔，含假負例過濾）-> 產出 SFT chat-format jsonl -> （可選）推到 HF private dataset。

詳細設計依據見 PLAN.md §2.2（DRCD 事實）、§4.2（負例合成）、§4.3（SFT 格式）、Phase 1 驗收標準。
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
from dataclasses import dataclass, field
from pathlib import Path

import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"

DRCD_BASE_URL = "https://raw.githubusercontent.com/DRCKnowledgeTeam/DRCD/master/"
DRCD_FILES = {
    "train": "DRCD_training.json",
    "dev": "DRCD_dev.json",
    "test": "DRCD_test.json",
}
# PLAN.md §2.2 查證數字，下載後用來驗證來源沒有偷改
EXPECTED_QUESTION_COUNTS = {"train": 26936, "dev": 3524, "test": 3493}

SYSTEM_PROMPT = (
    "你是精確的閱讀理解助手。根據「文章」回答「問題」：\n"
    "- 答案必須是文章中的連續原文片段，一字不改\n"
    "- 若文章中找不到答案，answer 填空字串、answerable 填 false\n"
    "- 只輸出 JSON：{\"answer\": \"...\", \"answerable\": true|false}"
)


# --------------------------------------------------------------------------
# 下載與解析
# --------------------------------------------------------------------------

def download_drcd(force: bool = False) -> dict[str, Path]:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    paths = {}
    for split, fname in DRCD_FILES.items():
        dest = RAW_DIR / fname
        if dest.exists() and not force:
            print(f"[download] {split}: 已快取 {dest}")
        else:
            url = DRCD_BASE_URL + fname
            print(f"[download] {split}: {url}")
            resp = requests.get(url, timeout=60)
            resp.raise_for_status()
            dest.write_bytes(resp.content)
        paths[split] = dest
    return paths


@dataclass
class Paragraph:
    article_id: str
    article_title: str
    paragraph_idx: int  # 在文章內的順序
    paragraph_id: str
    context: str


@dataclass
class Question:
    split: str
    qid: str
    question: str
    article_id: str
    article_title: str
    paragraph_id: str
    paragraph_idx: int
    context: str
    answers: list[str]  # 全部參考答案（train 只有 1 個，dev/test 多個）


def load_split(split: str, path: Path) -> tuple[list[Paragraph], list[Question]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    paragraphs: list[Paragraph] = []
    questions: list[Question] = []
    for article in raw["data"]:
        article_id = str(article["id"])
        article_title = article["title"]
        for p_idx, para in enumerate(article["paragraphs"]):
            paragraphs.append(
                Paragraph(
                    article_id=article_id,
                    article_title=article_title,
                    paragraph_idx=p_idx,
                    paragraph_id=str(para["id"]),
                    context=para["context"],
                )
            )
            for qa in para["qas"]:
                questions.append(
                    Question(
                        split=split,
                        qid=str(qa["id"]),
                        question=qa["question"],
                        article_id=article_id,
                        article_title=article_title,
                        paragraph_id=str(para["id"]),
                        paragraph_idx=p_idx,
                        context=para["context"],
                        answers=[a["text"] for a in qa["answers"]],
                    )
                )
    return paragraphs, questions


def build_article_index(paragraphs: list[Paragraph]) -> dict[str, list[Paragraph]]:
    """article_id -> 該文章底下所有段落（依原順序）"""
    index: dict[str, list[Paragraph]] = {}
    for p in paragraphs:
        index.setdefault(p.article_id, []).append(p)
    return index


def check_train_dev_leakage(train_paragraphs: list[Paragraph], dev_paragraphs: list[Paragraph]) -> dict:
    """DRCD 官方是「段落級」切分，同一篇 Wikipedia 文章可能同時貢獻段落給 train 和 dev
    （article_id 因此會重疊），這不是資料洩漏；真正要驗證的是「段落文字」本身互不重複。
    """
    train_idx = build_article_index(train_paragraphs)
    dev_idx = build_article_index(dev_paragraphs)
    shared_article_ids = set(train_idx) & set(dev_idx)

    train_contexts_by_article = {aid: {p.context for p in ps} for aid, ps in train_idx.items()}
    leaked_contexts = 0
    dev_paras_under_shared = 0
    for aid in shared_article_ids:
        train_ctx = train_contexts_by_article[aid]
        for p in dev_idx[aid]:
            dev_paras_under_shared += 1
            if p.context in train_ctx:
                leaked_contexts += 1

    return {
        "shared_article_count": len(shared_article_ids),
        "dev_paragraphs_under_shared_articles": dev_paras_under_shared,
        "leaked_paragraph_count": leaked_contexts,
        "clean": leaked_contexts == 0,
    }


# --------------------------------------------------------------------------
# 統計
# --------------------------------------------------------------------------

def _len_stats(lengths: list[int]) -> dict:
    if not lengths:
        return {}
    lengths_sorted = sorted(lengths)

    def pct(p: float) -> int:
        idx = min(len(lengths_sorted) - 1, int(len(lengths_sorted) * p))
        return lengths_sorted[idx]

    return {
        "min": min(lengths),
        "mean": round(statistics.mean(lengths), 1),
        "median": statistics.median(lengths),
        "p90": pct(0.90),
        "p95": pct(0.95),
        "max": max(lengths),
    }


def try_load_qwen3_tokenizer():
    """token 長度統計是加分項；本機 transformers/huggingface_hub 版本若不相容則優雅跳過。"""
    try:
        from transformers import AutoTokenizer  # noqa: WPS433 (延遲載入)

        return AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
    except Exception as exc:  # noqa: BLE001 - 環境相容性問題不應中斷 Phase 1
        print(f"[stats] 略過 token 長度統計（transformers 載入失敗：{exc.__class__.__name__}: {exc}）")
        return None


def compute_split_stats(split: str, paragraphs: list[Paragraph], questions: list[Question], tokenizer=None) -> dict:
    n_articles = len({p.article_id for p in paragraphs})
    context_char_lens = [len(p.context) for p in paragraphs]
    question_char_lens = [len(q.question) for q in questions]
    answer_char_lens = [len(a) for q in questions for a in q.answers]

    stats = {
        "n_articles": n_articles,
        "n_paragraphs": len(paragraphs),
        "n_questions": len(questions),
        "expected_n_questions": EXPECTED_QUESTION_COUNTS.get(split),
        "matches_expected": len(questions) == EXPECTED_QUESTION_COUNTS.get(split),
        "context_char_length": _len_stats(context_char_lens),
        "question_char_length": _len_stats(question_char_lens),
        "answer_char_length": _len_stats(answer_char_lens),
    }

    if tokenizer is not None:
        # 用 system+user 模板粗估 prompt token 數（不含 assistant），抽樣 500 題避免太慢
        sample = questions if len(questions) <= 500 else random.sample(questions, 500)
        prompt_token_lens = []
        for q in sample:
            user_msg = f"文章：{q.context}\n\n問題：{q.question}"
            rendered = tokenizer.apply_chat_template(
                [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_msg}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            prompt_token_lens.append(len(tokenizer(rendered).input_ids))
        stats["prompt_token_length_sampled"] = _len_stats(prompt_token_lens)

    return stats


# --------------------------------------------------------------------------
# unanswerable 負例合成（PLAN.md §4.2）
# --------------------------------------------------------------------------

@dataclass
class SFTExample:
    qid: str
    split: str
    question: str
    context: str
    answer: str  # 正例=原文片段；負例=""
    answerable: bool
    is_synthetic_negative: bool = False
    neg_type: str | None = None  # "hard" | "easy" | None
    article_id: str = ""
    paragraph_id: str = ""
    gold_answers: list[str] = field(default_factory=list)  # 評估用：全部參考答案


def positive_examples(questions: list[Question]) -> list[SFTExample]:
    return [
        SFTExample(
            qid=q.qid,
            split=q.split,
            question=q.question,
            context=q.context,
            answer=q.answers[0],
            answerable=True,
            article_id=q.article_id,
            paragraph_id=q.paragraph_id,
            gold_answers=q.answers,
        )
        for q in questions
    ]


def synthesize_negatives(
    source_questions: list[Question],
    article_index: dict[str, list[Paragraph]],
    n_target: int,
    hard_frac: float,
    rng: random.Random,
    max_retries: int = 5,
) -> tuple[list[SFTExample], dict]:
    """從 source_questions 抽題，配上「答案不在裡面」的段落，產生負例。

    hard: 同文章不同段落（詞彙重疊但答案不在，DRCD 每篇文章平均 ~4 段可用）
    easy: 跨文章隨機段落（防止模型學到「低重疊=拒答」捷徑）
    假負例過濾：candidate 段落若恰好包含 gold answer 字串，重抽，重試用盡則跳過。
    """
    all_article_ids = list(article_index.keys())
    pool = source_questions.copy()
    rng.shuffle(pool)

    negatives: list[SFTExample] = []
    stats = {"requested": n_target, "hard": 0, "easy": 0, "false_negative_dropped": 0, "fallback_hard_to_easy": 0}

    for q in pool:
        if len(negatives) >= n_target:
            break
        gold = q.answers[0]
        want_hard = rng.random() < hard_frac
        candidate: Paragraph | None = None
        neg_type = "hard" if want_hard else "easy"

        same_article_paras = [p for p in article_index.get(q.article_id, []) if p.paragraph_id != q.paragraph_id]
        if want_hard and not same_article_paras:
            # 單段文章沒有其他段落可用，退化成 easy
            neg_type = "easy"
            stats["fallback_hard_to_easy"] += 1

        for _attempt in range(max_retries):
            if neg_type == "hard":
                candidate = rng.choice(same_article_paras)
            else:
                other_article_id = rng.choice(all_article_ids)
                tries = 0
                while other_article_id == q.article_id and tries < 5:
                    other_article_id = rng.choice(all_article_ids)
                    tries += 1
                candidate = rng.choice(article_index[other_article_id])

            if gold not in candidate.context:
                break  # 通過假負例過濾
            candidate = None  # 觸發重試

        if candidate is None:
            stats["false_negative_dropped"] += 1
            continue

        negatives.append(
            SFTExample(
                qid=f"{q.qid}-neg",
                split=q.split,
                question=q.question,
                context=candidate.context,
                answer="",
                answerable=False,
                is_synthetic_negative=True,
                neg_type=neg_type,
                article_id=candidate.article_id,
                paragraph_id=candidate.paragraph_id,
                gold_answers=[],
            )
        )
        stats[neg_type] += 1

    stats["generated"] = len(negatives)
    return negatives, stats


# --------------------------------------------------------------------------
# SFT chat 格式輸出
# --------------------------------------------------------------------------

def to_sft_record(ex: SFTExample, include_assistant: bool = True) -> dict:
    user_msg = f"文章：{ex.context}\n\n問題：{ex.question}"
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]
    if include_assistant:
        assistant_content = json.dumps({"answer": ex.answer, "answerable": ex.answerable}, ensure_ascii=False)
        messages.append({"role": "assistant", "content": assistant_content})

    return {
        "qid": ex.qid,
        "split": ex.split,
        "messages": messages,
        "answerable": ex.answerable,
        "gold_answers": ex.gold_answers,
        "is_synthetic_negative": ex.is_synthetic_negative,
        "neg_type": ex.neg_type,
        "article_id": ex.article_id,
        "paragraph_id": ex.paragraph_id,
    }


def write_jsonl(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[write] {path} ({len(records)} 筆)")


def write_sample_review(records: list[dict], path: Path, n: int, rng: random.Random) -> None:
    sample = rng.sample(records, min(n, len(records)))
    lines = ["# Phase 1 人工抽查樣本\n", f"隨機抽 {len(sample)} 筆（正負例混合），供人工複查。\n"]
    for i, r in enumerate(sample, 1):
        user_content = r["messages"][1]["content"]
        assistant_content = r["messages"][-1]["content"] if r["messages"][-1]["role"] == "assistant" else "(無 - eval 用)"
        context_preview = user_content.split("問題：")[0].replace("文章：", "").strip()
        context_preview = context_preview[:200] + ("…" if len(context_preview) > 200 else "")
        question_preview = user_content.split("問題：")[-1].strip()
        lines.append(f"## {i}. qid={r['qid']}  answerable={r['answerable']}  neg_type={r.get('neg_type')}\n")
        lines.append(f"- 文章片段：{context_preview}\n")
        lines.append(f"- 問題：{question_preview}\n")
        lines.append(f"- 標註輸出：`{assistant_content}`\n")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[write] {path} ({len(sample)} 筆抽查樣本)")


# --------------------------------------------------------------------------
# HF Hub 發佈（預設不執行，需 --push-to-hub）
# --------------------------------------------------------------------------

DATASET_CARD_TEMPLATE = """---
license: cc-by-sa-4.0
language:
- zh
task_categories:
- question-answering
tags:
- drcd
- extractive-qa
- zh-tw
- sft
- traditional-chinese
---

# {repo_id}

繁體中文抽取式閱讀理解 SFT 資料集，衍生自 DRCD（Delta Reading Comprehension Dataset）。

## 來源與授權（重要）

- 原始資料：[DRCD](https://github.com/DRCKnowledgeTeam/DRCD)（Delta Research Center / 台達電子），
  授權 **CC BY-SA 3.0**，內容改編自繁體中文維基百科。
- 論文引用：Shao et al., "DRCD: a Chinese Machine Reading Comprehension Dataset", arXiv:1806.00920.
- 本資料集是 DRCD 的 Adaptation（改編作品），依 CC BY-SA 授權鏈條，以 **CC BY-SA 4.0** 釋出。

## 所做的修改

1. 將原始 SQuAD 風格 JSON 重新格式化為 chat SFT 格式（system/user/assistant 三則訊息，assistant 輸出固定 JSON schema）
2. 從 train split 合成 unanswerable 負例（同文章跨段落 hard negative 為主、跨文章隨機 easy negative 為輔，
   並過濾「候選段落恰好包含正確答案」的假負例），answerable : unanswerable ≈ 3:1
3. train 抽樣至 {train_size} 筆用於微調；**dev 全部 {dev_size} 題原封不動保留為 hold-out**，
   額外提供從 dev 獨立合成的 unanswerable 負例（{dev_neg_size} 筆）供評估使用

## 檔案

- `train_sft.jsonl`：訓練用（含正負例）
- `dev_answerable.jsonl`：DRCD 官方 dev 全部題目，未經任何修改，含多參考答案
- `dev_unanswerable.jsonl`：從 dev 合成的評估用負例
- `few_shot_examples.json`：3 個固定 few-shot 範例（取自 train，2 正 1 負）

## Schema

```json
{{
  "qid": "...",
  "split": "train|dev",
  "messages": [{{"role": "system", "content": "..."}}, {{"role": "user", "content": "..."}}, {{"role": "assistant", "content": "{{\\"answer\\": \\"...\\", \\"answerable\\": true}}"}}],
  "answerable": true,
  "gold_answers": ["..."],
  "is_synthetic_negative": false,
  "neg_type": null,
  "article_id": "...",
  "paragraph_id": "..."
}}
```

完整專案（QLoRA 微調 -> GGUF 量化 -> Ollama/LM Studio 部署 -> 評估）：https://github.com/kuotunyu/local-llm-lifecycle
方法論與逐 Phase 實作紀錄見 PLAN.md，完整評估報告見 EVAL_REPORT.md。
"""


def push_to_hub(repo_id: str, stats: dict) -> None:
    from huggingface_hub import HfApi

    api = HfApi()
    print(f"[hub] 建立/確認 private dataset repo: {repo_id}")
    api.create_repo(repo_id=repo_id, repo_type="dataset", private=True, exist_ok=True)

    card = DATASET_CARD_TEMPLATE.format(
        repo_id=repo_id,
        train_size=stats["train_final_size"],
        dev_size=stats["dev_answerable_size"],
        dev_neg_size=stats["dev_unanswerable_size"],
    )
    card_path = PROCESSED_DIR / "README.md"
    card_path.write_text(card, encoding="utf-8")

    for fname in ["train_sft.jsonl", "dev_answerable.jsonl", "dev_unanswerable.jsonl", "few_shot_examples.json", "README.md"]:
        fpath = PROCESSED_DIR / fname
        print(f"[hub] 上傳 {fpath.name}")
        api.upload_file(
            path_or_fileobj=str(fpath),
            path_in_repo=fpath.name,
            repo_id=repo_id,
            repo_type="dataset",
        )
    print(f"[hub] 完成：https://huggingface.co/datasets/{repo_id}（private）")


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="DRCD -> SFT 資料集建構（Phase 1）")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-size", type=int, default=10000, help="train_sft.jsonl 目標總筆數（正+負）")
    parser.add_argument("--unanswerable-ratio", type=float, default=0.25, help="負例佔比，0.25 約為 3:1")
    parser.add_argument("--hard-negative-frac", type=float, default=0.75, help="負例中 hard（同文章跨段落）的比例")
    parser.add_argument("--max-negative-retries", type=int, default=5)
    parser.add_argument("--sample-review-n", type=int, default=20)
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--skip-token-stats", action="store_true", help="跳過 tokenizer 載入（環境無可用 transformers 時使用）")
    parser.add_argument("--push-to-hub", action="store_true")
    parser.add_argument("--repo-id", type=str, default=None, help="HF dataset repo id，例如 user/drcd-zhtw-extractive-qa-sft")
    args = parser.parse_args()

    if args.push_to_hub and not args.repo_id:
        parser.error("--push-to-hub 需要搭配 --repo-id")

    rng = random.Random(args.seed)
    random.seed(args.seed)

    # 1) 下載 + 解析
    paths = download_drcd(force=args.force_download)
    splits = {}
    for split in ["train", "dev", "test"]:
        paragraphs, questions = load_split(split, paths[split])
        splits[split] = (paragraphs, questions)
        print(f"[parse] {split}: {len(paragraphs)} 段落 / {len(questions)} 題")

    # 2) 統計
    tokenizer = None if args.skip_token_stats else try_load_qwen3_tokenizer()
    all_stats: dict = {"splits": {}}
    for split, (paragraphs, questions) in splits.items():
        all_stats["splits"][split] = compute_split_stats(split, paragraphs, questions, tokenizer=tokenizer)
        s = all_stats["splits"][split]
        flag = "OK" if s["matches_expected"] else "!! 與 PLAN.md 查證數字不符，請檢查來源是否變動"
        print(f"[stats] {split}: {s['n_questions']} 題（預期 {s['expected_n_questions']}）[{flag}]")

    leakage = check_train_dev_leakage(splits["train"][0], splits["dev"][0])
    all_stats["train_dev_leakage_check"] = leakage
    leak_flag = "OK（無段落文字重疊）" if leakage["clean"] else "!! 發現段落文字重疊，需調查"
    print(
        f"[leakage] train/dev 共用 {leakage['shared_article_count']} 篇文章（DRCD 官方段落級切分的既有特性）"
        f"，段落文字重疊 {leakage['leaked_paragraph_count']} 筆 [{leak_flag}]"
    )

    # 3) 負例合成 — train
    train_paragraphs, train_questions = splits["train"]
    train_article_index = build_article_index(train_paragraphs)

    n_unanswerable = round(args.train_size * args.unanswerable_ratio)
    n_answerable = args.train_size - n_unanswerable

    sampled_answerable_q = rng.sample(train_questions, min(n_answerable, len(train_questions)))
    answerable_examples = positive_examples(sampled_answerable_q)

    neg_source_pool = train_questions  # 負例來源題目可與 answerable 抽樣重疊，只是借用其 question 文字
    negatives, neg_stats = synthesize_negatives(
        neg_source_pool, train_article_index, n_unanswerable, args.hard_negative_frac, rng, args.max_negative_retries
    )
    print(f"[negatives:train] {neg_stats}")

    train_examples = answerable_examples + negatives
    rng.shuffle(train_examples)
    train_records = [to_sft_record(ex, include_assistant=True) for ex in train_examples]
    write_jsonl(train_records, PROCESSED_DIR / "train_sft.jsonl")

    # 4) dev：完整保留 + 獨立合成負例
    dev_paragraphs, dev_questions = splits["dev"]
    dev_article_index = build_article_index(dev_paragraphs)

    dev_answerable_examples = positive_examples(dev_questions)  # 全部 3524 題，未抽樣
    dev_answerable_records = [to_sft_record(ex, include_assistant=True) for ex in dev_answerable_examples]
    write_jsonl(dev_answerable_records, PROCESSED_DIR / "dev_answerable.jsonl")

    n_dev_unanswerable = round(len(dev_questions) * args.unanswerable_ratio / (1 - args.unanswerable_ratio))
    dev_negatives, dev_neg_stats = synthesize_negatives(
        dev_questions, dev_article_index, n_dev_unanswerable, args.hard_negative_frac, rng, args.max_negative_retries
    )
    print(f"[negatives:dev] {dev_neg_stats}")
    dev_unanswerable_records = [to_sft_record(ex, include_assistant=True) for ex in dev_negatives]
    write_jsonl(dev_unanswerable_records, PROCESSED_DIR / "dev_unanswerable.jsonl")

    # 5) few-shot 固定範例（取自 train，避免污染 dev；2 正 1 負，seed 固定可重現）
    fewshot_pos = rng.sample(answerable_examples, 2)
    fewshot_neg_candidates = [n for n in negatives if n.neg_type == "hard"] or negatives
    fewshot_neg = rng.sample(fewshot_neg_candidates, 1) if fewshot_neg_candidates else []
    fewshot_examples = [
        {"question": ex.question, "context": ex.context, "answer": ex.answer, "answerable": ex.answerable}
        for ex in (fewshot_pos + fewshot_neg)
    ]
    (PROCESSED_DIR / "few_shot_examples.json").write_text(
        json.dumps(fewshot_examples, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[write] {PROCESSED_DIR / 'few_shot_examples.json'} ({len(fewshot_examples)} 筆)")

    # 6) 人工抽查樣本
    write_sample_review(train_records, RESULTS_DIR / "sample_review.md", args.sample_review_n, rng)

    # 7) 統計報告落盤
    all_stats["train_negative_synthesis"] = neg_stats
    all_stats["dev_negative_synthesis"] = dev_neg_stats
    all_stats["train_final_size"] = len(train_records)
    all_stats["train_final_answerable"] = len(answerable_examples)
    all_stats["train_final_unanswerable"] = len(negatives)
    all_stats["dev_answerable_size"] = len(dev_answerable_records)
    all_stats["dev_unanswerable_size"] = len(dev_unanswerable_records)
    all_stats["args"] = vars(args)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stats_path = RESULTS_DIR / "data_stats.json"
    stats_path.write_text(json.dumps(all_stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[write] {stats_path}")

    print("\n===== 摘要 =====")
    print(f"train_sft.jsonl: {len(train_records)} 筆（answerable={len(answerable_examples)}, unanswerable={len(negatives)}）")
    print(f"dev_answerable.jsonl: {len(dev_answerable_records)} 筆（應為 {EXPECTED_QUESTION_COUNTS['dev']}）")
    print(f"dev_unanswerable.jsonl: {len(dev_unanswerable_records)} 筆")

    # 8) 可選：推到 HF private dataset
    if args.push_to_hub:
        push_to_hub(args.repo_id, all_stats)
    else:
        print("\n[hub] 未推送（加 --push-to-hub --repo-id <user>/<name> 以發佈為 HF private dataset）")


if __name__ == "__main__":
    main()
