"""Phase 6：HF 正式發佈——LoRA adapter repo + GGUF repo 轉正（public）、SFT dataset 轉 public。

跟訓練期的 private checkpoint repo（`qwen3-8b-drcd-qa-ckpt`）是分開的：這支腳本建立/更新的是
給外部使用者看的正式 repo，命名跟授權見 PLAN.md §3「HF 資產命名」。

三個目標：
  1. LoRA adapter repo（public, Apache-2.0）：從本機 adapter 目錄上傳，寫完整 model card
  2. GGUF repo（public, Apache-2.0）：上傳 Q8_0（近無損參考）+ Q4_K_M（部署建議），寫完整 model card
  3. SFT dataset repo：訓練期 private → 轉 public（CC BY-SA 4.0，卡片沿用 Phase 1 已寫好的版本）

用法：
  python3 60_publish_hf.py \
      --adapter-dir work/adapter \
      --gguf-q8 work/gguf/qwen3-8b-drcd-qa-Q8_0.gguf \
      --gguf-q4 work/gguf/qwen3-8b-drcd-qa-Q4_K_M.gguf \
      --lora-repo steven0226/Qwen3-8B-DRCD-zhTW-QA-LoRA \
      --gguf-repo steven0226/Qwen3-8B-DRCD-zhTW-QA-GGUF \
      --dataset-repo steven0226/drcd-zhtw-extractive-qa-sft
"""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import HfApi

BASE_MODEL = "unsloth/Qwen3-8B"
OFFICIAL_BASE_MODEL = "Qwen/Qwen3-8B"
LLAMA_CPP_COMMIT = "a5822222909b785f23ddc74ce3c8f85bd0e38562"

SYSTEM_PROMPT = (
    "你是精確的閱讀理解助手。根據「文章」回答「問題」：\n"
    "- 答案必須是文章中的連續原文片段，一字不改\n"
    "- 若文章中找不到答案，answer 填空字串、answerable 填 false\n"
    "- 只輸出 JSON：{\"answer\": \"...\", \"answerable\": true|false}"
)

# 五組對照結果（results/eval_summary.json，2026-07-17 全量 4699 題）+ TMMLU+ forgetting
# （results/tmmlu_summary.json）——數字已在 EVAL_REPORT.md 定稿，這裡直接引用，不在發佈時
# 重新讀檔計算（避免發佈腳本依賴當時可能已經清掉的本機評估暫存檔）。
RESULTS_TABLE = """\
| # | 組別 | overall EM | overall F1 | JSON 合法率 |
|---|------|-----------:|-----------:|-------------:|
| 1 | base zero-shot（原廠） | 0.4756 | 0.6858 | 95.62% |
| 2 | base few-shot（3-shot） | 0.8253 | 0.9191 | 99.98% |
| 3 | **本 adapter（未量化 bf16）** | **0.9325** | **0.9704** | 100% |
| 4 | 本 GGUF（Q8_0） | 0.9328 | 0.9706 | 100% |
| 5 | 本 GGUF（Q4_K_M，部署建議） | 0.9330 | 0.9700 | 100% |

n=4,699（DRCD 官方 dev split 完整題目，3,524 可回答 + 1,175 unanswerable）。
量化幾乎沒有吃掉微調效果（組3→組5 EM 幾乎沒有下降）。詳細方法論、TMMLU+ forgetting
check（macro accuracy 掉 12.25 個百分點）、錯誤案例分析見專案 EVAL_REPORT.md。\
"""

FORGETTING_NOTE = """\
> **已知限制**：這個 checkpoint 在 DRCD 抽取式 QA 上表現接近滿分，但 TMMLU+（通用知識選擇題）
> macro accuracy 從原廠的 0.630 掉到 0.508（−12.25 個百分點），確認存在明顯的
> catastrophic forgetting。如果需要保留通用能力，建議：(a) 只在需要精確抽取式 QA 的場景使用
> 這個 adapter，(b) 或參考本專案方法論自行用較低 epoch / 加入通用資料混合訓練。\
"""

LORA_CARD = """\
---
license: apache-2.0
base_model: {base_model}
language:
- zh
tags:
- qlora
- lora
- peft
- question-answering
- extractive-qa
- traditional-chinese
- drcd
- qwen3
---

# Qwen3-8B-DRCD-zhTW-QA-LoRA

繁體中文抽取式閱讀理解（extractive QA）QLoRA adapter，微調自 `{base_model}`，訓練資料為
[DRCD](https://github.com/DRCKnowledgeTeam/DRCD)（改編版，見下方 SFT dataset）。

輸出固定 JSON schema：`{{"answer": "文中連續原文片段", "answerable": true|false}}`。

{forgetting_note}

## 評估結果（DRCD dev，完整 4,699 題）

{results_table}

## 訓練細節

- 基底模型：`{base_model}`（LoRA 實際合併/推論用的起點權重；跟官方
  [`{official_base_model}`](https://huggingface.co/{official_base_model}) 同源，
  unsloth 版本額外做了 tokenizer 修正）
- 方法：QLoRA（4-bit 訓練），Unsloth，r=16 / alpha=16 / dropout=0，target modules 涵蓋
  q/k/v/o/gate/up/down 全線性層
- 資料：9,800 筆訓練（DRCD train split 抽樣，另留 200 筆 eval holdout），2 epochs，
  effective batch 16，lr 2e-4 cosine schedule
- 硬體：Google Colab Pro，NVIDIA L4
- 訓練曲線：[W&B run](https://wandb.ai/tunyu1/qwen3-drcd-qlora/runs/e8h2wq6x)
  （train loss 收斂至 0.005，eval loss 穩定在 0.017-0.018，無過擬合反彈）
- Chat template：Qwen3 hybrid thinking 架構，訓練與推論**一律** `enable_thinking=False`
  （非思考模式）

## 用法（transformers + PEFT）

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

base = AutoModelForCausalLM.from_pretrained(
    "{base_model}", dtype=torch.bfloat16, device_map="cuda"
)
model = PeftModel.from_pretrained(base, "{lora_repo}")
tokenizer = AutoTokenizer.from_pretrained("{lora_repo}")

messages = [
    {{"role": "system", "content": {system_prompt!r}}},
    {{"role": "user", "content": "文章：...\\n\\n問題：..."}},
]
prompt = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
)
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
out = model.generate(**inputs, max_new_tokens=128, do_sample=False)
print(tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))
```

量化部署版（GGUF，含 llama.cpp / Ollama / LM Studio 用法）：
[{gguf_repo}](https://huggingface.co/{gguf_repo})

## 資料授權與歸屬

- 訓練資料衍生自 [DRCD](https://github.com/DRCKnowledgeTeam/DRCD)（Delta Research
  Center／台達電子），原始授權 **CC BY-SA 3.0**，內容改編自繁體中文維基百科；論文引用
  Shao et al., *"DRCD: a Chinese Machine Reading Comprehension Dataset"*,
  [arXiv:1806.00920](https://arxiv.org/abs/1806.00920)
- 本 adapter 權重以 **Apache-2.0** 釋出（LoRA 權重本身不含 DRCD 原文，業界通說
  ShareAlike 不傳染到權重）；完整 SFT 資料集（CC BY-SA 4.0，含改編說明）：
  [{dataset_repo}](https://huggingface.co/datasets/{dataset_repo})
- 基底模型 `{base_model}` 授權 Apache-2.0，歸屬 Qwen team / unsloth

完整專案（Colab QLoRA → 本機合併/量化 → Ollama/LM Studio 部署 → 五組評估）：見對應
GitHub repo（README/DESIGN 定稿）。
"""

GGUF_CARD = """\
---
license: apache-2.0
base_model: {base_model}
language:
- zh
tags:
- gguf
- llama.cpp
- ollama
- lm-studio
- question-answering
- extractive-qa
- traditional-chinese
- drcd
- qwen3
---

# Qwen3-8B-DRCD-zhTW-QA-GGUF

`{lora_repo}` 的量化部署版，繁體中文抽取式閱讀理解（extractive QA）。輸出固定 JSON schema：
`{{"answer": "文中連續原文片段", "answerable": true|false}}`。

{forgetting_note}

## 檔案

| 檔名 | 量化 | 大小 | 用途 |
|------|------|-----:|------|
| `{q8_filename}` | Q8_0 | ~8.7GB | 近無損參考／驗證用 |
| `{q4_filename}` | Q4_K_M | ~5.0GB | **建議部署版**（4090 本機評估確認量化幾乎零損耗） |

轉檔鏈：`{base_model}` + LoRA → bf16 merge → f16 GGUF
（`convert_hf_to_gguf.py`）→ `llama-quantize`。llama.cpp build commit：`{llama_cpp_commit}`。

## 評估結果（DRCD dev，完整 4,699 題）

{results_table}

## 用法

### llama.cpp

```bash
llama-server -m {q4_filename} -ngl 99 -c 4096 --jinja
# --jinja 讓 llama-server 直接執行 GGUF 內嵌的原生 chat_template（非思考模式）
```

呼叫 OpenAI 相容 API 時記得帶 `"chat_template_kwargs": {{"enable_thinking": false}}`
（本模型訓練時一律非思考模式；GGUF 內嵌 template 在 `enable_thinking=false` 時會自動插入
空 `<think>` block，不需要另外處理）。

### Ollama

```bash
ollama create qwen3-8b-drcd-qa -f Modelfile
ollama run qwen3-8b-drcd-qa --think=false
```

Modelfile 範例：

```
FROM {q4_filename}
PARAMETER temperature 0
PARAMETER num_ctx 4096
SYSTEM \"\"\"{system_prompt}\"\"\"
```

實測：Ollama 新版引擎會直接讀取並執行 GGUF 內嵌的原生 jinja template（不是退化成內建 Go
template 比對），不需要手寫 TEMPLATE。務必用 `ollama show <model> --template` 確認實際套用的
模板，並用 `--think=false` 關閉思考模式。

### LM Studio

```bash
lms import {q4_filename} --copy
```

匯入後於 Developer 分頁確認渲染的 prompt 沒有 `<think>` 外洩；建議設定 temperature=0、
context length=4096。

## 資料授權與歸屬

- 訓練資料衍生自 [DRCD](https://github.com/DRCKnowledgeTeam/DRCD)（Delta Research
  Center／台達電子），原始授權 **CC BY-SA 3.0**；論文引用 Shao et al.,
  *"DRCD: a Chinese Machine Reading Comprehension Dataset"*,
  [arXiv:1806.00920](https://arxiv.org/abs/1806.00920)
- 模型權重（含本 GGUF）以 **Apache-2.0** 釋出；完整 SFT 資料集（CC BY-SA 4.0）：
  [{dataset_repo}](https://huggingface.co/datasets/{dataset_repo})
- 基底模型 `{base_model}` 授權 Apache-2.0，歸屬 Qwen team / unsloth
- 未量化 LoRA adapter：[{lora_repo}](https://huggingface.co/{lora_repo})

完整專案（Colab QLoRA → 本機合併/量化 → Ollama/LM Studio 部署 → 五組評估）：見對應
GitHub repo（README/DESIGN 定稿）。
"""


def publish_lora(api: HfApi, repo_id: str, adapter_dir: Path, gguf_repo: str, dataset_repo: str) -> None:
    print(f"建立/確認 LoRA repo：{repo_id}（public）")
    api.create_repo(repo_id=repo_id, repo_type="model", private=False, exist_ok=True)
    card = LORA_CARD.format(
        base_model=BASE_MODEL,
        official_base_model=OFFICIAL_BASE_MODEL,
        lora_repo=repo_id,
        gguf_repo=gguf_repo,
        dataset_repo=dataset_repo,
        results_table=RESULTS_TABLE,
        forgetting_note=FORGETTING_NOTE,
        system_prompt=SYSTEM_PROMPT,
    )
    (adapter_dir / "README.md").write_text(card, encoding="utf-8")
    print(f"上傳 adapter 檔案：{adapter_dir} -> {repo_id}")
    api.upload_folder(
        repo_id=repo_id,
        repo_type="model",
        folder_path=str(adapter_dir),
        ignore_patterns=[".cache/*", ".gitattributes"],
    )
    print(f"LoRA repo 發佈完成：https://huggingface.co/{repo_id}")


def publish_gguf(api: HfApi, repo_id: str, gguf_q8: Path, gguf_q4: Path, lora_repo: str, dataset_repo: str) -> None:
    print(f"建立/確認 GGUF repo：{repo_id}（public）")
    api.create_repo(repo_id=repo_id, repo_type="model", private=False, exist_ok=True)
    card = GGUF_CARD.format(
        base_model=BASE_MODEL,
        lora_repo=lora_repo,
        dataset_repo=dataset_repo,
        q8_filename=gguf_q8.name,
        q4_filename=gguf_q4.name,
        llama_cpp_commit=LLAMA_CPP_COMMIT,
        results_table=RESULTS_TABLE,
        forgetting_note=FORGETTING_NOTE,
        system_prompt=SYSTEM_PROMPT,
    )
    card_path = gguf_q8.parent / "README.md"
    card_path.write_text(card, encoding="utf-8")
    api.upload_file(
        path_or_fileobj=str(card_path), path_in_repo="README.md", repo_id=repo_id, repo_type="model"
    )
    for gguf_path in (gguf_q8, gguf_q4):
        print(f"上傳 {gguf_path.name}（{gguf_path.stat().st_size / 1e9:.1f}GB）...")
        api.upload_file(
            path_or_fileobj=str(gguf_path),
            path_in_repo=gguf_path.name,
            repo_id=repo_id,
            repo_type="model",
        )
    print(f"GGUF repo 發佈完成：https://huggingface.co/{repo_id}")


def publish_dataset(api: HfApi, repo_id: str) -> None:
    print(f"將 dataset repo 轉為 public：{repo_id}")
    api.update_repo_visibility(repo_id=repo_id, repo_type="dataset", private=False)
    print(f"dataset repo 已轉 public：https://huggingface.co/datasets/{repo_id}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter-dir", required=True)
    parser.add_argument("--gguf-q8", required=True)
    parser.add_argument("--gguf-q4", required=True)
    parser.add_argument("--lora-repo", required=True)
    parser.add_argument("--gguf-repo", required=True)
    parser.add_argument("--dataset-repo", required=True)
    parser.add_argument("--skip-lora", action="store_true")
    parser.add_argument("--skip-gguf", action="store_true")
    parser.add_argument("--skip-dataset", action="store_true")
    args = parser.parse_args()

    api = HfApi()

    if not args.skip_lora:
        publish_lora(api, args.lora_repo, Path(args.adapter_dir), args.gguf_repo, args.dataset_repo)
    if not args.skip_gguf:
        publish_gguf(api, args.gguf_repo, Path(args.gguf_q8), Path(args.gguf_q4), args.lora_repo, args.dataset_repo)
    if not args.skip_dataset:
        publish_dataset(api, args.dataset_repo)

    print("\n===== 發佈完成 =====")
    print(f"LoRA:    https://huggingface.co/{args.lora_repo}")
    print(f"GGUF:    https://huggingface.co/{args.gguf_repo}")
    print(f"Dataset: https://huggingface.co/datasets/{args.dataset_repo}")


if __name__ == "__main__":
    main()
