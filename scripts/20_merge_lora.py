"""Phase 4 Step 2: 合併 LoRA adapter 回 base model（bf16，非量化狀態）。

已知坑（PLAN.md §4「本機合併/轉檔」）：
1. 絕不合併進 4-bit base（會毀品質，huggingface/peft#2105、huggingface/transformers#31293）。
   PEFT 訓練時記錄的 base_model_name_or_path 是 unsloth/Qwen3-8B-unsloth-bnb-4bit（4-bit 版），
   這裡改用它背後對應的**未量化版本** unsloth/Qwen3-8B（同團隊、同樣的 tokenizer 修正，只是沒有
   量化），避免載入 4-bit repo 再"解量化"造成的精度損失。
2. tokenizer 一定要從 adapter 輸出目錄搬（訓練時可能異動 chat_template/special tokens），
   不能從 base model 存的那份。
3. chat_template 若同時存在於 tokenizer_config.json（內嵌）與 chat_template.jinja（獨立檔），
   convert_hf_to_gguf.py 會因為重複 key 直接 crash（ggml-org/llama.cpp#7923, unslothai/unsloth#791）。
   這裡偵測到就自動移除 tokenizer_config.json 內嵌的那份，只保留獨立檔案。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# 對應 adapter 訓練時 4-bit base（unsloth/Qwen3-8B-unsloth-bnb-4bit）的未量化版本
MERGE_BASE_MODEL = "unsloth/Qwen3-8B"


def fix_duplicate_chat_template(out_dir: Path) -> None:
    tok_cfg_path = out_dir / "tokenizer_config.json"
    chat_tpl_path = out_dir / "chat_template.jinja"

    if not tok_cfg_path.exists():
        return

    tok_cfg = json.loads(tok_cfg_path.read_text(encoding="utf-8"))
    has_inline = "chat_template" in tok_cfg
    has_separate = chat_tpl_path.exists()

    print(f"tokenizer_config.json 內嵌 chat_template: {has_inline}")
    print(f"獨立 chat_template.jinja 存在: {has_separate}")

    if has_inline and has_separate:
        print("偵測到重複 key（已知坑 #3）——移除 tokenizer_config.json 內嵌版本，只保留獨立檔案")
        del tok_cfg["chat_template"]
        tok_cfg_path.write_text(json.dumps(tok_cfg, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter-dir", required=True, help="10_download_adapter.py 下載好的本地路徑")
    parser.add_argument("--out-dir", required=True, help="合併後模型輸出路徑")
    args = parser.parse_args()

    adapter_dir = Path(args.adapter_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"載入 base model（bf16，非量化）：{MERGE_BASE_MODEL}")
    base_model = AutoModelForCausalLM.from_pretrained(
        MERGE_BASE_MODEL,
        dtype=torch.bfloat16,
        device_map="cuda",
    )

    print(f"套用 LoRA adapter：{adapter_dir}")
    model = PeftModel.from_pretrained(base_model, str(adapter_dir))

    print("合併中（merge_and_unload）...")
    model = model.merge_and_unload()

    print(f"儲存合併後模型至：{out_dir}")
    model.save_pretrained(out_dir, safe_serialization=True)
    del model, base_model
    torch.cuda.empty_cache()

    print("搬運 tokenizer（從 adapter 目錄，確保跟訓練時一致）...")
    tokenizer = AutoTokenizer.from_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(out_dir)

    fix_duplicate_chat_template(out_dir)

    files = sorted(p.name for p in out_dir.iterdir() if p.is_file())
    print(f"合併完成，輸出 {len(files)} 個檔案：{files}")


if __name__ == "__main__":
    main()
