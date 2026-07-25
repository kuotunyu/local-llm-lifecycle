#!/usr/bin/env bash
# Phase 4 Step 3: 合併後模型 -> GGUF f16 中間格式
#
# 用法：./30_convert_gguf.sh <merged_model_dir> <llama_cpp_dir> <out_gguf_path>
#
# f16 是轉檔中間格式（不是最終部署格式），31_quantize.sh 會從這個檔案再量化出
# Q8_0 / Q4_K_M。convert_hf_to_gguf.py 會讀 tokenizer_config.json 的 chat_template
# 並嵌入 GGUF metadata，所以 20_merge_lora.py 搬運 tokenizer 的步驟一定要先做對。

set -euo pipefail

MERGED_DIR="${1:?用法: 30_convert_gguf.sh <merged_model_dir> <llama_cpp_dir> <out_gguf_path>}"
LLAMA_CPP_DIR="${2:?需要 llama.cpp repo 路徑}"
OUT_GGUF="${3:?需要輸出的 .gguf 路徑}"

VENV_PYTHON="$(dirname "$LLAMA_CPP_DIR")/.venv/bin/python3"

echo "轉換 $MERGED_DIR -> $OUT_GGUF (f16)"
"$VENV_PYTHON" "$LLAMA_CPP_DIR/convert_hf_to_gguf.py" \
    "$MERGED_DIR" \
    --outtype f16 \
    --outfile "$OUT_GGUF"

echo "轉換完成：$(du -h "$OUT_GGUF" | cut -f1)"
