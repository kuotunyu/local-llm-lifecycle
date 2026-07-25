#!/usr/bin/env bash
# Phase 4 Step 4: f16 GGUF -> Q8_0（近無損參照）+ Q4_K_M（部署用）
#
# 用法：./31_quantize.sh <f16_gguf_path> <llama_cpp_dir> <out_dir>
#
# Q8_0 是評估用的「轉檔損耗探針」（PLAN.md §4.5 組4）：如果 Q8_0 的推論結果就跟
# transformers 版對不上，代表問題出在轉檔/template，不是量化本身。Q4_K_M 才是實際
# 要部署到 Ollama/LM Studio 的版本。

set -euo pipefail

F16_GGUF="${1:?用法: 31_quantize.sh <f16_gguf_path> <llama_cpp_dir> <out_dir>}"
LLAMA_CPP_DIR="${2:?需要 llama.cpp repo 路徑}"
OUT_DIR="${3:?需要輸出資料夾}"

QUANTIZE_BIN="$LLAMA_CPP_DIR/build/bin/llama-quantize"
BASENAME="$(basename "$F16_GGUF" .gguf)"
BASENAME="${BASENAME%-f16}"

mkdir -p "$OUT_DIR"

echo "量化 Q8_0..."
"$QUANTIZE_BIN" "$F16_GGUF" "$OUT_DIR/${BASENAME}-Q8_0.gguf" Q8_0

echo "量化 Q4_K_M..."
"$QUANTIZE_BIN" "$F16_GGUF" "$OUT_DIR/${BASENAME}-Q4_K_M.gguf" Q4_K_M

echo "完成："
du -h "$OUT_DIR/${BASENAME}-Q8_0.gguf" "$OUT_DIR/${BASENAME}-Q4_K_M.gguf"
