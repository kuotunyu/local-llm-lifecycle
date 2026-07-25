# Ollama 部署說明

## 安裝與匯入

Ollama 建議裝在 **Windows 原生**（不是 WSL2 內）——GGUF 檔案放在 Windows 檔案系統，Ollama 跟 LM Studio 可以共用同一份，也不會有 WSL2↔Windows 之間 9P 檔案系統橋接的效能損耗（PLAN.md §4「GGUF 轉檔鏈」）。

1. 確認 Ollama 服務有跑起來（`ollama serve`，或桌面版會自動啟動背景服務）
2. GGUF 檔案放在 `D:\models\qwen3-8b-drcd-qa\`（Q8_0 跟 Q4_K_M 兩個版本都放，Q4_K_M 是實際部署用的）
3. `deploy/Modelfile` 的 `FROM` 指向 Q4_K_M（部署用量化版；Q8_0 是驗證/評估用，不需要另外匯入 Ollama）
4. 在專案根目錄執行：

   ```powershell
   ollama create qwen3-8b-drcd-qa -f deploy\Modelfile
   ```

## 已知坑：Ollama 不會直接執行 GGUF 內嵌的 jinja template

Ollama 的 Modelfile `TEMPLATE` 是 Go text/template 語法，不是 jinja；`ollama create` 讀到 GGUF 裡的
jinja chat_template 時，是用「跟已知樣板比對」的方式猜要套用哪個 Go template，不是真的執行 jinja。
自訂/微調過的 template 比對失敗時，可能會**靜默退化成裸 passthrough**（模型看到沒有角色標記的純文字）。

**這份 Modelfile 刻意沒有手寫 TEMPLATE**：Ollama 新版引擎對 Qwen3 這個架構有專用的內建 Go
renderer/parser（處理 `<think>` 標籤解析），就算手寫 TEMPLATE 也可能被忽略（[ollama/ollama#14560](https://github.com/ollama/ollama/issues/14560)）。
所以策略是讓 Ollama 自動偵測，**但一定要實測驗證**，不能假設它是對的。

## 部署後必做的驗證

```powershell
# 1. 確認 Ollama 實際套用的 template（跟 GGUF 內嵌的、跟 transformers 渲染的比對）
ollama show qwen3-8b-drcd-qa --template

# 2. 開 debug log，看真正送進模型的 prompt 長怎樣
$env:OLLAMA_DEBUG=1
ollama serve  # 另開一個視窗跑，觀察 log

# 3. 實際對話測試，注意兩件事：
#    (a) 輸出是不是乾淨的 JSON，沒有多餘文字
#    (b) 有沒有 <think> 標籤外洩（本模型訓練時是 non-thinking 格式，不該看到 <think>...</think> 內容）
ollama run qwen3-8b-drcd-qa --think=false
```

測試用例（貼到互動對話裡）：

```
文章：自蒂爾西特和約簽訂於1807年後，沙皇亞歷山大一世在英國進攻丹麥後正式向英宣戰。

問題：蒂爾西特和約是在哪一年簽署的?
```

預期輸出：`{"answer": "1807年", "answerable": true}`，不含任何 `<think>` 內容或其他文字。

## 如果 template 驗證失敗

如果 Step 2/3 發現渲染出來的 prompt 跟 transformers 版不一致（例如角色標記消失、`<think>` 沒被正確關掉），
改成手動在 Modelfile 裡加 `TEMPLATE` 區塊，內容從 `scripts/40_verify_template.py` Step 2 驗證通過的
transformers 渲染字串反推對應的 Go template 語法，重新 `ollama create` 覆蓋。

## 效能備註

- `num_ctx 4096` 是 Modelfile 裡設定的預設 context 長度，DRCD 文章 + 問題通常遠小於這個長度（Phase 1
  統計 p95 context 約 718 字），有餘裕
- `temperature 0` 對應評估用的 greedy decoding 設定，跟 `scripts/40_verify_template.py`、
  `scripts/50_eval_qa.py` 保持一致，方便結果互相比對
