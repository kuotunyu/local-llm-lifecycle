# LM Studio 部署說明（Windows）

## 匯入 GGUF

LM Studio 認得的模型目錄結構是**兩層**：`<models 根目錄>/<publisher>/<model>/<file>.gguf`，
放錯層數會掃不到。

**方法一：CLI（推薦）**

```powershell
lms import "D:\models\qwen3-8b-drcd-qa\qwen3-8b-drcd-qa-Q4_K_M.gguf"
```

互動式流程會問 publisher/model 名稱，或用 `--user-repo <name>/<model>` 跳過互動直接指定。

**方法二：手動放置**

複製到 `C:\Users\<你>\.lmstudio\models\<自訂 publisher>\qwen3-8b-drcd-qa\qwen3-8b-drcd-qa-Q4_K_M.gguf`
（兩層資料夾都要有，缺一層 LM Studio 認不到）。

**不要**把 LM Studio 的模型搜尋路徑指到 `\\wsl$\...`——一樣會踩到 9P 橋接效能問題，GGUF 檔案要先
複製到 Windows 本機磁碟（我們已經放在 `D:\models\qwen3-8b-drcd-qa\`）。

## Template 處理

LM Studio 會直接執行 GGUF 內嵌的 jinja chat_template（用自己的 JS jinja 引擎），跟 Ollama 的「Go
template 模式比對」是完全不同的機制——**這代表 llama.cpp / Ollama 驗證通過，不保證 LM Studio 也過**，
一定要在這裡單獨測一次。

已知的 LM Studio jinja 解析器限制（跟 transformers 的正規 Jinja2 不完全相容）：

- 不支援數字用點號索引（`m.content.0`），要改成 `m.content[0]`
- 不支援負數索引（`arr[-1]`），要改成 `arr[x | last]`
- template 裡若有沒跳脫的雙引號（尤其是內嵌 JSON），可能解析失敗

如果載入時跳出「Failed to parse Jinja template」或「Error rendering prompt with jinja template」：
到 **My Models →（模型旁邊的齒輪圖示）→ Prompt Template**，貼上修正過的 template 覆蓋掉自動讀到的版本。

## 部署後必做的驗證

1. 載入模型，開啟 **Developer** 分頁的 log 視窗
2. 送出跟 `deploy/OLLAMA.md` 同一組測試用例：

   ```
   文章：自蒂爾西特和約簽訂於1807年後，沙皇亞歷山大一世在英國進攻丹麥後正式向英宣戰。

   問題：蒂爾西特和約是在哪一年簽署的?
   ```

3. 檢查兩件事：
   - Developer log 裡實際送進模型的 rendered prompt，跟 `scripts/40_verify_template.py` 驗證過的
     transformers 渲染結果是否一致
   - 回應是不是乾淨的 `{"answer": "1807年", "answerable": true}`，沒有 `<think>` 標籤外洩

## 建議設定

- **Temperature**：0（對應評估時的 greedy decoding，跟 Ollama/驗證腳本一致）
- **Context Length**：4096（DRCD 文章+問題通常遠小於這個長度）
- 如果模型的 EOS/停止字元判斷跑掉（一直生成不停），檢查 GGUF metadata 的 eog token 清單是否正確
  （`scripts/40_verify_template.py` Step 1 有做過這個檢查）
