---
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

# steven0226/drcd-zhtw-extractive-qa-sft

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
3. train 抽樣至 10000 筆用於微調；**dev 全部 3524 題原封不動保留為 hold-out**，
   額外提供從 dev 獨立合成的 unanswerable 負例（1175 筆）供評估使用

## 檔案

- `train_sft.jsonl`：訓練用（含正負例）
- `dev_answerable.jsonl`：DRCD 官方 dev 全部題目，未經任何修改，含多參考答案
- `dev_unanswerable.jsonl`：從 dev 合成的評估用負例
- `few_shot_examples.json`：3 個固定 few-shot 範例（取自 train，2 正 1 負）

## Schema

```json
{
  "qid": "...",
  "split": "train|dev",
  "messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}, {"role": "assistant", "content": "{\"answer\": \"...\", \"answerable\": true}"}],
  "answerable": true,
  "gold_answers": ["..."],
  "is_synthetic_negative": false,
  "neg_type": null,
  "article_id": "...",
  "paragraph_id": "..."
}
```

完整專案（QLoRA 微調 -> GGUF 量化 -> Ollama/LM Studio 部署 -> 評估）：
[github.com/kuotunyu/local-llm-lifecycle](https://github.com/kuotunyu/local-llm-lifecycle)

方法論總覽見 [README](https://github.com/kuotunyu/local-llm-lifecycle#readme)，完整評估報告見
[EVAL_REPORT.md](https://github.com/kuotunyu/local-llm-lifecycle/blob/main/EVAL_REPORT.md)。
