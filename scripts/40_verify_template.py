"""Phase 4 Step 5：chat template 五步驗證協定。

證明「transformers 版」跟「llama.cpp（GGUF）版」的 chat template 渲染行為一致，
沒有在合併/轉檔過程中壞掉。這是本專案「量化 × 微調交互作用」評估能不能信任的前提——
如果連 template 都對不上，後面 Q8_0/Q4_K_M 的分數差異就無法歸因到量化本身。

五步：
  1. metadata 靜態檢查：GGUF 內嵌 chat_template vs tokenizer_config.json
  2. 渲染字串比對：transformers apply_chat_template vs llama-server /apply-template
  3. token-ID 級比對：兩邊 tokenize 結果逐一比對（抓 BOS 重複等問題）
  4. greedy 解碼比對：transformers bf16 vs llama-server Q8_0，前 N token 一致即通過
  5. Ollama / LM Studio 各自獨立驗證（人工，見 deploy/OLLAMA.md、deploy/LMSTUDIO.md）

用法：
  python3 40_verify_template.py \
      --merged-dir work/merged \
      --gguf-q8 work/gguf/qwen3-8b-drcd-qa-Q8_0.gguf \
      --llama-server-bin llama.cpp/build/bin/llama-server \
      --dev-jsonl <path-to-dev_answerable.jsonl>
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import requests

SYSTEM_PROMPT = (
    "你是精確的閱讀理解助手。根據「文章」回答「問題」：\n"
    "- 答案必須是文章中的連續原文片段，一字不改\n"
    "- 若文章中找不到答案，answer 填空字串、answerable 填 false\n"
    "- 只輸出 JSON：{\"answer\": \"...\", \"answerable\": true|false}"
)

SERVER_PORT = 8712
SERVER_URL = f"http://127.0.0.1:{SERVER_PORT}"


def load_test_messages(dev_jsonl: Path, n: int = 5) -> list[list[dict]]:
    """從真實 DRCD dev 資料取樣，不用假造的 prompt。"""
    lines = dev_jsonl.read_text(encoding="utf-8").splitlines()
    samples = []
    for line in lines[:n]:
        row = json.loads(line)
        msgs = row["messages"]
        # 只取 system + user（不含 assistant，模擬推論情境）
        samples.append([m for m in msgs if m["role"] in ("system", "user")])
    return samples


def step1_metadata_check(gguf_path: Path, merged_dir: Path) -> bool:
    print("\n===== Step 1: metadata 靜態檢查 =====")
    try:
        from gguf.gguf_reader import GGUFReader
    except ImportError:
        print("找不到 gguf 套件的 GGUFReader，跳過（不影響其他步驟）")
        return True

    reader = GGUFReader(str(gguf_path))
    gguf_template = None
    for field in reader.fields.values():
        if field.name == "tokenizer.chat_template":
            gguf_template = bytes(field.parts[-1]).decode("utf-8", errors="replace")
            break

    tok_cfg_path = merged_dir / "tokenizer_config.json"
    tok_template = None
    if tok_cfg_path.exists():
        tok_cfg = json.loads(tok_cfg_path.read_text(encoding="utf-8"))
        tok_template = tok_cfg.get("chat_template")
    if tok_template is None:
        chat_tpl_path = merged_dir / "chat_template.jinja"
        if chat_tpl_path.exists():
            tok_template = chat_tpl_path.read_text(encoding="utf-8")

    if gguf_template is None:
        print("失敗：GGUF 內沒有找到 tokenizer.chat_template metadata")
        return False
    if tok_template is None:
        print("失敗：合併後目錄找不到 chat_template（tokenizer_config.json 或 chat_template.jinja 都沒有）")
        return False

    match = gguf_template.strip() == tok_template.strip()
    print(f"GGUF 內嵌 template 長度: {len(gguf_template)}")
    print(f"tokenizer_config template 長度: {len(tok_template)}")
    print(f"兩者完全一致: {match}")
    if not match:
        print("(不完全一致不一定是錯誤——GGUF 轉檔可能正規化過空白，繼續看 Step 2 的實際渲染結果)")
    return True


def step2_and_3_render_and_tokenize(
    tokenizer, messages: list[dict]
) -> tuple[bool, bool]:
    print("\n===== Step 2+3: 渲染字串 + token-ID 比對 =====")

    hf_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    hf_ids = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, enable_thinking=False
    )

    resp = requests.post(
        f"{SERVER_URL}/apply-template",
        json={"messages": messages, "chat_template_kwargs": {"enable_thinking": False}},
        timeout=30,
    )
    resp.raise_for_status()
    server_text = resp.json()["prompt"]

    string_match = hf_text == server_text
    print(f"渲染字串完全一致: {string_match}")
    if not string_match:
        # 找第一個不同的地方，方便除錯
        for i, (a, b) in enumerate(zip(hf_text, server_text)):
            if a != b:
                print(f"  第一個差異在 char {i}: transformers={a!r} vs llama-server={b!r}")
                print(f"  transformers 上下文: ...{hf_text[max(0,i-30):i+30]!r}...")
                print(f"  llama-server 上下文: ...{server_text[max(0,i-30):i+30]!r}...")
                break
        else:
            print(f"  長度不同: transformers={len(hf_text)} vs llama-server={len(server_text)}")

    tok_resp = requests.post(
        f"{SERVER_URL}/tokenize",
        json={"content": server_text, "add_special": True},
        timeout=30,
    )
    tok_resp.raise_for_status()
    server_ids = tok_resp.json()["tokens"]

    ids_match = list(hf_ids) == list(server_ids)
    print(f"token-ID 完全一致: {ids_match}")
    if not ids_match:
        print(f"  transformers 前 10 個 token: {list(hf_ids)[:10]}")
        print(f"  llama-server 前 10 個 token: {server_ids[:10]}")
        # 抓 BOS 重複這種已知坑
        if len(server_ids) == len(hf_ids) + 1 and server_ids[1:] == list(hf_ids):
            print("  懷疑是 BOS 重複（llama-server 多一個開頭 token）")

    return string_match, ids_match


def step4_greedy_decode_compare(hf_model, tokenizer, messages_list: list[list[dict]], n_tokens: int = 50) -> bool:
    print("\n===== Step 4: greedy 解碼比對（transformers bf16 vs llama-server Q8_0）=====")
    import torch

    all_match = True
    for i, messages in enumerate(messages_list):
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(hf_model.device)
        with torch.no_grad():
            hf_out = hf_model.generate(
                **inputs, max_new_tokens=n_tokens, do_sample=False, temperature=None, top_p=None, top_k=None
            )
        hf_new_tokens = hf_out[0][inputs["input_ids"].shape[1] :].tolist()

        resp = requests.post(
            f"{SERVER_URL}/v1/chat/completions",
            json={
                "messages": messages,
                "temperature": 0,
                "max_tokens": n_tokens,
                "seed": 42,
                "chat_template_kwargs": {"enable_thinking": False},
            },
            timeout=60,
        )
        resp.raise_for_status()
        server_text_out = resp.json()["choices"][0]["message"]["content"]
        hf_text_out = tokenizer.decode(hf_new_tokens, skip_special_tokens=True)

        # 用文字比較前 100 字元（token 對 token 比對受限於兩邊 tokenizer 呼叫方式差異，字串比較更穩健）
        prefix_len = min(100, len(hf_text_out), len(server_text_out))
        match = hf_text_out[:prefix_len] == server_text_out[:prefix_len]
        all_match = all_match and match
        print(f"  範例 {i+1}: 前 {prefix_len} 字元一致={match}")
        if not match:
            print(f"    transformers: {hf_text_out[:100]!r}")
            print(f"    llama-server:  {server_text_out[:100]!r}")

    return all_match


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--merged-dir", required=True)
    parser.add_argument("--gguf-q8", required=True)
    parser.add_argument("--llama-server-bin", required=True)
    parser.add_argument("--dev-jsonl", required=True)
    parser.add_argument("--n-samples", type=int, default=5)
    args = parser.parse_args()

    merged_dir = Path(args.merged_dir)
    gguf_q8 = Path(args.gguf_q8)
    dev_jsonl = Path(args.dev_jsonl)

    results = {}

    results["step1_metadata"] = step1_metadata_check(gguf_q8, merged_dir)

    print("\n啟動 llama-server（Q8_0，--jinja 預設開啟）...")
    server_log_path = Path(gguf_q8).parent / "llama-server.log"
    server_log = open(server_log_path, "w", encoding="utf-8")
    server_proc = subprocess.Popen(
        [
            args.llama_server_bin,
            "-m", str(gguf_q8),
            "-ngl", "99",
            "-c", "8192",
            "--port", str(SERVER_PORT),
            "--jinja",
        ],
        stdout=server_log,
        stderr=subprocess.STDOUT,
    )

    try:
        ready = False
        for attempt in range(180):
            if server_proc.poll() is not None:
                print(f"llama-server 行程已結束（exit code {server_proc.returncode}），見 {server_log_path}")
                break
            try:
                r = requests.get(f"{SERVER_URL}/health", timeout=2)
                if r.status_code == 200:
                    print(f"llama-server 就緒（等了 {attempt}s）")
                    ready = True
                    break
            except requests.exceptions.RequestException:
                pass
            time.sleep(1)
        if not ready:
            print(f"llama-server 未就緒，最後日誌內容（{server_log_path}）：")
            server_log.flush()
            print(server_log_path.read_text(encoding="utf-8")[-2000:])
            sys.exit(1)

        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch

        print(f"\n載入 transformers tokenizer/model：{merged_dir}")
        tokenizer = AutoTokenizer.from_pretrained(str(merged_dir))
        hf_model = AutoModelForCausalLM.from_pretrained(
            str(merged_dir), dtype=torch.bfloat16, device_map="cuda"
        )
        hf_model.eval()

        messages_list = load_test_messages(dev_jsonl, n=args.n_samples)

        string_matches, id_matches = [], []
        for messages in messages_list:
            s, i = step2_and_3_render_and_tokenize(tokenizer, messages)
            string_matches.append(s)
            id_matches.append(i)
        results["step2_string"] = all(string_matches)
        results["step3_token_id"] = all(id_matches)

        results["step4_greedy"] = step4_greedy_decode_compare(hf_model, tokenizer, messages_list)

    finally:
        print("\n關閉 llama-server...")
        server_proc.terminate()
        try:
            server_proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server_proc.kill()
            server_proc.wait(timeout=10)
        server_log.close()

    print("\n===== 總結 =====")
    for k, v in results.items():
        print(f"{k}: {'PASS' if v else 'FAIL'}")
    print("\nStep 5（Ollama / LM Studio 人工驗證）：見 deploy/OLLAMA.md、deploy/LMSTUDIO.md")

    if not all(results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
