"""Phase 4 Step 1: 從 HF Hub 下載 Phase 3 訓練完成的 LoRA adapter。

排除 last-checkpoint/（訓練用的 optimizer/scheduler/rng 狀態，合併不需要）與
.smoke_test（Phase 2 pre-flight 煙霧測試留下的測試檔）。
"""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default="steven0226/qwen3-8b-drcd-qa-ckpt")
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"下載 adapter：{args.repo_id} -> {out_dir}")
    snapshot_download(
        repo_id=args.repo_id,
        local_dir=str(out_dir),
        ignore_patterns=["last-checkpoint/*", ".smoke_test"],
    )

    files = sorted(p.name for p in out_dir.iterdir() if p.is_file())
    print(f"下載完成，{len(files)} 個檔案：{files}")


if __name__ == "__main__":
    main()
