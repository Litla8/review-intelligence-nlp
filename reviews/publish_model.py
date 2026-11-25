"""Upload the fine-tuned model folder to a Hugging Face model repo so the deployed app can download it.

Usage (after ``hf auth login`` or with HF_TOKEN set)::

    python -m reviews.publish_model --repo YOUR_HF_USERNAME/review-sentiment-distilbert
"""
from __future__ import annotations

import argparse
import logging
from dataclasses import replace
from pathlib import Path
from typing import Optional, Sequence

from reviews.config import ProjectConfig
from reviews.logging_utils import configure_logging

LOGGER = logging.getLogger(__name__)


def publish(model_dir: Path, repo_id: str, private: bool = False) -> str:
    """Create the repo if needed and upload ``model_dir``; returns the repo URL.

    Raises:
        FileNotFoundError: If ``model_dir`` is missing or holds no ``inference_config.json``.
        ImportError: If huggingface_hub is missing.
    """
    if not (model_dir / "inference_config.json").is_file():
        raise FileNotFoundError(f"{model_dir} is missing or incomplete. Run `python -m reviews.train_transformer` first.")
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise ImportError("pip install huggingface_hub") from exc
    api = HfApi()
    api.create_repo(repo_id, repo_type="model", private=private, exist_ok=True)
    api.upload_folder(folder_path=str(model_dir), repo_id=repo_id, repo_type="model",
                      ignore_patterns=["checkpoint-*", "*.pt", "runs/*"])
    return f"https://huggingface.co/{repo_id}"


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Upload the fine-tuned model to the Hugging Face Hub.")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--repo", required=True, help="USERNAME/repo-name")
    parser.add_argument("--private", action="store_true", help="private repo (the deployed app then needs HF_TOKEN)")
    args = parser.parse_args(argv)
    configure_logging("INFO")
    base = ProjectConfig()
    cfg = replace(base, paths=replace(base.paths, root=args.root))
    LOGGER.info("Uploaded to %s", publish(cfg.paths.models_path / cfg.transformer.model_dir, args.repo, args.private))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
