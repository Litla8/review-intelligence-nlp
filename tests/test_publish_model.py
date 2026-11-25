"""Tests for reviews.publish_model (the Hub is mocked)."""
from __future__ import annotations

from pathlib import Path
from unittest import mock

import pytest

from reviews import publish_model


def test_publish_requires_model(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        publish_model.publish(tmp_path, "u/m")


def test_publish_uploads(tmp_path: Path) -> None:
    (tmp_path / "inference_config.json").write_text("{}")
    hub = mock.MagicMock()
    with mock.patch.dict("sys.modules", {"huggingface_hub": hub}):
        url = publish_model.publish(tmp_path, "u/m")
    assert url == "https://huggingface.co/u/m"
    hub.HfApi.return_value.create_repo.assert_called_once()
    hub.HfApi.return_value.upload_folder.assert_called_once()
