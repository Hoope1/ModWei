import argparse
from pathlib import Path
from unittest import mock

import numpy as np

import edge_batch_runner as ebr


def test_to_line_inverts_and_thresholds():
    gray = np.array([[255, 255], [255, 255]], dtype=np.uint8)
    result = ebr.to_line(gray)
    assert result.max() == 0
    assert result.min() == 0


def test_parse_args_unknown(monkeypatch):
    with mock.patch.object(
        argparse.ArgumentParser,
        "parse_known_args",
        return_value=(argparse.Namespace(input=None, streamlit=False), ["--bad"]),
    ):
        with mock.patch.object(ebr.logger, "error") as log_err:
            with mock.patch.object(ebr.sys, "exit") as exit_mock:
                ebr.parse_args([])
                exit_mock.assert_called_with(40)
                log_err.assert_called()


def test_choose_dir_from_arg():
    path = Path("some_dir")
    assert ebr.choose_input_dir(path) == path


def test_clear_vram_calls_torch():
    with mock.patch("edge_batch_runner.torch") as torch_mock:
        torch_mock.cuda.is_available.return_value = True
        ebr.clear_vram()
        torch_mock.cuda.empty_cache.assert_called()


def test_main_runs(monkeypatch, tmp_path):
    calls = {}
    monkeypatch.setattr(ebr, "clone_repo", lambda repo: calls.setdefault("clone", True))
    monkeypatch.setattr(
        ebr, "download_weight", lambda url, dst: calls.setdefault("download", True)
    )
    monkeypatch.setattr(
        ebr, "process_images", lambda path: calls.setdefault("process", path)
    )
    result = ebr.main(["--input", str(tmp_path)])
    assert result == 0
    assert calls["process"] == tmp_path
