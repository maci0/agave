#!/usr/bin/env python3
"""Unit tests for collect_continuations HTTP parsing (stdlib urllib, no network)."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
import urllib.error
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import collect_continuations as cc
from collect_continuations import collect_one, is_done, load_results, write_results


class _FakeResponse:
    def __init__(self, body: bytes) -> None:
        self._body = body

    def read(self) -> bytes:
        return self._body

    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, *args: object) -> None:
        return None


class CollectOneTests(unittest.TestCase):
    def test_parses_message_and_logprobs(self) -> None:
        payload = {
            "choices": [
                {
                    "message": {"content": "4"},
                    "logprobs": {"content": [{"token": "4"}]},
                }
            ]
        }
        fake = _FakeResponse(json.dumps(payload).encode())
        with patch("urllib.request.urlopen", return_value=fake):
            result = collect_one("http://example.test/v1", "m", "What is 2+2?", "k", 8)
        self.assertEqual(result["continuation"], "4")
        self.assertEqual(result["tokens_text"], ["4"])
        self.assertEqual(result["model"], "m")

    def test_http_error_becomes_runtime_error(self) -> None:
        err = urllib.error.HTTPError(
            url="http://example.test/v1",
            code=401,
            msg="Unauthorized",
            hdrs={},
            fp=BytesIO(b'{"error":"bad key"}'),
        )
        with patch("urllib.request.urlopen", side_effect=err), self.assertRaises(RuntimeError) as ctx:
            collect_one("http://example.test/v1", "m", "hi", "k", 8)
        self.assertIn("401", str(ctx.exception))


class ResumeTests(unittest.TestCase):
    def test_rerun_skips_prompts_already_collected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prompts = root / "prompts.txt"
            prompts.write_text("2+2?\ncapital of France?\n")
            out = root / "continuations.jsonl"

            calls: list[str] = []

            def fake(endpoint, model, prompt, key, max_tokens):
                calls.append(prompt)
                return {"prompt": prompt, "continuation": "x", "tokens_text": [], "model": model}

            argv = ["collect_continuations.py", "--endpoint", "http://example.test/v1",
                    "--model", "m", "--prompts", str(prompts), "--out", str(out),
                    "--api-key", "k", "--delay", "0"]
            with patch.object(sys, "argv", argv), patch.object(cc, "collect_one", fake):
                cc.main()
            self.assertEqual(calls, ["2+2?", "capital of France?"])
            first = out.read_text()

            # Second run: nothing new to bill, output unchanged.
            with patch.object(sys, "argv", argv), patch.object(cc, "collect_one", fake):
                cc.main()
            self.assertEqual(calls, ["2+2?", "capital of France?"])
            self.assertEqual(out.read_text(), first)
            self.assertEqual(len(out.read_text().strip().splitlines()), 2)

    def test_error_record_is_retried_and_replaced_not_duplicated(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out = root / "continuations.jsonl"
            write_results(out, {"q": {"prompt": "q", "model": "m", "error": "HTTP 500"}}, ["q"])
            self.assertFalse(is_done(load_results(out), "q", "m"))

            calls: list[str] = []

            def fake(endpoint, model, prompt, key, max_tokens):
                calls.append(prompt)
                return {"prompt": prompt, "continuation": "x", "tokens_text": [], "model": model}

            prompts = root / "prompts.txt"
            prompts.write_text("q\n")
            argv = ["collect_continuations.py", "--endpoint", "http://example.test/v1",
                    "--model", "m", "--prompts", str(prompts), "--out", str(out),
                    "--api-key", "k", "--delay", "0"]
            with patch.object(sys, "argv", argv), patch.object(cc, "collect_one", fake):
                cc.main()
            self.assertEqual(calls, ["q"])
            lines = out.read_text().strip().splitlines()
            self.assertEqual(len(lines), 1)
            self.assertIn("continuation", json.loads(lines[0]))


if __name__ == "__main__":
    unittest.main()
