#!/usr/bin/env python3
"""Unit tests for collect_continuations HTTP parsing (stdlib urllib, no network)."""

from __future__ import annotations

import json
import sys
import unittest
import urllib.error
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from collect_continuations import collect_one


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
        with patch("urllib.request.urlopen", side_effect=err):
            with self.assertRaises(RuntimeError) as ctx:
                collect_one("http://example.test/v1", "m", "hi", "k", 8)
        self.assertIn("401", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
