"""Offline HTTP contract tests; no real inference service or model is contacted."""

import base64
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

from image_reader.__main__ import main
from image_reader.client import Conversation, ImageReaderError, OllamaClient, OCR_PROMPT, encode_images

PNG = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAAAAAA6fptVAAAACklEQVR4nGP4DwABAQEAsTj2FAAAAABJRU5ErkJggg==")


@contextmanager
def server(reply=None, status=200):
    requests = []
    response = reply if reply is not None else {"message": {"role": "assistant", "content": "# Example\nReadable text"}, "done": True, "done_reason": "stop"}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            requests.append({"path": self.path, "body": json.loads(self.rfile.read(int(self.headers["Content-Length"]))), "content_type": self.headers["Content-Type"]})
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            if status == 302:
                self.send_header("Location", "/redirected")
            self.end_headers()
            self.wfile.write(response if isinstance(response, bytes) else json.dumps(response).encode())

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{httpd.server_port}", requests
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join()


class ReaderTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.image = Path(self.temp.name) / "synthetic.png"
        self.image.write_bytes(PNG)

    def test_image_encoding_uses_bytes_not_path(self):
        self.assertEqual(encode_images([self.image]), [base64.b64encode(PNG).decode()])

    def test_multiple_images_keep_order(self):
        second = Path(self.temp.name) / "other.jpg"
        second.write_bytes(b"\xff\xd8\xff\x00")
        self.assertEqual([base64.b64decode(x) for x in encode_images([self.image, second])], [PNG, second.read_bytes()])

    def test_bad_image_signature_fails(self):
        self.image.write_text("not an image")
        with self.assertRaises(ImageReaderError):
            encode_images([self.image])

    def test_missing_image_fails(self):
        with self.assertRaises(ImageReaderError):
            encode_images([Path(self.temp.name) / "missing.png"])

    def test_empty_image_fails(self):
        self.image.write_bytes(b"")
        with self.assertRaises(ImageReaderError):
            encode_images([self.image])

    def test_image_size_and_count_limits(self):
        with patch("image_reader.client.MAX_IMAGE_BYTES", len(PNG) - 1):
            with self.assertRaises(ImageReaderError):
                encode_images([self.image])
        with patch("image_reader.client.MAX_TOTAL_IMAGE_BYTES", len(PNG)):
            with self.assertRaises(ImageReaderError):
                encode_images([self.image, self.image])
        with self.assertRaises(ImageReaderError):
            encode_images([self.image] * 11)

    def test_real_http_chat_contract_and_history(self):
        with server() as (endpoint, requests):
            session = Conversation(OllamaClient(endpoint), "Answer carefully.")
            answer = session.ask("What is written?", [self.image])
            session.ask("Summarize it.")
        first, second = requests
        self.assertEqual(first["path"], "/api/chat")
        self.assertEqual(first["content_type"], "application/json")
        self.assertEqual(first["body"]["model"], "llama3.2-vision")
        self.assertIs(first["body"]["stream"], False)
        self.assertEqual(first["body"]["options"], {"temperature": 0})
        self.assertEqual(first["body"]["messages"][1]["images"], encode_images([self.image]))
        self.assertEqual([m["role"] for m in second["body"]["messages"]], ["system", "user", "assistant", "user"])
        self.assertEqual(second["body"]["messages"][2]["content"], answer)
        self.assertNotIn("images", second["body"]["messages"][-1])
        self.assertEqual(len(session.messages), 5)

    def test_ocr_prompt_and_multiple_images(self):
        with server() as (endpoint, requests):
            Conversation(OllamaClient(endpoint)).ocr([self.image, self.image])
        self.assertEqual(requests[0]["body"]["messages"][0]["content"], OCR_PROMPT)
        self.assertEqual(len(requests[0]["body"]["messages"][0]["images"]), 2)

    def test_ocr_without_image_fails(self):
        with self.assertRaises(ImageReaderError):
            Conversation(OllamaClient()).ocr([])

    def test_failure_does_not_mutate_history(self):
        with server({"error": "missing model"}, 404) as (endpoint, requests):
            session = Conversation(OllamaClient(endpoint), "system")
            before = session.messages
            with self.assertRaisesRegex(ImageReaderError, "HTTP 404"):
                session.ask("Read", [self.image])
        self.assertEqual(session.messages, before)

    def test_history_copy_is_independent(self):
        session = Conversation(OllamaClient(), "hello")
        view = session.messages
        view[0]["content"] = "mutated"
        self.assertEqual(session.messages[0]["content"], "hello")

    def test_invalid_endpoints_are_rejected(self):
        for endpoint in ["file:///x", "https://user:password@example.com", "http://localhost:0", "http://localhost:bad", "http://localhost/path", "http://localhost?x=1", "http://localhost/#x", "http://local host"]:
            with self.subTest(endpoint=endpoint), self.assertRaises(ImageReaderError):
                OllamaClient(endpoint)

    def test_invalid_timeouts_and_model(self):
        for timeout in [0, -1, float("nan"), float("inf"), True, "1"]:
            with self.subTest(timeout=timeout), self.assertRaises(ImageReaderError):
                OllamaClient(timeout=timeout)
        with self.assertRaises(ImageReaderError):
            OllamaClient(model="")

    def test_blank_or_huge_prompt_rejected_before_network(self):
        session = Conversation(OllamaClient())
        for prompt in ["", " ", "x" * 32001]:
            with self.subTest(length=len(prompt)), self.assertRaises(ImageReaderError):
                session.ask(prompt)

    def test_redirect_is_not_followed(self):
        with server(status=302) as (endpoint, requests):
            with self.assertRaisesRegex(ImageReaderError, "HTTP 302"):
                Conversation(OllamaClient(endpoint)).ask("test")
        self.assertEqual(len(requests), 1)

    def test_invalid_json(self):
        with server(b"not json") as (endpoint, requests):
            with self.assertRaisesRegex(ImageReaderError, "invalid JSON"):
                Conversation(OllamaClient(endpoint)).ask("test")

    def test_deep_json_is_rejected_without_mutating_history(self):
        nested = b"[" * 100_000 + b"0" + b"]" * 100_000
        with server(nested) as (endpoint, requests):
            session = Conversation(OllamaClient(endpoint), "system")
            before = session.messages
            with self.assertRaisesRegex(ImageReaderError, "invalid JSON"):
                session.ask("Read", [self.image])
        self.assertEqual(session.messages, before)

    def test_incomplete_and_malformed_responses(self):
        responses = [[], {"error": "failed"}, {"done": False}, {"done": True, "done_reason": "length", "message": {"role": "assistant", "content": "partial"}}, {"done": True, "message": {"role": "user", "content": "wrong role"}}, {"done": True, "message": {"role": "assistant", "content": " "}}, {"done": True, "message": {"role": "assistant", "content": "tools", "tool_calls": [{"name": "tool"}]}}]
        for response in responses:
            with self.subTest(response=response), server(response) as (endpoint, requests):
                with self.assertRaises(ImageReaderError):
                    Conversation(OllamaClient(endpoint)).ask("test")

    def test_response_size_limit(self):
        with server() as (endpoint, requests), patch("image_reader.client.MAX_RESPONSE_BYTES", 8):
            with self.assertRaisesRegex(ImageReaderError, "exceeds"):
                Conversation(OllamaClient(endpoint)).ask("test")

    def test_cli_ocr_writes_new_markdown_file(self):
        output = Path(self.temp.name) / "result.md"
        with server() as (endpoint, requests), redirect_stdout(io.StringIO()):
            result = main(["--endpoint", endpoint, "ocr", "--image", str(self.image), "--output", str(output)])
        self.assertEqual(result, 0)
        self.assertEqual(output.read_text(), "# Example\nReadable text\n")

    def test_cli_interactive_followup(self):
        with server() as (endpoint, requests), patch("builtins.input", side_effect=["What else?", "/quit"]), redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            result = main(["--endpoint", endpoint, "chat", "--image", str(self.image), "Read this"])
        self.assertEqual(result, 0)
        self.assertEqual(len(requests), 2)
        self.assertEqual(requests[1]["body"]["messages"][-1]["content"], "What else?")

    def test_cli_existing_output_is_preserved(self):
        output = Path(self.temp.name) / "result.md"
        output.write_text("existing")
        with server() as (endpoint, requests), redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            result = main(["--endpoint", endpoint, "ocr", "--image", str(self.image), "--output", str(output)])
        self.assertEqual(result, 1)
        self.assertEqual(output.read_text(), "existing")
        self.assertEqual(requests, [])


if __name__ == "__main__":
    unittest.main()
