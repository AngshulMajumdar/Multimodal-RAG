"""Generator backends against local HTTP servers that implement the Ollama and OpenAI wire protocols."""
import base64
import io
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from PIL import Image

from mrag.config import Config
from mrag.generate import Generator, ImageHit, TextHit


class FakeLLM:
    def __init__(self):
        self.requests = []
        outer = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                outer.requests.append((self.path, dict(self.headers), body))
                words = ["The ", "warranty ", "is ", "36 ", "months ", "[1]."]
                if self.path == "/api/chat":
                    if body["model"] == "missing":
                        return self._send(404, {"error": "model 'missing' not found"})
                    if not body["stream"]:
                        return self._send(200, {"message": {"role": "assistant", "content": "".join(words)},
                                                "done": True})
                    self.send_response(200)
                    self.send_header("Content-Type", "application/x-ndjson")
                    self.end_headers()
                    for w in words:
                        self.wfile.write((json.dumps({"message": {"content": w}, "done": False}) + "\n").encode())
                    self.wfile.write((json.dumps({"message": {"content": ""}, "done": True}) + "\n").encode())
                elif self.path == "/v1/chat/completions":
                    if not body["stream"]:
                        return self._send(200, {"choices": [{"message": {"content": "".join(words)}}]})
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.end_headers()
                    for w in words:
                        self.wfile.write(f"data: {json.dumps({'choices': [{'delta': {'content': w}}]})}\n\n".encode())
                    self.wfile.write(b"data: [DONE]\n\n")
                else:
                    self._send(404, {"error": "no route"})

            def _send(self, code, obj):
                b = json.dumps(obj).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(b)))
                self.end_headers()
                self.wfile.write(b)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()


@pytest.fixture(scope="module")
def fake():
    f = FakeLLM()
    yield f
    f.server.shutdown()


@pytest.fixture
def hits(tmp_path):
    p = tmp_path / "chart.png"
    Image.new("RGB", (2000, 1000), "red").save(p)
    texts = [TextHit("t1", "Every R2 ships with a 36-month warranty.", "/d/report.pdf", 3, "ocr", 0.8)]
    images = [ImageHit("i1", str(p), "/d/report.pdf", 3, "warranty", 0.3)]
    return texts, images


@pytest.mark.parametrize("stream", [False, True])
def test_ollama(fake, hits, stream):
    g = Generator(Config(llm_backend="ollama", llm_url=fake.url, llm_model="qwen2.5vl:3b", max_image_side=512))
    out = g.answer("How long is the warranty?", *hits, stream=stream)
    text = "".join(out) if stream else out
    assert text == "The warranty is 36 months [1]."
    path, _, body = fake.requests[-1]
    assert path == "/api/chat" and body["model"] == "qwen2.5vl:3b" and body["stream"] is stream
    sys_msg, user = body["messages"]
    assert sys_msg["role"] == "system" and "[1] (report.pdf, p.3, OCR of an image)" in user["content"]
    img = Image.open(io.BytesIO(base64.b64decode(user["images"][0])))
    assert img.size == (512, 256)                                  # downscaled, aspect preserved
    assert body["options"]["num_ctx"] == 8192


def test_ollama_text_only_model(fake, hits):
    g = Generator(Config(llm_backend="ollama", llm_url=fake.url, llm_model="llama3.2:3b", llm_vision=False))
    g.answer("q", *hits)
    assert "images" not in fake.requests[-1][2]["messages"][1]


def test_ollama_errors(fake, hits):
    g = Generator(Config(llm_backend="ollama", llm_url=fake.url, llm_model="missing"))
    with pytest.raises(RuntimeError, match="404"):
        g.answer("q", *hits)
    g = Generator(Config(llm_backend="ollama", llm_url="http://127.0.0.1:9", llm_model="x"))
    with pytest.raises(RuntimeError, match="ollama serve"):
        g.answer("q", *hits)


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("suffix", ["", "/v1"])
def test_openai_compatible(fake, hits, stream, suffix):
    g = Generator(Config(llm_backend="openai", llm_url=fake.url + suffix, llm_model="local", llm_api_key="k"))
    out = g.answer("How long is the warranty?", *hits, stream=stream)
    assert ("".join(out) if stream else out) == "The warranty is 36 months [1]."
    path, headers, body = fake.requests[-1]
    assert path == "/v1/chat/completions" and headers["Authorization"] == "Bearer k"
    parts = body["messages"][1]["content"]
    assert parts[0]["type"] == "text" and parts[1]["image_url"]["url"].startswith("data:image/png;base64,")
