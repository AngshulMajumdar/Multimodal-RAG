"""The ZeroGPU Space: transformers VLM backend and the Gradio app, driven over its API with gradio_client.

Uses a tiny random Qwen3-VL built on disk (tests/tiny_vlm.py) so nothing is downloaded; set
MRAG_VLM_ID to a real model (e.g. Qwen/Qwen3-VL-2B-Instruct) to run the same tests against it.
"""
import os
import socket
import sys
from pathlib import Path

import pytest

from conftest import ROOT

pytest.importorskip("transformers")


@pytest.fixture(scope="session")
def vlm_id(tmp_path_factory):
    if os.environ.get("MRAG_VLM_ID"):
        return os.environ["MRAG_VLM_ID"]
    try:
        from transformers import Qwen3VLForConditionalGeneration  # noqa: F401
    except ImportError:
        pytest.skip("transformers without Qwen3-VL (needs >=4.57)")
    sys.path.insert(0, str(ROOT / "tests"))
    import tiny_vlm

    return str(tiny_vlm.build(tmp_path_factory.mktemp("tiny_qwen3vl")))


@pytest.fixture(scope="session")
def vlm(vlm_id):
    from mrag.hf_vlm import HFVLM

    return HFVLM(vlm_id)


def test_vlm_image_tokens_match_vision_grid(vlm):
    from PIL import Image

    ims = [Image.new("RGB", (900, 500), "red"), Image.new("RGB", (300, 600), "blue")]
    inp = vlm.build_inputs("system", "Question: what colour?", ims)
    grid = inp["image_grid_thw"]
    merge = vlm.processor.image_processor.merge_size
    n_img_tokens = int((inp["input_ids"] == vlm.model.config.image_token_id).sum())
    assert grid.shape[0] == 2 and n_img_tokens == int(grid.prod(-1).sum()) // merge ** 2


def test_vlm_streams_text(vlm):
    from PIL import Image

    pieces = list(vlm.stream("system", "Question: hi", [Image.new("RGB", (64, 64))], max_new_tokens=8))
    assert pieces and all(isinstance(p, str) for p in pieces)


def test_vlm_errors_are_raised_not_hung(vlm):
    with pytest.raises(Exception):
        list(vlm.stream("system", "x", [], max_new_tokens=-5))


def test_generator_transformers_backend(vlm, tmp_path):
    from PIL import Image

    from mrag.config import Config
    from mrag.generate import Generator, ImageHit, TextHit

    p = tmp_path / "a.png"
    Image.new("RGB", (3000, 1500), "green").save(p)
    g = Generator(Config(llm_backend="transformers", max_new_tokens=6, max_image_side=256), vlm=vlm)
    out = g.answer("q?", [TextHit("1", "fact", "/d/x.pdf", 1, "text", 1.0)],
                   [ImageHit("i", str(p), "/d/x.pdf", 1, "", 0.3)])
    assert isinstance(out, str)


def _free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


@pytest.fixture(scope="module")
def client(vlm_id, tmp_path_factory):
    pytest.importorskip("gradio_client")
    os.environ["MRAG_VLM_ID"] = vlm_id
    os.environ.setdefault("MRAG_TEXT_BACKEND", "wordllama")
    os.chdir(tmp_path_factory.mktemp("space"))      # demo_docs is built next to app.py; keep tests clean
    sys.path.insert(0, str(ROOT))
    import app
    from gradio_client import Client

    port = _free_port()
    app.demo.queue().launch(server_port=port, prevent_thread_lock=True, quiet=True)
    yield Client(f"http://127.0.0.1:{port}/", verbose=False), app
    app.demo.close()


def test_app_demo_docs_and_streamed_answer(client):
    c, app = client
    status = c.predict(api_name="/load_demo")
    assert "7 file(s)" in status and "report.pdf" in status
    chat, gallery, sources, box, *_ = c.predict(
        "How long is the warranty on the Nimbus R2?", [], 6, 3, True,
        api_name="/prepare_answer")
    assert len(chat) == 2 and chat[0]["role"] == "user"
    assert "36-month" in sources and "report.pdf p.3" in sources
    assert gallery and "report.pdf p.3" in gallery[0]["caption"]
    job = c.submit(64, chat, api_name="/vlm_answer")
    outputs = list(job)
    final = job.result()
    assert outputs and final[-1]["role"] == "assistant" and final[-1]["content"]
    assert box["value"] == "" and box["interactive"] is False
    # gradio_client calls endpoints individually; the browser runs this chained event.
    c.predict(api_name="/finish_answer")


def test_app_upload_and_session_isolation(client, corpus):
    from gradio_client import Client, handle_file

    c, _ = client
    fresh = Client(c.src, verbose=False)
    with pytest.raises(Exception, match="Index some files"):
        fresh.predict("a cat?", [], 6, 3, True, api_name="/prepare_answer")
    status, _ = fresh.predict([handle_file(str(corpus / "images" / "office_cat.jpg"))], api_name="/index_files")
    assert "1 file(s)" in status and "office_cat.jpg" in status
    chat, gallery, sources, *_ = fresh.predict("show me a picture of a cat", [], 6, 3, True, api_name="/prepare_answer")
    assert gallery and "office_cat.jpg" in gallery[0]["caption"]
    assert "report.pdf" not in sources
    final = fresh.predict(64, chat, api_name="/vlm_answer")
    assert final[-1]["role"] == "assistant"
    fresh.predict(api_name="/finish_answer")


def test_app_busy_guard_and_failure_cleanup(client):
    c, app = client
    chat, *_ = c.predict("Who is the CEO?", [], 6, 0, False, api_name="/prepare_answer")
    with pytest.raises(Exception, match="still running"):
        c.predict(api_name="/reset")
    # This handler also runs when ZeroGPU rejects a request or generation fails.
    c.predict(api_name="/finish_answer")
    with pytest.raises(Exception, match="Submit a question"):
        c.predict(64, chat, api_name="/vlm_answer")
    c.predict(api_name="/finish_answer")


def test_app_cumulative_upload_limit(client, tmp_path):
    from gradio_client import Client, handle_file

    c, app = client
    fresh = Client(c.src, verbose=False)
    paths = []
    for i in range(app.MAX_FILES + 1):
        p = tmp_path / f"file{i}.txt"
        p.write_text(f"Document {i}: the robot payload is 25 kilograms.")
        paths.append(handle_file(str(p)))
    status, _ = fresh.predict(paths[:app.MAX_FILES], api_name="/index_files")
    assert f"{app.MAX_FILES} file(s)" in status
    with pytest.raises(Exception, match="indexed files per session"):
        fresh.predict(paths[-1:], api_name="/index_files")
    fresh.predict(api_name="/reset")


def test_app_generation_is_success_only_and_serialized(client):
    _, app = client
    gpu_id = next(i for i, fn in app.demo.fns.items() if fn.fn == app.vlm_answer)
    binding = next(d for d in app.demo.config["dependencies"] if d["id"] == gpu_id)
    assert binding["trigger_only_on_success"] is True
    assert app.demo.fns[gpu_id].concurrency_limit == 1
