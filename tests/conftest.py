import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "examples"))

from mrag.config import Config  # noqa: E402


@pytest.fixture(scope="session")
def corpus(tmp_path_factory):
    import make_demo_corpus

    return make_demo_corpus.build(tmp_path_factory.mktemp("corpus"))


def model_cfg(data_dir) -> Config:
    """Real models. Text: WordLlama (weights ship in the wheel) unless MRAG_TEXT_BACKEND says otherwise.
    CLIP: MRAG_CLIP_PRETRAINED (tag or local .pt), default laion400m_e32."""
    return Config.load(
        data_dir=str(data_dir),
        text_backend=os.environ.get("MRAG_TEXT_BACKEND", "wordllama"),
        llm_backend="extractive",
    )


@pytest.fixture(scope="session")
def rag(corpus, tmp_path_factory):
    from mrag.pipeline import MultimodalRAG

    r = MultimodalRAG(model_cfg(tmp_path_factory.mktemp("index")))
    stats = r.ingest([corpus])
    assert stats["errors"] == [], stats
    yield r
    r.close()
