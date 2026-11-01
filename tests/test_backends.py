"""sentence-transformers backend (the default, BGE-small on a normal machine), exercised with a tiny
locally-built BERT so the test needs no download."""
import re

import numpy as np
import pytest

from mrag.config import Config


@pytest.fixture(scope="module")
def tiny_st(tmp_path_factory):
    pytest.importorskip("sentence_transformers")
    from sentence_transformers import SentenceTransformer, models
    from transformers import BertConfig, BertModel, BertTokenizerFast

    d = tmp_path_factory.mktemp("tiny")
    words = sorted(set(re.findall(r"[a-z]+", "the warranty lasts thirty six months for every robot unit")))
    (d / "vocab.txt").write_text("\n".join(["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + words))
    tok = BertTokenizerFast(str(d / "vocab.txt"))
    m = BertModel(BertConfig(vocab_size=len(words) + 5, hidden_size=32, num_hidden_layers=1,
                             num_attention_heads=2, intermediate_size=64))
    tok.save_pretrained(d / "hf")
    m.save_pretrained(d / "hf")
    st = SentenceTransformer(modules=[models.Transformer(str(d / "hf")), models.Pooling(32, "cls"), models.Normalize()])
    st.save(str(d / "st"))
    return str(d / "st")


def test_sentence_transformers_backend(tiny_st):
    from mrag.models import TextEmbedder

    e = TextEmbedder(Config(text_backend="sentence-transformers", text_model=tiny_st, query_prefix="query: "))
    D = e.embed_docs(["the warranty lasts", "every robot unit"])
    q = e.embed_query("warranty")
    assert e.dim == 32 and D.shape == (2, 32) and q.shape == (32,)
    assert np.allclose(np.linalg.norm(D, axis=1), 1, atol=1e-5) and abs(np.linalg.norm(q) - 1) < 1e-5
    assert e.name == f"st:{tiny_st}"


def test_wordllama_backend():
    from mrag.models import TextEmbedder

    e = TextEmbedder(Config(text_backend="wordllama"))
    v = e.embed_docs(["The cat sat on the mat.", "A kitten is resting on a rug.", "Quarterly revenue grew."])
    sims = v @ v.T
    assert sims[0, 1] > sims[0, 2] + 0.2      # semantic, not lexical: cat~kitten share no words


def test_unknown_backend():
    from mrag.models import TextEmbedder

    with pytest.raises(ValueError):
        TextEmbedder(Config(text_backend="nope"))
