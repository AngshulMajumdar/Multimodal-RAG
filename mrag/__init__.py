"""mrag: multimodal retrieval-augmented generation with free, local models."""
from .config import Config

__all__ = ["Config", "MultimodalRAG"]
__version__ = "1.0.0"


def __getattr__(name):
    if name == "MultimodalRAG":
        from .pipeline import MultimodalRAG
        return MultimodalRAG
    raise AttributeError(name)
