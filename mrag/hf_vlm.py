"""In-process vision-language model via Hugging Face transformers (default Qwen3-VL-4B-Instruct).

Works on CUDA, on ZeroGPU (load at import time with device="cuda", call `stream` inside @spaces.GPU),
and on CPU for small models.
"""
from __future__ import annotations

import os
from collections.abc import Iterator
from threading import Thread

from PIL import Image


def default_device() -> str:
    import torch

    if os.environ.get("SPACES_ZERO_GPU") or torch.cuda.is_available():
        return "cuda"  # on ZeroGPU, .to("cuda") at startup is deferred until a GPU is attached
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class HFVLM:
    def __init__(self, model_id: str, device: str | None = None):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.device = device or default_device()
        dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForImageTextToText.from_pretrained(model_id, dtype=dtype).to(self.device).eval()
        self.model_id = model_id

    def build_inputs(self, system: str, prompt: str, images: list[Image.Image]):
        content = [{"type": "image", "image": im} for im in images] + [{"type": "text", "text": prompt}]
        messages = [{"role": "system", "content": [{"type": "text", "text": system}]},
                    {"role": "user", "content": content}]
        return self.processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
        ).to(self.model.device)

    def stream(self, system: str, prompt: str, images: list[Image.Image],
               max_new_tokens: int = 512, temperature: float = 0.0) -> Iterator[str]:
        from transformers import TextIteratorStreamer

        inputs = self.build_inputs(system, prompt, images)
        streamer = TextIteratorStreamer(self.processor.tokenizer, skip_prompt=True,
                                        skip_special_tokens=True, timeout=300)
        kw = dict(**inputs, streamer=streamer, max_new_tokens=max_new_tokens,
                  do_sample=temperature > 0)
        if temperature > 0:
            kw["temperature"] = temperature
        error: list[BaseException] = []

        def run():
            try:
                self.model.generate(**kw)
            except BaseException as e:  # surface errors instead of hanging the streamer
                error.append(e)
                streamer.end()

        t = Thread(target=run, daemon=True)
        t.start()
        for piece in streamer:
            if piece:
                yield piece
        t.join()
        if error:
            raise error[0]
