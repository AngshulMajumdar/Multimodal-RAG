"""Builds a tiny, randomly initialised Qwen3-VL (model + processor) on disk with no downloads.

It has the same architecture, processor, chat-template mechanics (image placeholder expansion,
vision tower, deepstack, M-RoPE) as Qwen/Qwen3-VL-*-Instruct, so loading it through
AutoProcessor / AutoModelForImageTextToText exercises exactly the code path the Space uses.
Its outputs are of course meaningless.
"""
from __future__ import annotations

from pathlib import Path

SPECIAL = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|vision_start|>", "<|vision_end|>",
           "<|image_pad|>", "<|video_pad|>"]

CHAT_TEMPLATE = (
    "{% for message in messages %}<|im_start|>{{ message['role'] }}\n"
    "{% if message['content'] is string %}{{ message['content'] }}{% else %}"
    "{% for c in message['content'] %}{% if c['type'] == 'image' %}<|vision_start|><|image_pad|><|vision_end|>"
    "{% elif c['type'] == 'text' %}{{ c['text'] }}{% endif %}{% endfor %}{% endif %}<|im_end|>\n"
    "{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
)


def build(out: str | Path) -> Path:
    import torch
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import (Qwen2TokenizerFast, Qwen2VLImageProcessorFast, Qwen3VLConfig,
                              Qwen3VLForConditionalGeneration, Qwen3VLProcessor)
    from transformers.models.qwen3_vl.video_processing_qwen3_vl import Qwen3VLVideoProcessor

    out = Path(out)
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    corpus = ["The warranty is 36 months.", "Revenue by region FY2025.", "Text sources: [1] report.pdf, p.3",
              "Question: Answer: Cite text as [1] and images as [Img 1]."] * 50
    tok.train_from_iterator(corpus, trainers.BpeTrainer(vocab_size=400, special_tokens=SPECIAL,
                                                        initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    tokenizer = Qwen2TokenizerFast(tokenizer_object=tok, eos_token="<|im_end|>", pad_token="<|endoftext|>",
                                   unk_token=None, bos_token=None)
    ids = {t: tokenizer.convert_tokens_to_ids(t) for t in SPECIAL}

    processor = Qwen3VLProcessor(
        image_processor=Qwen2VLImageProcessorFast(patch_size=16, merge_size=2, temporal_patch_size=2,
                                                  size={"shortest_edge": 32 * 32, "longest_edge": 128 * 128}),
        tokenizer=tokenizer, video_processor=Qwen3VLVideoProcessor(), chat_template=CHAT_TEMPLATE)

    cfg = Qwen3VLConfig(
        text_config=dict(vocab_size=len(tokenizer), hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                         num_attention_heads=4, num_key_value_heads=2, head_dim=16, max_position_embeddings=4096,
                         rope_scaling={"rope_type": "default", "mrope_section": [4, 2, 2], "mrope_interleaved": True},
                         eos_token_id=ids["<|im_end|>"], pad_token_id=ids["<|endoftext|>"]),
        vision_config=dict(depth=2, hidden_size=32, intermediate_size=64, num_heads=2, patch_size=16,
                           spatial_merge_size=2, temporal_patch_size=2, out_hidden_size=64,
                           num_position_embeddings=256, deepstack_visual_indexes=[0]),
        image_token_id=ids["<|image_pad|>"], video_token_id=ids["<|video_pad|>"],
        vision_start_token_id=ids["<|vision_start|>"], vision_end_token_id=ids["<|vision_end|>"],
        eos_token_id=ids["<|im_end|>"], pad_token_id=ids["<|endoftext|>"],
    )
    torch.manual_seed(0)
    model = Qwen3VLForConditionalGeneration(cfg)
    model.generation_config.eos_token_id = ids["<|im_end|>"]
    model.generation_config.pad_token_id = ids["<|endoftext|>"]
    model.save_pretrained(out)
    processor.save_pretrained(out)
    return out


if __name__ == "__main__":
    import sys

    print(build(sys.argv[1] if len(sys.argv) > 1 else "tiny_qwen3vl"))
