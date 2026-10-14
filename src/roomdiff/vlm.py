"""Qwen2.5-VL-7B-Instruct (Apache 2.0) for change names and before/after verdicts (roomdiff.labelling).

`LocalVLM` runs on this machine's GPU (bf16, ~16 GB). infra/modal_vlm.py serves the same class on
Modal for the benchmark tools.
"""

from __future__ import annotations

import io

MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"


class LocalVLM:
    def __init__(self, device: str = "cuda", max_pixels: int = 448 * 448):
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(MODEL, torch_dtype=torch.bfloat16,
                                                                        device_map=device)
        self.proc = AutoProcessor.from_pretrained(MODEL, max_pixels=max_pixels)

    def ask(self, items: list[list[bytes]], prompt: str | list[str], max_new_tokens: int = 16) -> list[str]:
        """One answer per item; an item is a list of JPEG images shown together with `prompt` (one for all
        items, or one per item). Greedy decoding."""
        from PIL import Image

        prompts = [prompt] * len(items) if isinstance(prompt, str) else prompt
        out = []
        for jpegs, text_prompt in zip(items, prompts, strict=True):
            imgs = [Image.open(io.BytesIO(b)).convert("RGB") for b in jpegs]
            msgs = [{"role": "user", "content": [*({"type": "image"} for _ in imgs), {"type": "text", "text": text_prompt}]}]
            text = self.proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            inp = self.proc(text=[text], images=imgs, return_tensors="pt").to(self.model.device)
            gen = self.model.generate(**inp, max_new_tokens=max_new_tokens, do_sample=False)
            out.append(self.proc.batch_decode(gen[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)[0])
        return out

    def close(self) -> None:
        import torch

        del self.model
        torch.cuda.empty_cache()
