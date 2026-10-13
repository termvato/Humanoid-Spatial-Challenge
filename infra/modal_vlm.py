"""Vision-language model on Modal: name the object in each change's crops (roomdiff.label).

Qwen2.5-VL-7B-Instruct (Apache 2.0), bf16 on one A10G. Called by tools/label.py; one call
labels a whole pair. Weights are cached in the `hf-cache` volume.
"""

import modal

MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.5.1", "torchvision==0.20.1", index_url="https://download.pytorch.org/whl/cu121")
    .pip_install("transformers==4.51.3", "accelerate", "qwen-vl-utils==0.0.10", "pillow")
)
hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)
app = modal.App("roomdiff-vlm")


@app.cls(image=image, gpu="A10G", timeout=1800, scaledown_window=60, volumes={"/root/.cache/huggingface": hf_cache})
class VLM:
    @modal.enter()
    def load(self) -> None:
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(MODEL, torch_dtype=torch.bfloat16,
                                                                        device_map="cuda")
        self.proc = AutoProcessor.from_pretrained(MODEL, max_pixels=448 * 448)

    @modal.method()
    def ask(self, items: list[list[bytes]], prompt: str | list[str], max_new_tokens: int = 16) -> list[str]:
        """One answer per item; an item is a list of JPEG images shown together with `prompt` (one for all
        items, or one per item)."""
        import io

        from PIL import Image

        out = []
        prompts = [prompt] * len(items) if isinstance(prompt, str) else prompt
        for jpegs, text_prompt in zip(items, prompts, strict=True):
            imgs = [Image.open(io.BytesIO(b)).convert("RGB") for b in jpegs]
            msgs = [{"role": "user", "content": [*({"type": "image"} for _ in imgs), {"type": "text", "text": text_prompt}]}]
            text = self.proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            inp = self.proc(text=[text], images=imgs, return_tensors="pt").to("cuda")
            gen = self.model.generate(**inp, max_new_tokens=max_new_tokens, do_sample=False)
            out.append(self.proc.batch_decode(gen[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)[0])
        return out
