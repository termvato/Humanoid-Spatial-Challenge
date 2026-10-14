"""Vision-language model on Modal for the benchmark and dev tools (tools/label.py, tools/bench_predict.py).

roomdiff.vlm.LocalVLM (Qwen2.5-VL-7B-Instruct, bf16) on one A10G; weights cached in the `hf-cache`
volume. The end-to-end app (infra/modal_app.py) runs the VLM in its own container instead.
"""

import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.5.1", "torchvision==0.20.1", index_url="https://download.pytorch.org/whl/cu121")
    .pip_install("transformers==4.51.3", "accelerate", "qwen-vl-utils==0.0.10", "pillow")
    .add_local_python_source("roomdiff")
)
hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)
app = modal.App("roomdiff-vlm")


@app.cls(image=image, gpu="A10G", timeout=1800, scaledown_window=60, volumes={"/root/.cache/huggingface": hf_cache})
class VLM:
    @modal.enter()
    def load(self) -> None:
        from roomdiff.vlm import LocalVLM

        self.vlm = LocalVLM()

    @modal.method()
    def ask(self, items: list[list[bytes]], prompt: str | list[str], max_new_tokens: int = 16) -> list[str]:
        """One answer per item (roomdiff.vlm.LocalVLM.ask)."""
        return self.vlm.ask(items, prompt, max_new_tokens)
