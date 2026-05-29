from PIL import Image
import torch

from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
)


class VisionParser:

    _model = None
    _processor = None

    def __init__(
        self,
        model_name="Qwen/Qwen2.5-VL-7B-Instruct",
    ):

        if VisionParser._model is None:

            VisionParser._processor = AutoProcessor.from_pretrained(
                model_name,
                trust_remote_code=True,
            )

            VisionParser._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
            )

        self.model = VisionParser._model
        self.processor = VisionParser._processor

    def should_process(self, image: Image.Image) -> bool:

        w, h = image.size
        aspect = w / h

        # too small → logo/icon
        if w < 120 or h < 120:
            return False

        # banner/header strip → likely not useful
        if w > 800 and h < 120:
            return False

        # tall text page fragments → appendix/header pages
        if h > 1000 and w < 250:
            return False

        # near-square small blocks → logos
        if 0.8 < aspect < 1.2 and w < 300:
            return False

        return True

    def analyze_image(self, image_path: str) -> str:

        image = Image.open(image_path).convert("RGB")

        if not self.should_process(image):
            return ""

        prompt = """
You are a visual text extraction engine for retrieval systems.

RULES (VERY STRICT):
- Do NOT interpret the image
- Do NOT explain meaning
- Do NOT describe what the image represents
- Do NOT guess diagrams, workflows, systems, or concepts
- Do NOT add summaries

ONLY extract what is explicitly visible.

OUTPUT RULES:
- If image contains readable structured content (charts, tables, UI, diagrams):
    → output ONLY visible labels, text, and objects as bullet points
- If image is unclear, decorative, logo, or mostly text header:
    → return EMPTY OUTPUT

WHAT TO EXTRACT:
- exact visible text
- box labels
- axis labels in charts
- table headers and values
- UI button/text labels
- arrows ONLY if labeled

WHAT TO IGNORE COMPLETELY:
- logos
- branding
- decorative shapes
- stylized designs
- titles (APPENDIX, CHAPTER, etc.)
- any inferred meaning

FORMAT:
- bullet points only
- no sentences
- no explanations
"""

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.processor(
            text=[text],
            images=[image],
            return_tensors="pt",
        ).to(self.model.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=150,
                do_sample=False,
                temperature=0.0,
            )

        prompt_len = inputs["input_ids"].shape[1]
        output_ids = generated_ids[0][prompt_len:]

        output = self.processor.decode(
            output_ids,
            skip_special_tokens=True,
        ).strip()

        if self._is_noise(output):
            return ""

        return output

    def _is_noise(self, text: str) -> bool:

        if not text:
            return True

        t = text.lower()

        blacklist = [
            "logo",
            "branding",
            "geometric",
            "design element",
            "decorative",
            "stylized",
            "symbol",
            "corporate",
            "caf",
            "appendix",
            "title",
        ]

        if any(b in t for b in blacklist):
            return True

        if len(t.split()) < 6:
            return True

        return False