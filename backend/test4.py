import torch
import base64
from io import BytesIO
from PIL import Image
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

from olmocr.data.renderpdf import render_pdf_to_base64png
from olmocr.prompts.prompts import build_finetuning_prompt
from olmocr.prompts.anchor import get_anchor_text

# Path to PDF file
FILE_PATH = "/home/auropro/Documents/pr_ocr/olmocr/codes/Screenshot from 2025-04-14 17-31-12.pdf"

print(FILE_PATH)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

torch.cuda.empty_cache()
torch.cuda.ipc_collect()

# Load model
model = Qwen2VLForConditionalGeneration.from_pretrained(
    "allenai/olmOCR-7B-0225-preview",
    torch_dtype=torch.float16,
    device_map="auto",
    max_memory={0: "16GB"}
).eval().to(device)

# Load processor
processor = AutoProcessor.from_pretrained("allenai/olmOCR-7B-0225-preview")

# Render page to high-res image
image_base64 = render_pdf_to_base64png(FILE_PATH, 1, target_longest_image_dim=1000)
main_image = Image.open(BytesIO(base64.b64decode(image_base64)))
main_image.save("debug_page.png")  # optional for visual inspection

# 🧠 Prompt that lets model infer table layout on its own
prompt = """
You are an OCR agent. The image contains a table.
Your task is to extract the full table **as-is**, inferring the number of rows and columns automatically.

Please return in CSV format

Make sure:
- No columns are skipped.
- All rows and columns are captured in full.
- Do not assume a header
"""

# Format message
messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}},
        ],
    }
]

# Prepare model input
text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = processor(
    text=[text],
    images=[main_image],
    padding=True,
    return_tensors="pt",
)
inputs = {key: value.to(device) for key, value in inputs.items()}

# Generate output
output = model.generate(
    **inputs,
    temperature=0.2,
    max_new_tokens=2500,
    num_return_sequences=1,
    do_sample=False  # deterministic
)

# Decode
prompt_length = inputs["input_ids"].shape[1]
new_tokens = output[:, prompt_length:]
text_output = processor.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)

print("---- Extracted Table ----")
print(text_output[0])




