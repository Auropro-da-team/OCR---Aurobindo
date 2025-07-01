import torch
import base64
from io import BytesIO
from PIL import Image
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

from olmocr.data.renderpdf import render_pdf_to_base64png
from olmocr.prompts.prompts import build_finetuning_prompt
from olmocr.prompts.anchor import get_anchor_text


# Path to the PDF file
FILE_PATH = "/home/auropro/Documents/pr_ocr/olmocr/codes/Screenshot from 2025-04-14 17-16-17.pdf"


#home/auropro/Documents/pr_ocr/olmocr/codes/test3.py

print(FILE_PATH)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

torch.cuda.empty_cache()
torch.cuda.ipc_collect()

# Load the model with memory optimization
model = Qwen2VLForConditionalGeneration.from_pretrained(
    "allenai/olmOCR-7B-0225-preview",
    torch_dtype=torch.float16,  # Reduce memory usage
    device_map="auto",  # Auto-distribute model across available GPUs
    max_memory={0: "18GB"}  # Restrict memory usage to avoid OOM
).eval()
model.to(device)

# Load the processor
processor = AutoProcessor.from_pretrained("allenai/olmOCR-7B-0225-preview")

# Render page 1 to an image (Reduce image size for memory efficiency)
image_base64 = render_pdf_to_base64png(FILE_PATH, 1, target_longest_image_dim=1000)

 
# Extract anchor text from the document
anchor_text = get_anchor_text(FILE_PATH, 1, pdf_engine="pdfreport", target_length=4000)
prompt = build_finetuning_prompt(anchor_text)


prompt += "\n\nPlease extract all rows and all columns of the table exactly as shown in the image. Do not skip any cells. Maintain left-to-right column order."

# Construct messages for the model
messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}},
        ],
    }
]

# Process inputs
text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
main_image = Image.open(BytesIO(base64.b64decode(image_base64)))

inputs = processor(
    text=[text],
    images=[main_image],
    padding=True,
    return_tensors="pt",
)
inputs = {key: value.to(device) for key, value in inputs.items()}

# Generate output with optimized parameters
output = model.generate(
    **inputs,
    temperature=0.2,
    max_new_tokens=2500,
    num_return_sequences=1,
    do_sample=False,
    top_k=50,  # Consider only top 50 predictions
    top_p=1.0
)

# Decode the output
prompt_length = inputs["input_ids"].shape[1]
new_tokens = output[:, prompt_length:]
text_output = processor.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)

print(text_output)