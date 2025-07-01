import torch
import time
import argparse
import numpy as np

from ipex_llm.transformers import Qwen2VLForConditionalGeneration

import re
import os

import base64

from io import BytesIO
from PIL import Image

from pypdf import PdfReader
from olmocr.data.renderpdf import render_pdf_to_base64png
from olmocr.prompts import build_finetuning_prompt
from olmocr.prompts.anchor import get_anchor_text

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict Tokens using generate() API for Qwen2-VL model')
    parser.add_argument('file_path', type=str, nargs=1,
                        help='Path to the PDF file to process')
    parser.add_argument('--repo-id-or-model-path', type=str, default="allenai/olmOCR-7B-0225-preview",
                        help='The huggingface repo id for the Qwen2-VL model to be downloaded'
                             ', or the path to the huggingface checkpoint folder')
    parser.add_argument('--modelscope', action="store_true", default=False,
                        help="Use models from modelscope")
    parser.add_argument('--load-in-low-bit', type=str,
        default='sym_int4' ,
        help='load_in_low_bit, "float" to not use low bit, other options are sym_int4, asym_int4, sym_int5, asym_int5, sym_int8,nf3,nf4, fp4, fp8, fp8_e4m3, fp8_e5m2, fp6, gguf_iq2_xxs, gguf_iq2_xs, gguf_iq1_s, gguf_q4k_m, gguf_q4k_s,fp16, bf16, fp6_k, seeing https://github.com/intel/ipex-llm/blob/main/docs/mddocs/Overview/KeyFeatures/optimize_model.md')

    args = parser.parse_args()
    if args.modelscope:
        from modelscope import AutoProcessor
        model_hub = 'modelscope'
    else:
        from transformers import AutoProcessor
        model_hub = 'huggingface'

    model_path = args.repo_id_or_model_path

    if args.load_in_low_bit=="float":
        model = Qwen2VLForConditionalGeneration.from_pretrained(model_path,
                                                                optimize_model=True,
                                                                trust_remote_code=True,
                                                                #modules_to_not_convert=["vision"],
                                                                use_cache=True,
                                                                model_hub=model_hub)
        # Use .float() for better output, and use .half() for better speed
        model = model.float().to("xpu")
    else:
        model = Qwen2VLForConditionalGeneration.from_pretrained(model_path,
                                                                #load_in_4bit=True,
                                                                load_in_low_bit=args.load_in_low_bit,
                                                                optimize_model=True,
                                                                trust_remote_code=True,
                                                                modules_to_not_convert=["vision"],
                                                                use_cache=True,
                                                                model_hub=model_hub)
        # Use .float() for better output, and use .half() for better speed
        model = model.half().to("xpu")

    # The following code for generation is adapted from https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct#quickstart

    # The default range for the number of visual tokens per image in the model is 4-16384.
    # You can set min_pixels and max_pixels according to your needs, such as a token count range of 256-1280,
    # to balance speed and memory usage.
    min_pixels = 256*28*28
    max_pixels = 1280*28*28
    processor = AutoProcessor.from_pretrained(model_path, min_pixels=min_pixels, max_pixels=max_pixels)

    file_path = args.file_path[0]
    reader = PdfReader(file_path)
    # Get the total number of pages in the PDF
    num_pages = len(reader.pages)
    print(f"Total pages in PDF: {num_pages}")

    with torch.inference_mode():
        # Loop through all pages
        for pg_num in range(1, num_pages + 1):
            print(f"Processing page {pg_num} of {num_pages}")

            # Render page 1 to an image
            image_base64 = render_pdf_to_base64png(file_path, pg_num, target_longest_image_dim=1024)

            # Build the prompt, using document metadata
            anchor_text = get_anchor_text(file_path, pg_num, pdf_engine="pdfreport", target_length=4000)#TODO: reuse reader for the same PdfReader is used inside
            prompt = build_finetuning_prompt(anchor_text)

            # Build the full prompt
            messages = [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}},
                            ],
                        }
                    ]

            # Apply the chat template and processor
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            main_image = Image.open(BytesIO(base64.b64decode(image_base64)))

            inputs = processor(
                text=[text],
                images=[main_image],
                padding=True,
                return_tensors="pt",
            )
            inputs = {key: value.to('xpu') for (key, value) in inputs.items()}

            st = time.time()
            # Generate the output
            output = model.generate(
                        **inputs,
                        temperature=0.8,
                        max_new_tokens=4096,
                        num_return_sequences=1,
                        do_sample=True,
                    )
            torch.xpu.synchronize()
            end = time.time()
            output = output.cpu()

            # Decode the output
            prompt_length = inputs["input_ids"].shape[1]
            new_tokens = output[:, prompt_length:]
            text_output = processor.tokenizer.batch_decode(
                new_tokens, skip_special_tokens=True
            )

            print(text_output)

            print('-'*20)
            print(f'Inference time: {end-st} s')

            # Save to a markdown file
            output_filename = f"{file_path}-{args.load_in_low_bit}-{pg_num}.md"
            with open(output_filename, "w", encoding="utf-8") as f:

                # Extract and save natural_text from the output
                try:
                    # Parse the JSON string from the text_output
                    import json
                    output_text = text_output[0]  # Get the first (and only) element from the list
                    output_json = json.loads(output_text)

                    # Extract the natural_text field
                    natural_text = output_json.get("natural_text", "")

                    f.write(natural_text)
                except Exception as e:
                    print(f"Error extracting and saving natural text: {e}")
                    f.write(str(text_output))

                f.write("\n"+"-"*20+f'\nInference time: {end-st} s')
                print(f"Saved natural_text to {output_filename}")

            torch.xpu.empty_cache()
# set SYCL_CACHE_PERSISTENT=1
# python ipex-llm-olmocr.py --load-in-low-bit fp8 "%1"