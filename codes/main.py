#MAIN.PY WITH TEXT AND TABLE READING:


from fastapi import FastAPI, UploadFile, File, HTTPException, Form, Request
from fastapi.responses import JSONResponse, Response
from fastapi.middleware.cors import CORSMiddleware
import traceback
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from PIL import Image
from io import BytesIO
import torch
import base64
import tempfile
import os
from contextlib import asynccontextmanager
from olmocr.data.renderpdf import render_pdf_to_base64png
import httpx

# Global model variables
model = None
processor = None

# Define lifespan handler
@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, processor
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

        model = Qwen2VLForConditionalGeneration.from_pretrained(
            "allenai/olmOCR-7B-0225-preview",
            torch_dtype=torch.float16,
            device_map="auto",
            max_memory={0: "18GB"}
        ).eval().to(device)

        processor = AutoProcessor.from_pretrained("allenai/olmOCR-7B-0225-preview")
        print("Model and processor loaded successfully")

    except Exception as e:
        print(f"Failed to load model: {str(e)}")
        raise RuntimeError("Model load failed") from e

    yield
    print("Shutting down...")

# Initialize FastAPI
app = FastAPI(
    title="VisionGrid",
    description="An OCR API to extract table and text data from PDFs",
    version="1.0",
    lifespan=lifespan
)

origins = [
    "http://localhost:5173",  # your local frontend URL
    "https://c644-183-82-1-156.ngrok-free.app/",  # your ngrok URL
]

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# OCR logic
def extract_text_and_table_from_pdf(pdf_path):
    try:
        image_base64 = render_pdf_to_base64png(pdf_path, 1, target_longest_image_dim=1200)
        main_image = Image.open(BytesIO(base64.b64decode(image_base64)))

        if main_image.mode != "RGB":
            main_image = main_image.convert("RGB")

        prompt = """
        You are an OCR agent. The image may contain both regular text and tables. It could also contain JUST a table
        Your job is to extract:

        1. Any regular (handwritten or printed) text in reading order.
        2. Any tables in CSV format (clearly labeled), without losing rows or columns.

        Output format:
        
        TEXT:
        <extracted text here>

        TABLE:
        <CSV formatted table here>
        
        Do not include any metadata, comments, or explanations.
        """

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}},
                ],
            }
        ]

        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[main_image], padding=True, return_tensors="pt")
        device = next(model.parameters()).device
        inputs = {key: value.to(device) for key, value in inputs.items()}

        with torch.no_grad():
            output = model.generate(
                **inputs,
                temperature=0.2,
                max_new_tokens=3000,
                num_return_sequences=1,
                do_sample=False
            )

        prompt_length = inputs["input_ids"].shape[1]
        new_tokens = output[:, prompt_length:]
        text_output = processor.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
        return text_output[0].replace("\\n", "\n").strip()

    except Exception as e:
        raise Exception(f"Error in OCR extraction: {str(e)}")

# Endpoint
@app.post("/api/extract-table")
async def extract_table(pdf: UploadFile = File(...), rows: str = Form(None), columns: str = Form(None)):
    if not pdf.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="File must be a PDF")

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(await pdf.read())
            tmp.flush()
            extracted_data = extract_text_and_table_from_pdf(tmp.name)
        return {"extracted_data": extracted_data}

    except Exception as e:
        raise HTTPException(status_code=500, detail=traceback.format_exc())

    finally:
        if os.path.exists(tmp.name):
            os.remove(tmp.name)

# Proxy frontend
FRONTEND_PORT = 5173
FRONTEND_URL = f"http://localhost:{FRONTEND_PORT}"

@app.api_route("/{full_path:path}", methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"])
async def proxy_to_frontend(full_path: str, request: Request):
    frontend_url = f"{FRONTEND_URL}/{full_path}"
    headers = dict(request.headers)

    async with httpx.AsyncClient() as client:
        resp = await client.request(
            method=request.method,
            url=frontend_url,
            content=await request.body(),
            headers=headers,
        )

    return Response(
        content=resp.content,
        status_code=resp.status_code,
        headers=dict(resp.headers),
    )
