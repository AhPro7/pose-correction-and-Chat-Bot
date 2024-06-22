from fastapi import FastAPI, File, Form, UploadFile
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
import io

app = FastAPI()

model_id = "vikhyatk/moondream2"
revision = "2024-05-20"
model = AutoModelForCausalLM.from_pretrained(
    model_id, trust_remote_code=True, revision=revision
)
tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)

@app.post("/analyze-image/")
async def analyze_image(file: UploadFile = File(...), question: str = Form(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))
    enc_image = model.encode_image(image)
    answer = model.answer_question(enc_image, question, tokenizer)
    return {"answer": answer}