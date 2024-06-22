from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
import gradio as gr
import numpy as np

# Load the model and tokenizer
model_id = "vikhyatk/moondream2"
revision = "2024-05-20"
model = AutoModelForCausalLM.from_pretrained(
    model_id, trust_remote_code=True, revision=revision
)
tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)

def analyze_image_direct(image, question):
    # Convert PIL Image to the format expected by the model
    # Note: This step depends on the model's expected input format
    # For demonstration, assuming the model accepts PIL images directly
    enc_image = model.encode_image(image)  # This method might not exist; adjust based on actual model capabilities
    
    # Generate an answer to the question based on the encoded image
    # Note: This step is hypothetical and depends on the model's capabilities
    answer = model.answer_question(enc_image, question, tokenizer)  # Adjust based on actual model capabilities
    
    return answer

# Create Gradio interface
iface = gr.Interface(fn=analyze_image_direct,
                     inputs=[gr.Image(type="pil"), gr.Textbox(lines=2, placeholder="Enter your question here...")],
                     outputs='text',
                     title="Direct Image Question Answering",
                     description="Upload an image and ask a question about it directly using the model.")

# Launch the interface
iface.launch()
