import cv2
import numpy as np
import gradio as gr
import requests
from io import BytesIO

def send_to_fastapi(image, question):
    url = 'https://ahmed007-modarb-api.hf.space/analyze-image/'

    
    # Convert the numpy image to bytes
    _, encoded_image = cv2.imencode('.png', image)
    image_bytes = BytesIO(encoded_image.tobytes())
    
    files = {'file': ('image.png', image_bytes, 'image/png')}
    data = {'question': question}
    response = requests.post(url, files=files, data=data)
    return response.json()['answer']

iface = gr.Interface(fn=send_to_fastapi,
                     inputs=[gr.Image(), gr.Textbox(lines=2, placeholder="Enter your question here...")],
                     outputs='text',
                     title="Image Question Answering",
                     description="Upload an image and ask a question about it.")

iface.launch()