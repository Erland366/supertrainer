from typing import Union
import random

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
import io


app = FastAPI()


@app.get("/")
def read_root():
    return random.choice([{"prediction": "hard"}, {"prediction": "soft"}])


from swift import Swift
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from swift.llm import (
    get_model_tokenizer, get_template, inference,
    get_default_template_type, inference_stream
)
from swift.utils import seed_everything


model_type, model_checkpoint = ("deepseek-vl-1_3b-chat", "/home/rifo.genadi/material_classification/output/deepseek-vl-1_3b-chat/v0-20241115-100136/checkpoint-422")
template_type = get_default_template_type(model_type)
print(f'template_type: {template_type}')
model, tokenizer = get_model_tokenizer(model_type, model_kwargs={'device_map': "cuda:0"})
model = Swift.from_pretrained(model, model_checkpoint)


model.generation_config.max_new_tokens = 128
model.generation_config.temperature=0.01
template = get_template(template_type, tokenizer)
seed_everything(42)

from datasets import load_dataset
soft_grips = ['ceramic', 'fabric', 'food', 'paper', 'glass', 'other'] 
hard_grips = ['wood', 'metal', 'unknown', 'plastic']

model.generation_config.max_new_tokens = 144
model.generation_config.temperature=0.01
template = get_template(template_type, tokenizer)

query = f"You are a helpful language and vision assistant. You are able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language.\n\nUser: <image_placeholder>\nWhat material is this object made of? Respond unknown if you are not sure. Answer only with the name of the material.\nShort answer:\n\nAssistant:"

@app.post("/inference/")
async def process_image(file: UploadFile = File(...)):
    try:
        # Check the file type
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be an image.")
        
        # Read the uploaded file as bytes
        contents = await file.read()
        
        # Open the image with PIL to confirm it's a valid image
        image = Image.open(io.BytesIO(contents))
        
        # You can now process the image, save it, or perform other operations
        # For this example, we'll just return the image's format and size
        image_info = {
            "filename": file.filename,
            "format": image.format,
            "size": image.size,  # (width, height)
        }

        # inference
        images = [image]
        response, _ = inference(model, template, query, images=images)
        pred = 'hard' if response in hard_grips else 'soft'
        return JSONResponse(content={"status": "success", "class": response, "prediction": pred})
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {e}")

