from typing import List

import uvicorn
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import cv2
import base64
import io
from PIL import Image
from infer import infer

class ImageInput(BaseModel):
    img_name: str
    img_str: str


def run_infer(image):
    # Perform inference
    output_image = infer(image)
    # Encode the processed image back to base64
    _, buffer = cv2.imencode('.jpg', output_image)
    output_img_str = base64.b64encode(buffer).decode('utf-8')

    return output_img_str

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000","https://faults-handler.vercel.app", "https://faultshandler.com"],  # Adjust to your frontend's URL
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)

@app.get("/")
async def root():
    return {"message": "Fault Handler v0.1. Run <BASE_URL>/infer to detect fault"}

@app.post("/infer", tags=['Upload'], name='Upload image file')
async def create_upload_file(files: List[UploadFile] = File(...)):
    results = []
    for file in files:
        print(f"Received file: {file.filename}")
        # Read the file content
        content = await file.read()
        image = Image.open(io.BytesIO(content))
        # Convert the image to a NumPy array
        image_array = np.array(image)
        output_image = run_infer(image_array)
        # Append the name and base64 content to results
        results.append({
            "name": file.filename,
            "base64": output_image
        })

    return {"data": results, "message": "Image successfully Inferred" }

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8080)
