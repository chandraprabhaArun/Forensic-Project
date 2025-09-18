from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import cv2
import numpy as np
import os  # <-- Import the 'os' module

# --- Import your functions from the other file ---
from document_deskew import process_document_image, imencode_to_base64

app = FastAPI(title="Document Processing API")

# Define a Pydantic model for a clear response structure
class ProcessedDocumentResponse(BaseModel):
    rotation_angle_applied: float
    image_b64: str
    saved_path: str  # <-- Add a field for the saved file path
    status: str

@app.post("/process-document", response_model=ProcessedDocumentResponse)
async def process_document(file: UploadFile = File(...)):
    """
    Accepts a document image, corrects its skew, crops it,
    saves the result locally, and returns the processed image.
    """
    # Read image from upload
    image_bytes = await file.read()
    np_arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    # Call your core logic function
    angle, cropped_image = process_document_image(img, padding=20)

    # --- NEW: Code to save the file locally ---
    # 1. Get the original filename and split it into name and extension
    original_filename = file.filename
    name, ext = os.path.splitext(original_filename)

    # 2. Create the new filename
    new_filename = f"{name}_processed{ext}"

    # 3. Save the processed image (which is a NumPy array) using OpenCV
    cv2.imwrite(new_filename, cropped_image)
    print(f"Successfully saved processed image to: {new_filename}")
    # --- END of new code ---

    # Encode the cropped image to Base64 for the response
    cropped_b64 = imencode_to_base64(cropped_image, ".jpg")

    return {
        "rotation_angle_applied": angle,
        "image_b64": cropped_b64,
        "saved_path": new_filename,  # <-- Include the new path in the response
        "status": "success"
    }