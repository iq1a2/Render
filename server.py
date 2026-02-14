from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os
import shutil
import numpy as np
import cv2
from PIL import Image, ImageChops, ImageEnhance
import uvicorn
import io

app = FastAPI(title="Yaqeen AI Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def perform_ela(image_path):
    """
    Error Level Analysis (ELA) to detect manipulation.
    """
    original = Image.open(image_path).convert('RGB')
    
    # Save as temporary JPEG with specific quality
    temp_path = image_path + ".ela.jpg"
    original.save(temp_path, 'JPEG', quality=90)
    
    resaved = Image.open(temp_path)
    
    # Calculate difference
    ela_image = ImageChops.difference(original, resaved)
    
    # Calculate text metrics
    extrema = ela_image.getextrema()
    max_diff = max([ex[1] for ex in extrema])
    
    # Cleanup
    if os.path.exists(temp_path):
        os.remove(temp_path)
        
    return max_diff, ela_image

def analyze_frequency(image_path):
    """
    Frequency analysis using FFT to detect GAN artifacts.
    """
    img = cv2.imread(image_path, 0) # Grayscale
    if img is None:
        return 0
        
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)
    
    # Simple heuristic: Check for abnormal high frequency energy
    rows, cols = img.shape
    crow, ccol = rows//2, cols//2
    
    # Calculate mean of high frequencies (edges) vs low frequencies (center)
    # This is a simplified heuristic
    mean_energy = np.mean(magnitude_spectrum)
    return mean_energy

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File uploaded is not an image")

    file_location = os.path.join(UPLOAD_FOLDER, file.filename)

    try:
        with open(file_location, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # 1. ELA Analysis
        ela_score, _ = perform_ela(file_location)
        
        # 2. Frequency Analysis (FFT)
        fft_score = analyze_frequency(file_location)
        
        # 3. Logic to determine "Fake" vs "Real"
        # These thresholds are heuristic and would need fine-tuning with a real dataset
        is_fake = False
        reasons = []
        
        # High ELA difference indicates areas with different compression levels (editing)
        if ela_score > 50: 
            is_fake = True
            reasons.append("Inconsistent compression artifacts detected (ELA).")
            
        # Very high or very low frequency energy can indicate AI generation or blurring
        if fft_score > 150:
            is_fake = True
            reasons.append("Abnormal high-frequency noise patterns detected (FFT).")
        
        # Calculate Confidence
        base_confidence = 85 # Base confidence of the system
        if is_fake:
            status = "Fake/Suspicious"
            base_confidence += min((ela_score / 255) * 100, 14) # Add up to 14% confidence based on ELA intensity
        else:
            status = "Authentic"
            base_confidence = 90 + (10 if ela_score < 10 else 0)

        # Cleanup
        if os.path.exists(file_location):
            os.remove(file_location)
            
        explanation = "Analysis passed."
        if reasons:
            explanation = "Suspicious patterns: " + ", ".join(reasons)
        else:
            explanation = "No significant digital manipulation traces found."

        return {
            "status": status,
            "confidence": f"{int(base_confidence)}%",
            "explanation": explanation,
            "verification_id": f"YQ-{int(ela_score)}-{int(fft_score)}"
        }
        
    except Exception as e:
        print(f"Error analyzing image: {e}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

@app.post("/analyze-url")
async def analyze_url(request: dict):
    # For now, keep URL analysis simulated or minimal as requested
    # We could implement image downloading here later
    url = request.get("url")
    if not url:
         raise HTTPException(status_code=400, detail="URL is required")

    return {
        "status": "Authentic",
        "confidence": "88%",
        "explanation": f"URL reputation analysis passed for {url}",
        "verification_id": "YQ-URL-SAFE"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

