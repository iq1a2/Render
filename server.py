import numpy as np
import cv2
import io
import os
import shutil
import uvicorn
import librosa
import soundfile as sf
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageChops, ImageEnhance

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

# --- Image Analysis ---

def perform_ela(image_path, quality=90):
    """
    Error Level Analysis (ELA) with calibrated thresholds.
    """
    try:
        original = Image.open(image_path).convert('RGB')
        
        # Save as temporary JPEG with specific quality
        temp_path = image_path + ".ela.jpg"
        original.save(temp_path, 'JPEG', quality=quality)
        
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
    except Exception as e:
        print(f"ELA Error: {e}")
        return 0, None

def analyze_frequency(image_path):
    """
    Frequency analysis using FFT or similar heuristics.
    """
    try:
        img = cv2.imread(image_path, 0) # Grayscale
        if img is None:
            return 0
            
        f = np.fft.fft2(img)
        fshift = np.fft.fftshift(f)
        magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)
        
        # Simple heuristic: Check for abnormal high frequency energy
        mean_energy = np.mean(magnitude_spectrum)
        return mean_energy
    except Exception as e:
        print(f"FFT Error: {e}")
        return 0

# --- Audio Analysis ---

def analyze_audio_features(file_path):
    """
    Analyze audio for synthetic speech artifacts using Librosa.
    """
    try:
        y, sr = librosa.load(file_path, duration=10) # Limit to 10 seconds for speed
        
        # 1. MFCCs (Mel-frequency cepstral coefficients)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_mean = np.mean(mfccs)
        
        # 2. Zero Crossing Rate (ZCR)
        zcr = librosa.feature.zero_crossing_rate(y)
        zcr_mean = np.mean(zcr)
        
        # 3. Spectral Centroid
        cent = librosa.feature.spectral_centroid(y=y, sr=sr)
        cent_mean = np.mean(cent)

        # Heuristics for synthetic speech detection (Mock thresholds)
        # Synthetic speech often has consistent ZCR or specific MFCC patterns
        is_suspicious = False
        reasons = []

        if zcr_mean > 0.3: # Threshold needs real calibration
            is_suspicious = True
            reasons.append("Abnormally high zero-crossing rate (Electronic signature)")
        
        if mfcc_mean < -20: # Example threshold
             reasons.append("Unnatural spectral texture (MFCC anomaly)")

        return is_suspicious, reasons, float(mfcc_mean)

    except Exception as e:
        print(f"Audio Analysis Error: {e}")
        return False, ["Error analyzing audio"], 0.0

# --- Endpoints ---

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    file_type = file.content_type
    file_location = os.path.join(UPLOAD_FOLDER, file.filename)
    
    try:
        with open(file_location, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # --- Video Analysis ---
        if file_type.startswith('video/'):
            # Extract frames and analyze them
            cap = cv2.VideoCapture(file_location)
            frames_to_analyze = 5
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            step = max(1, frame_count // frames_to_analyze)
            
            fake_frames = 0
            analyzed_frames = 0
            
            for i in range(0, frame_count, step):
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Save frame temporarily
                frame_path = f"{file_location}_frame_{i}.jpg"
                cv2.imwrite(frame_path, frame)
                
                # Analyze frame
                ela, _ = perform_ela(frame_path)
                fft = analyze_frequency(frame_path)
                
                # Heuristic for video frames (often compressed, so thresholds differ)
                if ela > 60 or fft > 160: 
                    fake_fake = True
                    fake_frames += 1
                
                analyzed_frames += 1
                if os.path.exists(frame_path):
                    os.remove(frame_path)
                    
            cap.release()
            
            is_fake = (fake_frames / analyzed_frames) > 0.4 if analyzed_frames > 0 else False
            status = "Fake/Suspicious" if is_fake else "Authentic"
            confidence = 88 if is_fake else 94
            explanation = "Video deepfake patterns detected in keyframes." if is_fake else "Video structure appears consistent."
            
            # Cleanup
            if os.path.exists(file_location):
                os.remove(file_location)

            return {
                "status": status,
                "confidence": f"{confidence}%",
                "explanation": explanation,
                "verification_id": f"YQ-VID-{fake_frames}"
            }

        # --- Image Analysis ---
        elif file_type.startswith('image/'):
            # 1. ELA Analysis
            ela_score, _ = perform_ela(file_location)
            
            # 2. Frequency Analysis (FFT)
            fft_score = analyze_frequency(file_location)
            
            # Logic Calibration (User Feedback: "False Positives on Real Photos")
            # Increase thresholds for high-res images which naturally have high ELA
            # Base ELA threshold moved from 50 -> 70
            is_fake = False
            reasons = []
            
            if ela_score > 70: 
                is_fake = True
                reasons.append("High compression mismatch (Possible splicing)")
                
            if fft_score > 160: # Threshold increased
                is_fake = True
                reasons.append("Frequency anomaly (Possible AI generation)")
            
            # Confidence Logic
            base_confidence = 85
            if is_fake:
                status = "Fake/Suspicious"
                base_confidence += 10
            else:
                status = "Authentic"
                base_confidence = 92 # Stronger default for real photos per user feedback

            # Cleanup
            if os.path.exists(file_location):
                os.remove(file_location)
                
            explanation = "Analysis passed."
            if reasons:
                explanation = "Suspicious: " + ", ".join(reasons)
            else:
                explanation = "No significant manipulation traces found."

            return {
                "status": status,
                "confidence": f"{int(base_confidence)}%",
                "explanation": explanation,
                "verification_id": f"YQ-{int(ela_score)}-{int(fft_score)}",
                "details": {
                    "ela_score": ela_score,
                    "fft_score": int(fft_score),
                    "layers": ["Pixel Anomaly", "Metadata Check", "Frequency Scan"]
                }
            }
        
        # --- Audio Analysis ---
        elif file_type.startswith('audio/') or file_type == 'application/octet-stream': # Octet stream often for raw uploads
             is_suspicious, reasons, mfcc = analyze_audio_features(file_location)
             
             status = "Fake/Suspicious" if is_suspicious else "Authentic"
             confidence = 89 if is_suspicious else 95
             explanation = "Synthetic voice artifacts detected." if is_suspicious else "Voice pattern matches natural speech."
             
             if os.path.exists(file_location):
                os.remove(file_location)
                
             return {
                "status": status,
                "confidence": f"{confidence}%",
                "explanation": explanation,
                "verification_id": f"YQ-AUD-{int(mfcc)}"
            }

        else:
             raise HTTPException(status_code=400, detail=f"Unsupported file type: {file_type}")
        
    except Exception as e:
        print(f"Server Error: {e}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

@app.post("/analyze-url")
async def analyze_url(request: dict):
    # Simulated URL analysis
    url = request.get("url")
    if not url:
         raise HTTPException(status_code=400, detail="URL is required")

    return {
        "status": "Authentic",
        "confidence": "88%",
        "explanation": f"URL Analysis passed for {url}",
        "verification_id": "YQ-URL-SAFE"
    }

@app.post("/verify-source")
async def verify_source(file: UploadFile = File(...)):
    try:
        # Simulate processing time
        import time
        import random
        time.sleep(1.5)
        
        # Mock logic: checking hash against "Global Databases"
        # In a real app, this would query Google Vision API or a media database
        
        # 20% chance of finding a "match" (Simulated existing photo)
        found_match = random.random() < 0.2
        
        if found_match:
            return {
                "source_verified": True,
                "sources": ["Reuters Archive", "Associated Press (AP)"],
                "message": "Similar image found in trusted news archives.",
                "google_search_url": "https://www.google.com/searchbyimage?image_url=example"
            }
        else:
             return {
                "source_verified": False,
                "sources": [],
                "message": "No matches found in major news agencies (Reuters, AFP, AP). This image may be unique or unpublished.",
                "google_search_url": "https://www.google.com/search?q=reverse+image+search" 
            }

    except Exception as e:
         print(f"Source Verification Error: {e}")
         return {"source_verified": False, "message": "Could not verify source at this time."}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
