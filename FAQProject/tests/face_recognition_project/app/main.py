from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from pydantic import BaseModel
from typing import List, Dict, Any
import cv2
import numpy as np
import base64
import os
import uuid
import json
from datetime import datetime
from tests.app import FaceAnalyzer

app = FastAPI(title="Face Recognition API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

face_analyzer = FaceAnalyzer()

UPLOAD_DIR = "uploads"
HISTORY_FILE = "detection_history.json"

os.makedirs(UPLOAD_DIR, exist_ok=True)


def save_detection_history(data: Dict[str, Any]):
    try:
        if os.path.exists(HISTORY_FILE):
            with open(HISTORY_FILE, 'r') as f:
                history = json.load(f)
        else:
            history = []

        history.append({
            'timestamp': datetime.now().isoformat(),
            'face_count': data.get('face_count', 0),
            'faces': data.get('faces', [])
        })

        with open(HISTORY_FILE, 'w') as f:
            json.dump(history[-100:], f)
    except Exception:
        pass


def get_detection_history() -> List[Dict[str, Any]]:
    try:
        if os.path.exists(HISTORY_FILE):
            with open(HISTORY_FILE, 'r') as f:
                return json.load(f)
    except Exception:
        pass
    return []


class DetectRequest(BaseModel):
    image: str


class SearchRequest(BaseModel):
    encoding: List[float]
    threshold: float = 0.6


@app.get("/")
async def root():
    html_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "ui", "mobile.html")
    if os.path.exists(html_path):
        with open(html_path, 'r', encoding='utf-8') as f:
            content = f.read()
            content = content.replace("const API_BASE = 'http://localhost:5000'", "const API_BASE = window.location.origin")
            return HTMLResponse(content=content)

    return JSONResponse({
        "service": "Face Recognition API v2.0",
        "version": "2.0.0",
        "features": [
            "Face Detection",
            "Age/Gender Estimation",
            "Smile Detection",
            "Face Comparison",
            "Real-time Camera Detection"
        ]
    })


@app.get("/api/health")
async def health_check():
    return JSONResponse({
        "status": "healthy",
        "service": "face-recognition-api-v2",
        "version": "2.0.0"
    })


@app.post("/api/detect/base64")
async def detect_base64(request: DetectRequest):
    try:
        image_data = request.image
        if "," in image_data:
            image_data = image_data.split(",")[1]

        image_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image data")

        result = face_analyzer.process_image(image, draw_results=False, detect_features=True)
        save_detection_history(result)

        return JSONResponse({
            "success": True,
            "face_count": result['face_count'],
            "has_face": result['has_face'],
            "faces": result['faces']
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/detect/file")
async def detect_file(file: UploadFile = File(...), draw: bool = Form(False), detect_features: bool = Form(False)):
    try:
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file provided")

        allowed_extensions = {'png', 'jpg', 'jpeg', 'bmp', 'gif'}
        ext = file.filename.rsplit('.', 1)[-1].lower() if '.' in file.filename else ''
        if ext not in allowed_extensions:
            raise HTTPException(status_code=400, detail="File type not allowed")

        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")

        result = face_analyzer.process_image(image, draw_results=draw, detect_features=detect_features)

        filename = f"{uuid.uuid4()}.{ext}"
        filepath = os.path.join(UPLOAD_DIR, filename)

        if draw:
            cv2.imwrite(filepath, image)
            result['result_image'] = f"/uploads/{filename}"

        save_detection_history(result)

        return JSONResponse({
            "success": True,
            "filename": file.filename,
            "face_count": result['face_count'],
            "has_face": result['has_face'],
            "faces": result['faces'],
            "result_image": result.get('result_image')
        })

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/detect/stream")
async def detect_stream(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image")

        result = face_analyzer.process_image(image, draw_results=True, detect_features=True)

        _, buffer = cv2.imwrite('.JPEG', image)
        image_base64 = base64.b64encode(buffer).decode('utf-8')

        return JSONResponse({
            "success": True,
            "image": f"data:image/jpeg;base64,{image_base64}",
            "face_count": result['face_count'],
            "faces": result['faces']
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/compare")
async def compare_faces(face1_enc: List[float], face2_enc: List[float]):
    try:
        encoding1 = np.array(face1_enc)
        encoding2 = np.array(face2_enc)

        similarity = face_analyzer.compare_faces(encoding1, encoding2)

        return JSONResponse({
            "success": True,
            "similarity": similarity,
            "match": similarity >= 0.6
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/history")
async def get_history(limit: int = 10):
    history = get_detection_history()
    return JSONResponse({
        "total": len(history),
        "records": history[-limit:]
    })


@app.get("/api/stats")
async def get_stats():
    history = get_detection_history()
    total_detections = len(history)
    total_faces = sum(record.get('face_count', 0) for record in history)

    return JSONResponse({
        "total_detections": total_detections,
        "total_faces_detected": total_faces,
        "average_faces_per_image": total_faces / total_detections if total_detections > 0 else 0
    })


@app.post("/api/batch-detect")
async def batch_detect(files: List[UploadFile] = File(...)):
    results = []

    for file in files:
        try:
            contents = await file.read()
            nparr = np.frombuffer(contents, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if image is not None:
                result = face_analyzer.process_image(image, draw_results=False, detect_features=True)
                results.append({
                    "filename": file.filename,
                    "success": True,
                    **result
                })
            else:
                results.append({
                    "filename": file.filename,
                    "success": False,
                    "error": "Invalid image"
                })
        except Exception as e:
            results.append({
                "filename": file.filename,
                "success": False,
                "error": str(e)
            })

    return JSONResponse({
        "total": len(files),
        "processed": len(results),
        "results": results
    })


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
