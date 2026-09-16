from fastapi import FastAPI, UploadFile, File
from ultralytics import YOLO
from PIL import Image
import io

app = FastAPI()

MODEL_PATH = "models/road_damage_yolo11s.pt"

model = YOLO(MODEL_PATH)


@app.get("/")
def root():
    return {"message": "RoadGuard AI Backend Running"}


@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes))

    results = model.predict(
        image,
        conf=0.25,
        imgsz=640
    )

    detections = []

    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])

            detections.append({
                "class_id": class_id,
                "confidence": confidence
            })

    return {
        "success": True,
        "detections": detections
    }