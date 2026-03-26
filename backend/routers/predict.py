"""
predict.py — /api/predict endpoint.
"""

from fastapi import APIRouter, UploadFile, File, HTTPException
from services import model_service

router = APIRouter()


@router.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Lütfen bir görüntü dosyası yükleyin.")

    image_bytes = await file.read()
    try:
        result = model_service.predict(image_bytes)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model hatası: {str(e)}")

    return result
