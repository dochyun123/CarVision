"""FastAPI 서버: 이미지 전처리 + ONNX 추론"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from PIL import Image
import io
import numpy as np
import onnxruntime as ort
import albumentations as A
from albumentations.pytorch import ToTensorV2
import yaml
from rembg import remove, new_session
import os
import time

# preprocess.py에서 crop 함수 import
from preprocess import crop
from inference import class_names

app = FastAPI(title="Car Model Prediction API")

# CORS 설정
import os

allowed_origins_env = os.getenv(
    "ALLOWED_ORIGINS",
    "https://carvisionwebsite.up.railway.app,http://localhost:5173,http://localhost:5174,http://localhost:5175,http://localhost:5176,http://localhost",
)

ALLOWED_ORIGINS = [
    origin.strip() for origin in allowed_origins_env.split(",") if origin.strip()
]

print("✅ Loaded ALLOWED_ORIGINS:", ALLOWED_ORIGINS)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
    allow_credentials=True,
)

print("ALLOWED_ORIGINS =", ALLOWED_ORIGINS)


# config 로드
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Transform (inference.py와 동일)
transform = A.Compose(
    [
        A.Resize(384, 384),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
)

# ONNX 세션 (전역으로 한 번만 로드)
print("=" * 70)
print("ONNX 모델 로딩 중...")
onnx_session = ort.InferenceSession("Bestefficientnet.onnx")
print("✓ ONNX 모델 로드 완료")
print(
    f"입력: {onnx_session.get_inputs()[0].name}, shape: {onnx_session.get_inputs()[0].shape}"
)
print(
    f"출력: {onnx_session.get_outputs()[0].name}, shape: {onnx_session.get_outputs()[0].shape}"
)
print("=" * 70 + "\n")

# Rembg 세션 (전역으로 한 번만 로드)
print("=" * 70)
print("🔥 Rembg 세션 로딩 중 (cold start)...")
rembg_session = new_session("u2net")
print("✅ Rembg 세션 로드 완료 (메모리에 캐시됨)")
print("=" * 70 + "\n")


@app.get("/")
async def root():
    """루트 엔드포인트"""
    return {
        "message": "Car Model Prediction API",
        "endpoints": {
            "GET /": "API 정보",
            "GET /health": "헬스 체크",
            "POST /predict": "이미지 예측 (multipart/form-data, field: image)",
        },
    }


@app.get("/health")
async def health():
    """헬스 체크"""
    return {
        "status": "healthy",
        "model": "Bestefficientnet.onnx",
        "num_classes": len(class_names),
    }


@app.get("/default-image")
async def get_default_image():
    """기본 이미지(DefaultTucson.jpg) 제공"""
    # 현재 디렉토리에서 DefaultTucson.jpg 찾기
    image_path = "DefaultTucson.jpg"

    if not os.path.exists(image_path):
        # car-finder/public 폴더에서 찾기
        image_path = os.path.join("car-finder", "public", "DefaultTucson.jpg")

    if not os.path.exists(image_path):
        raise HTTPException(
            status_code=404, detail="DefaultTucson.jpg 파일을 찾을 수 없습니다."
        )

    # FileResponse 사용 (CORS 미들웨어가 자동으로 헤더 추가)
    return FileResponse(image_path, media_type="image/jpeg")


@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    """
    이미지 업로드 → 전처리 (배경 제거 + 크롭) → ONNX 추론 → Top-5 결과 반환

    Parameters:
    - image: 업로드된 이미지 파일

    Returns:
    - predictions: Top-5 예측 결과
    """
    try:
        tstart = time.time()
        # 1. 이미지 읽기
        print("1. 이미지 수신 중...")
        image_bytes = await image.read()
        pil_image = Image.open(io.BytesIO(image_bytes))
        print(f"   ✓ 이미지 크기: {pil_image.size}, 모드: {pil_image.mode}")
        t1 = time.time()
        print(f"   ✓ 이미지 수신 완료: {t1 - tstart:.2f}초")

        # 2. 배경 제거 (캐시된 세션 사용)
        print("2. 배경 제거 중 (캐시된 세션 사용)...")
        removed_bg = remove(pil_image, session=rembg_session)
        if isinstance(removed_bg, bytes):
            removed_bg = Image.open(io.BytesIO(removed_bg))
        t2 = time.time()
        print(f"   ✓ 배경 제거 완료: {removed_bg.size}, {removed_bg.mode}")
        print(f"   ✓ 배경 제거 소요 시간: {t2 - t1:.2f}초 (캐시된 세션)")

        # 3. 자동 크롭 (여백 제거)
        print("3. 자동 크롭 중...")
        cropped = crop(removed_bg)
        t3 = time.time()
        print(f"   ✓ 크롭 완료: {cropped.size}")
        print(f"   ✓ 크롭 소요 시간: {t3 - t2:.2f}초")

        # 4. RGB 변환
        print("4. RGB 변환 중...")
        if cropped.mode == "RGBA":
            # 투명 배경을 흰색으로
            bg = Image.new("RGB", cropped.size, (255, 255, 255))
            bg.paste(cropped, mask=cropped.split()[3])
            cropped = bg
        elif cropped.mode != "RGB":
            cropped = cropped.convert("RGB")
        t4 = time.time()
        print(f"   ✓ RGB 변환 완료: {cropped.mode}")
        print(f"   ✓ RGB 변환 소요 시간: {t4 - t3:.2f}초")

        # 5. Transform 적용
        print("5. Transform 적용 중...")
        image_np = np.array(cropped)
        transformed = transform(image=image_np)
        image_tensor = transformed["image"]

        # numpy로 변환 및 배치 차원 추가
        image_tensor_np = image_tensor.numpy()
        image_tensor_np = np.expand_dims(image_tensor_np, axis=0).astype(np.float32)
        t5 = time.time()
        print(f"   ✓ Tensor shape: {image_tensor_np.shape}")
        print(f"   ✓ Transform 소요 시간: {t5 - t4:.2f}초")

        # 6. ONNX 추론
        print("6. ONNX 추론 중...")
        input_name = onnx_session.get_inputs()[0].name
        output_name = onnx_session.get_outputs()[0].name

        outputs = onnx_session.run([output_name], {input_name: image_tensor_np})
        logits = outputs[0][0]
        t6 = time.time()
        print(f"   ✓ Logits shape: {logits.shape}")
        print(f"   ✓ ONNX 추론 소요 시간: {t6 - t5:.2f}초")

        # 7. Softmax 적용
        print("7. Softmax 적용 중...")
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / np.sum(exp_logits)


        # 8. Top-5 추출
        print("8. Top-5 추출 중...")
        top5_indices = np.argsort(probs)[::-1][:5]
        predictions = []

        for rank, idx in enumerate(top5_indices, 1):
            predictions.append(
                {
                    "rank": rank,
                    "model": class_names[idx],
                    "confidence": float(probs[idx] * 100),
                }
            )
            print(f"   {rank}. {class_names[idx]}: {probs[idx]*100:.2f}%")
        tlast = time.time()
        print(f"   ✓ Top-5 추출 소요 시간: {tlast - t6:.2f}초")
        print(f"{'='*70}\n")

        return {
            "success": True,
            "top_model": predictions[0]["model"],
            "top_confidence": predictions[0]["confidence"],
            "predictions": predictions,
        }

    except Exception as e:
        print(f"\n✗ 에러 발생: {str(e)}")
        import traceback

        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    print("\n" + "=" * 70)
    print("FastAPI 서버 시작")
    print("=" * 70)
    print("URL: http://localhost:8000")
    print("Docs: http://localhost:8000/docs")
    print("=" * 70 + "\n")

    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
