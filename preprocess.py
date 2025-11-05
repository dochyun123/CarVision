from rembg import remove
from PIL import Image
import io
import numpy as np


def crop(image):
    """투명한 배경을 자동으로 크롭하는 함수"""
    # 이미지를 numpy 배열로 변환
    img_array = np.array(image)

    # RGBA 이미지인 경우 알파 채널 사용
    if image.mode == 'RGBA':
        alpha = img_array[:, :, 3]
    else:
        # RGB인 경우 검은색(0,0,0)을 배경으로 간주
        alpha = (img_array.sum(axis=2) > 0).astype(np.uint8) * 255

    # 알파 채널에서 0이 아닌(투명하지 않은) 픽셀 찾기
    non_transparent = np.where(alpha > 0)

    if len(non_transparent[0]) == 0:
        # 모든 픽셀이 투명한 경우 원본 반환
        return image

    # 경계 박스 계산
    y_min, y_max = non_transparent[0].min(), non_transparent[0].max()
    x_min, x_max = non_transparent[1].min(), non_transparent[1].max()

    # 크롭
    cropped = image.crop((x_min, y_min, x_max + 1, y_max + 1))

    return cropped


def rmbg(input_path, output_path):
    # 1️⃣ 배경 제거
    img = Image.open(input_path)
    result = remove(img)

    # 2️⃣ PNG 무손실 압축 적용
    buffer = io.BytesIO()
    result.save(buffer, format="PNG", optimize=True, compress_level=9)
    with open(output_path, "wb") as f:
        f.write(buffer.getvalue())


def resize(input_path,output_path, size):
    img = Image.open(input_path)
    img = img.resize(size)
    img.save(output_path)


def main(input_path, output_path):
    img = Image.open(input_path)
    # 배경 제거
    removed_bg = remove(img)

    # 자동 크롭 (여백 제거)
    cropped = crop(removed_bg)

    # 저장
    buffer = io.BytesIO()
    cropped.save(buffer, format="PNG", optimize=True, compress_level=9)
    with open(output_path, "wb") as f:
        f.write(buffer.getvalue())


if __name__ == "__main__":
    # 테스트: 새로운 main() 함수로 이미지 처리
    input_path = "testTucson2.jpg"
    output_path = "Tucson2_cropped.png"
    main(input_path, output_path)
