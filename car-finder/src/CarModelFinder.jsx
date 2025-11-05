import React, { useState, useRef } from 'react';
import { Upload, Car, Loader2, CheckCircle, Camera, Image as ImageIcon } from 'lucide-react';

const CarModelFinder = () => {
  // 환경 변수에서 API URL 가져오기 (기본값: localhost)
  const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

  const [image, setImage] = useState(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [result, setResult] = useState(null);
  const [dragActive, setDragActive] = useState(false);
  const [elapsed, setElapsed] = useState(0);
  const fileInputRef = useRef(null);
  const cameraInputRef = useRef(null);

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFile(e.dataTransfer.files[0]);
    }
  };

  const handleFileInput = (e) => {
    if (e.target.files && e.target.files[0]) {
      handleFile(e.target.files[0]);
    }
  };

  const handleFile = (file) => {
    if (file.type.startsWith('image/')) {
      const reader = new FileReader();
      reader.onload = (e) => {
        setImage(e.target.result);
        setResult(null);
      };
      reader.readAsDataURL(file);
    }
  };

  // FastAPI 서버로 이미지 분석 요청
  const analyzeImage = async () => {
    if (!image) {
      alert('먼저 이미지를 업로드해주세요.');
      return;
    }

    setIsAnalyzing(true);
    setResult(null);
    setElapsed(0);

    const interval = setInterval(() => {
      setElapsed(prev => prev + 1);
    }, 100);

    try {
      console.log('FastAPI 서버에 요청 중...');

      // 이미지를 Blob으로 변환
      const response = await fetch(image);
      const blob = await response.blob();

      // FormData 생성
      const formData = new FormData();
      formData.append('image', blob, 'image.jpg');

      // FastAPI 서버로 POST 요청
      const apiResponse = await fetch(`${API_URL}/predict`, {
        method: 'POST',
        body: formData,
      });

      if (!apiResponse.ok) {
        throw new Error(`HTTP error! status: ${apiResponse.status}`);
      }

      const data = await apiResponse.json();
      console.log('✓ 서버 응답:', data);

      clearInterval(interval);

      // 결과 설정
      const confidences = data.predictions.map(pred => ({
        model: pred.model,
        confidence: pred.confidence.toFixed(2)
      }));

      setResult({
        topModel: data.top_model,
        confidences: confidences
      });
      setIsAnalyzing(false);
      setElapsed(0);

      console.log('✓ 분석 완료:', data.top_model);

    } catch (error) {
      clearInterval(interval);
      console.error('분석 실패:', error);
      alert(`이미지 분석 중 오류가 발생했습니다: ${error.message}\n\nFastAPI 서버가 실행 중인지 확인해주세요. (${API_URL})`);
      setIsAnalyzing(false);
      setElapsed(0);
    }
  };

  const useDefaultImage = async () => {
    try {
      console.log('기본 이미지 로딩 중...');

      // API 서버에서 기본 이미지 가져오기
      const response = await fetch(`${API_URL}/default-image`);
      if (!response.ok) {
        throw new Error('이미지를 찾을 수 없습니다.');
      }
      const blob = await response.blob();
      const url = URL.createObjectURL(blob);
      setImage(url);
      setResult(null);
      console.log('✓ 기본 이미지 로드 완료');

    } catch (error) {
      console.error('기본 이미지 로드 실패:', error);
      alert(`DefaultTucson.jpg 파일을 찾을 수 없습니다.\nFastAPI 서버가 실행 중인지 확인해주세요. (${API_URL})`);
    }
  };

  return (
    <div className="min-h-screen bg-gray-50 p-6">
      <div className="max-w-2xl mx-auto">
        {/* 헤더 */}
        <div className="text-center mb-8 pt-6">
          <div className="inline-flex items-center justify-center w-16 h-16 bg-blue-500 rounded-2xl mb-4">
            <Car className="w-8 h-8 text-white" />
          </div>
          <h1 className="text-3xl font-bold text-gray-900 mb-2">자동차 모델 찾기</h1>
          <p className="text-gray-600">이미지를 업로드하면 AI가 자동차 모델을 분석해요</p>
        </div>

        {/* 메인 컨텐츠 */}
        <div className="bg-white rounded-3xl shadow-sm p-6 mb-4">
          {/* 이미지 업로드 영역 */}
          <div
            className={`relative border-2 border-dashed rounded-2xl p-10 mb-5 transition-all cursor-pointer ${
              dragActive 
                ? 'border-blue-500 bg-blue-50' 
                : 'border-gray-300 hover:border-gray-400 bg-gray-50'
            }`}
            onDragEnter={handleDrag}
            onDragLeave={handleDrag}
            onDragOver={handleDrag}
            onDrop={handleDrop}
            onClick={() => fileInputRef.current?.click()}
          >
            <input
              ref={fileInputRef}
              type="file"
              accept="image/*"
              onChange={handleFileInput}
              className="hidden"
            />
            <input
              ref={cameraInputRef}
              type="file"
              accept="image/*"
              capture="environment"
              onChange={handleFileInput}
              className="hidden"
            />

            {image ? (
              <div className="flex flex-col items-center">
                <img 
                  src={image} 
                  alt="Uploaded car" 
                  className="max-h-72 rounded-xl shadow-md mb-3"
                />
                <p className="text-gray-500 text-sm">다른 이미지를 업로드하려면 클릭하세요</p>
              </div>
            ) : (
              <div className="flex flex-col items-center text-center">
                <div className="w-16 h-16 bg-blue-100 rounded-full flex items-center justify-center mb-4">
                  <Upload className="w-8 h-8 text-blue-500" />
                </div>
                <p className="text-lg font-semibold text-gray-900 mb-1">이미지를 올려주세요</p>
                <p className="text-gray-500 text-sm hidden md:block">드래그 앤 드롭 또는 클릭하여 파일 선택</p>
                <p className="text-gray-500 text-sm md:hidden">아래 버튼으로 사진 선택</p>
              </div>
            )}
          </div>

          {/* 버튼 영역 */}
          <div className="space-y-3">
            {/* 모바일 전용 카메라/갤러리 버튼 */}
            {!image && (
              <div className="grid grid-cols-2 gap-3 md:hidden">
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    cameraInputRef.current?.click();
                  }}
                  className="bg-green-500 hover:bg-green-600 active:bg-green-700 text-white font-bold py-4 px-6 rounded-xl transition-all flex items-center justify-center gap-2 shadow-sm"
                >
                  <Camera className="w-5 h-5" />
                  카메라 촬영
                </button>
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    fileInputRef.current?.click();
                  }}
                  className="bg-purple-500 hover:bg-purple-600 active:bg-purple-700 text-white font-bold py-4 px-6 rounded-xl transition-all flex items-center justify-center gap-2 shadow-sm"
                >
                  <ImageIcon className="w-5 h-5" />
                  갤러리 선택
                </button>
              </div>
            )}

            <div className="flex gap-3">
              <button
                onClick={analyzeImage}
                disabled={isAnalyzing || !image}
                className="flex-1 bg-blue-500 hover:bg-blue-600 active:bg-blue-700 disabled:bg-gray-300 disabled:cursor-not-allowed text-white font-bold py-4 px-6 rounded-xl transition-all flex items-center justify-center gap-2 shadow-sm"
              >
                {isAnalyzing ? (
                  <>
                    <Loader2 className="w-5 h-5 animate-spin" />
                    분석 중 {(elapsed / 10).toFixed(1)}초
                  </>
                ) : (
                  '자동차 찾기'
                )}
              </button>

              {!image && (
                <button
                  onClick={useDefaultImage}
                  className="bg-gray-100 hover:bg-gray-200 active:bg-gray-300 text-gray-700 font-bold py-4 px-6 rounded-xl transition-all hidden md:block"
                >
                  기본 이미지
                </button>
              )}
            </div>
          </div>
        </div>

        {/* 결과 표시 */}
        {result && (
          <div className="bg-white rounded-3xl shadow-sm p-6 animate-fadeIn">
            <div className="flex items-center gap-2 mb-5 pb-5 border-b border-gray-100">
              <div className="w-10 h-10 bg-green-100 rounded-full flex items-center justify-center">
                <CheckCircle className="w-6 h-6 text-green-500" />
              </div>
              <h2 className="text-2xl font-bold text-gray-900">정답!</h2>
            </div>

            <div className="mb-6 p-5 bg-blue-50 rounded-2xl">
              <p className="text-sm font-medium text-blue-600 mb-1">인식된 모델</p>
              <p className="text-2xl font-bold text-gray-900">{result.topModel}</p>
            </div>

            <div>
              <h3 className="text-lg font-bold text-gray-900 mb-4">신뢰도 순위</h3>
              <div className="space-y-3">
                {result.confidences.map((item, index) => (
                  <div key={index} className="bg-gray-50 rounded-xl p-4">
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-gray-700 font-semibold text-sm">
                        {index + 1}. {item.model}
                      </span>
                      <span className="text-blue-500 font-bold text-sm">{item.confidence}%</span>
                    </div>
                    <div className="w-full bg-gray-200 rounded-full h-2 overflow-hidden">
                      <div 
                        className="bg-blue-500 h-full rounded-full transition-all duration-1000"
                        style={{ width: `${item.confidence}%` }}
                      />
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default CarModelFinder;