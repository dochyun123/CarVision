# Frontend Dockerfile for React + Vite
FROM node:20-alpine AS builder

# 작업 디렉토리 설정
WORKDIR /app

# package.json과 package-lock.json 복사
COPY package*.json ./

# 의존성 설치
RUN npm ci

# 소스 코드 복사
COPY . .

# 환경 변수를 빌드 시점에 주입 (Railway에서 설정)
ARG VITE_API_URL
ENV VITE_API_URL=$VITE_API_URL

# 프로덕션 빌드
RUN npm run build

# Nginx를 사용한 프로덕션 서버
FROM nginx:alpine

# 빌드된 파일을 Nginx로 복사
COPY --from=builder /app/dist /usr/share/nginx/html

# Nginx 설정 파일 복사
COPY nginx.conf /etc/nginx/conf.d/default.conf

# 포트 노출
EXPOSE 80

# Nginx 실행
CMD ["nginx", "-g", "daemon off;"]
