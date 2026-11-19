# Build from the repo root:  docker build -f client/Dockerfile --build-arg VITE_API_URL=http://localhost:3000 .
FROM node:22-alpine AS build
WORKDIR /src/client
COPY client/package*.json ./
RUN npm install --no-fund --no-audit
COPY shared /src/shared
COPY client ./
ARG VITE_API_URL=http://localhost:3000
ENV VITE_API_URL=$VITE_API_URL
RUN npm run build

FROM nginx:1.27-alpine
COPY --from=build /src/client/dist /usr/share/nginx/html
# ONNX models are mounted at runtime so they can be updated without a rebuild
RUN printf 'server {\n listen 80;\n root /usr/share/nginx/html;\n types { application/wasm wasm; application/octet-stream onnx; }\n location / { try_files $uri /index.html; }\n location /sw.js { add_header Cache-Control "no-cache"; }\n}\n' > /etc/nginx/conf.d/default.conf
EXPOSE 80
