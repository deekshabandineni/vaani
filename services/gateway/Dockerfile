# Build from the repo root:  docker build -f services/gateway/Dockerfile .
FROM node:22-alpine
WORKDIR /srv/services/gateway
COPY services/gateway/package*.json ./
RUN npm ci --omit=dev
COPY services/gateway/src src
COPY shared /srv/shared
ENV SHARED_DIR=/srv/shared PORT=3000 NODE_ENV=production
USER node
EXPOSE 3000
HEALTHCHECK CMD wget -qO- http://localhost:3000/health || exit 1
CMD ["node", "src/server.js"]
