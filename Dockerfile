FROM runpod/base:0.6.2-cuda12.1.0

RUN apt-get update && apt-get install -y \
    curl \
    python3 \
    make \
    g++ \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY package.json ./
RUN npm install --production --ignore-scripts && \
    npx --yes @mapbox/node-pre-gyp install --fallback-to-build --update-binary --directory node_modules/@tensorflow/tfjs-node-gpu || true

COPY handler.js ./

CMD ["node", "handler.js"]
