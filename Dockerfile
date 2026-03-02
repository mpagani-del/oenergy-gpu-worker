FROM runpod/base:0.6.2-cuda12.1.0

RUN apt-get update && \
    apt-get install -y curl && \
    curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
    apt-get install -y nodejs && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY package.json handler.js ./

RUN npm install

CMD ["node", "handler.js"]
