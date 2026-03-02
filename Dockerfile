FROM node:20-slim

WORKDIR /app

COPY package.json handler.js ./

RUN npm install

CMD ["node", "handler.js"]
