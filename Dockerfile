FROM node:20-slim

WORKDIR /app

COPY package.json handler.js ./

RUN npm install --omit=optional --ignore-scripts

CMD ["node", "handler.js"]
