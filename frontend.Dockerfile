FROM node:20 AS build

WORKDIR /app
COPY eventzilla-front/package*.json /app/
RUN npm install

COPY eventzilla-front /app
RUN npm run build

FROM nginx:alpine
COPY docker/nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/dist/eventzilla-front/browser /usr/share/nginx/html

EXPOSE 80
