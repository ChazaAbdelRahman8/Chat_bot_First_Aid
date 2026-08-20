FROM nginx:1.27-alpine

COPY chatbot-ui/nginx.conf /etc/nginx/conf.d/default.conf
COPY chatbot-ui/src /usr/share/nginx/html

EXPOSE 80
HEALTHCHECK --interval=10s --timeout=3s --retries=5 \
  CMD wget -q -O /dev/null http://127.0.0.1/ || exit 1
