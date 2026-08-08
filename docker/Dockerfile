FROM python:3.12-alpine

WORKDIR /app

# Install system dependencies (bash for the shell scripts, procps for process management)
RUN apk add --no-cache bash procps

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8765

CMD ["./docker/start.sh"]
