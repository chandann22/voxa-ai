FROM python:3.11-slim

WORKDIR /app

COPY backend/ ./backend/
COPY docs/ ./docs/
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

WORKDIR /app/backend

EXPOSE 7860

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
