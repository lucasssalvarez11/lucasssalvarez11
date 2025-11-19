FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY . /app

EXPOSE 7860

# En docker-compose seteamos BACKEND_URL=http://llm-backend:8000
ENV BACKEND_URL=http://llm-backend:8000

CMD ["python", "main.py"]
