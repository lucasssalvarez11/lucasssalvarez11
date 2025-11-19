FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY . /app

EXPOSE 7860

# En Docker, el backend se llama "backend"
ENV BACKEND_URL=http://backend:8000

CMD ["python", "main.py"]
