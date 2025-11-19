FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY main.py /app/main.py

EXPOSE 7860

# En docker-compose apuntamos BACKEND_URL a "http://backend:8000"
ENV BACKEND_URL=http://backend:8000

CMD ["python", "main.py"]
