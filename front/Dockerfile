FROM python:3.11-slim

WORKDIR /app

COPY front/requirements.txt ./front/requirements.txt
RUN pip install --no-cache-dir -r front/requirements.txt

COPY front/ ./front/
COPY training/data/ ./training/data/
COPY training/figures/ ./training/figures/
COPY training/metrics/ ./training/metrics/
COPY API/tests/test_values.json ./API/tests/test_values.json

EXPOSE 8501

CMD ["streamlit", "run", "front/app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]