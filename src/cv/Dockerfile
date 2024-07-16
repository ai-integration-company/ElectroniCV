FROM python:3.9-slim AS builder 

WORKDIR /cv

COPY ./requirements.txt .

RUN apt-get update && \
    apt-get install -y libgl1-mesa-glx libglib2.0-0 wget && \
    pip install --no-cache-dir -r requirements.txt 

RUN gdown https://drive.google.com/uc?id=1Nbb-Lp689T4b0nCEJkI05qpcD13uKImn

FROM builder AS stage

WORKDIR /cv

COPY . .

EXPOSE 3000

CMD ["uvicorn", "server:app", "--reload", "--host", "0.0.0.0", "--port", "3000"]