FROM python:3.9-slim AS builder 

WORKDIR /cv

COPY ./requirements.txt .

RUN apt-get update && \
    apt-get install -y libgl1-mesa-glx libglib2.0-0 wget && \
    pip install --no-cache-dir -r requirements.txt 

FROM builder AS stage

WORKDIR /cv

RUN gdown https://drive.google.com/uc?id=1vvjDTZDYbwqfwz6nh7-1_ux09o1ZHDGe
RUN gdown https://drive.google.com/uc?id=1XY8aT9LyBKXD-xzli9yVMHeNUTEp4D3e

COPY . .

EXPOSE 3000

CMD ["uvicorn", "server:app", "--reload", "--host", "0.0.0.0", "--port", "3000"]