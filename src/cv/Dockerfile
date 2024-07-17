FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu20.04 AS builder

WORKDIR /cv
COPY ./requirements.txt .

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y python3.9 python3.9-distutils git wget curl libgl1-mesa-glx libglib2.0-0 && \ 
    curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python3.9 get-pip.py && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    pip install --no-cache-dir -r requirements.txt 


FROM builder AS stage

WORKDIR /cv

RUN gdown https://drive.google.com/uc?id=1vqg9n6bsdHEiJwusYdy-AYlEeIZUn0XS
RUN gdown https://drive.google.com/uc?id=1if4EWNwiYQyxX7rzef9pWgiamTgY4ZJR

COPY . .

EXPOSE 3000

CMD ["uvicorn", "server:app", "--reload", "--host", "0.0.0.0", "--port", "3000"]