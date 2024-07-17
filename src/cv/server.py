import json
import base64
from PIL import Image

import numpy as np
import cv2
from fastapi import FastAPI
from pydantic import BaseModel
from io import BytesIO
from prometheus_fastapi_instrumentator import Instrumentator
from starlette.middleware.cors import CORSMiddleware
from typing import List
from surya.model.detection.model import load_model as load_detection_model, load_processor as load_detection_processor
from surya.model.recognition.model import load_model
from surya.model.recognition.processor import load_processor
from sahi import AutoDetectionModel
import uuid
from utils import get_filtred_boxes_coco, process_yolo_output, letters_list_transform, get_estimates
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

langs = ["ru"]
det_processor, det_model = load_detection_processor(), load_detection_model()
rec_model, rec_processor = load_model(), load_processor()

threshhold = 0.2
yolo_model = AutoDetectionModel.from_pretrained(
    model_type='yolov8',
    model_path='epoch44.pt',
    confidence_threshold=threshhold
)
logger.info("Можете начинать отправлять схемы")
app = FastAPI()

origins = [
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

Instrumentator().instrument(app).expose(app)


class ImageInput(BaseModel):
    image: str


class DataOutput(BaseModel):
    article: List[str]
    name: List[str]
    amount: List[int]
    price: List[int]
    sum: List[int]


def decode_base64_to_image(base64_str):
    image_data = base64.b64decode(base64_str)
    return Image.open(BytesIO(image_data))


@app.get("/health-check")
def healthcheck():
    return {
        'status': 'OK'
    }


@app.post("/process_image", response_model=DataOutput)
def process_image_endpoint(input: ImageInput):

    image = decode_base64_to_image(input.image)
    if image.mode == 'RGBA':
        background = Image.new('RGB', image.size, (255, 255, 255))

        rgb_image = Image.alpha_composite(background.convert('RGBA'), image).convert('RGB')
        image = rgb_image

    unique_filename = f"{uuid.uuid4()}.png"
    image.save(unique_filename)
    yolo_output = get_filtred_boxes_coco(unique_filename, yolo_model, display=False)
    res = process_yolo_output(yolo_output, unique_filename, langs, det_processor,
                              det_model, rec_model, rec_processor, display=False)
    res_tr = letters_list_transform(res)
    logger.info(f'res: {res}')
    logger.info(f'res_tr: {res_tr}')
    search_res = get_estimates(res_tr)


    os.remove(unique_filename)

    return search_res
