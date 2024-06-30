import os
from ultralytics import YOLOv10
import time
from tqdm import tqdm

import cv2


def process_image(img):
    img_out = cv2.cvtColor(img.plot(), cv2.COLOR_BGR2RGB)
    return img_out


class HelmetDetectionModel:
    def __init__(self, model_path):
        self.model = YOLOv10(model_path)

    def predict(self, image_path):
        return self.model(source=image_path)[0]


def load_model(model_path):
    return HelmetDetectionModel(model_path)


def run_inference_single(model, image_path):
    pred = model.predict(image_path)
    return pred
