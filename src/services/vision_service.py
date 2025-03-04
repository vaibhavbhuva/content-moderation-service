from google.cloud import vision
import os
from src.core.config import settings

os.environ["GOOGLE_APPLICATION_CREDENTIALS"]=settings.GOOGLE_APPLICATION_CREDENTIALS

class VisionService:
    def __init__(self):
        self.client = vision.ImageAnnotatorClient()

    def detect_safe_search(self, content: bytes):
        image = vision.Image(content=content)
        return self.client.safe_search_detection(image=image)