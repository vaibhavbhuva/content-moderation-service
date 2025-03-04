import os
from google.cloud import language_v1 as language
from src.core.config import settings

os.environ["GOOGLE_APPLICATION_CREDENTIALS"]=settings.GOOGLE_APPLICATION_CREDENTIALS

class LanguageService:
    def __init__(self):
        self.client = language.LanguageServiceClient()

    def moderate_text(self, text: str):
        document = language.Document(content=text, type_=language.Document.Type.PLAIN_TEXT)
        return self.client.moderate_text(document=document)