# download_models.py

from transformers import AutoModel, AutoTokenizer

MODELS = {
    "ENGLISH_TRANSFORMER_MODEL": "unitary/toxic-bert",
    "INDIC_TRANSFORMER_MODEL": "Hate-speech-CNERG/indic-abusive-allInOne-MuRIL",
    "LANGUAGE_DETECT_MODEL": "ZheYu03/xlm-r-langdetect-model",
}

for name, model_name in MODELS.items():
    print(f"Downloading {name}: {model_name}")
    try:
        AutoModel.from_pretrained(model_name)
        AutoTokenizer.from_pretrained(model_name)
        print(f"✅ Finished downloading {model_name}")
    except Exception as e:
        print(f"❌ Failed to download {model_name}: {e}")
