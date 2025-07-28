import logging
import time
import transformers

# Set logging level to INFO or DEBUG for more verbosity
logging.basicConfig(level=logging.DEBUG)
# or specifically for Hugging Face
logging.getLogger("transformers").setLevel(logging.DEBUG)
logging.getLogger("huggingface_hub").setLevel(logging.DEBUG)
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from transformers import pipeline

# Load model and tokenizer from Hugging Face
model_name = "ZheYu03/xlm-r-langdetect-model"
# model_name  = "./models/xlm-roberta-langdetect"
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForSequenceClassification.from_pretrained(model_name)

classifer= pipeline(
                "text-classification",
                model="ZheYu03/xlm-r-langdetect-model",
                top_k=None
            )  

# Sample text input
text = "thank you"
start_time = time.time()
# inputs = tokenizer(text, return_tensors="pt", truncation=True)
# outputs = model(**inputs)

# # Get the predicted class (language ID)
# predicted_class_id = torch.argmax(outputs.logits, dim=1).item()
classifer(text)
end_time = time.time()
elapsed_time_sec = end_time - start_time
elapsed_time_ms = elapsed_time_sec * 1000
print(f"Detect function time: {elapsed_time_sec:.3f} seconds ({elapsed_time_ms:.0f} ms)")

# Load language labels
id2label = model.config.id2label
predicted_language = id2label[predicted_class_id]

print(f"Detected Language: {predicted_language}")
