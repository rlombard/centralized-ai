import os
import shutil
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AutoModelForTokenClassification,
    BlipProcessor,
    BlipForConditionalGeneration,
)
import torch

# Paths to save the models
MODEL_BASE_PATH = "./app/models"
os.makedirs(MODEL_BASE_PATH, exist_ok=True)

# Models to download
MODELS = {
    "t5-small": {"type": "seq2seq", "path": f"{MODEL_BASE_PATH}/t5-small"},
    "ner": {"type": "token-classification", "path": f"{MODEL_BASE_PATH}/ner"},
    "blip": {"type": "blip", "path": f"{MODEL_BASE_PATH}/blip"},
    "yolov5s": {"type": "torchhub", "path": f"{MODEL_BASE_PATH}/yolov5s.pt"},
    "imagenet_classes": {"type": "file", "path": f"{MODEL_BASE_PATH}/imagenet_classes.txt"},
}

# Download T5 model for summarization
def download_t5():
    print("Downloading T5-small model for summarization...")
    AutoTokenizer.from_pretrained("t5-small").save_pretrained(MODELS["t5-small"]["path"])
    AutoModelForSeq2SeqLM.from_pretrained("t5-small").save_pretrained(MODELS["t5-small"]["path"])

# Download NER model
def download_ner():
    print("Downloading NER model...")
    AutoTokenizer.from_pretrained("dbmdz/bert-large-cased-finetuned-conll03-english").save_pretrained(MODELS["ner"]["path"])
    AutoModelForTokenClassification.from_pretrained("dbmdz/bert-large-cased-finetuned-conll03-english").save_pretrained(MODELS["ner"]["path"])

# Download BLIP model
def download_blip():
    print("Downloading BLIP model...")
    BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base").save_pretrained(MODELS["blip"]["path"])
    BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").save_pretrained(MODELS["blip"]["path"])

# Download YOLOv5 model
def download_yolov5():
    print("Downloading YOLOv5 model...")
    # Load the YOLOv5 model from Torch Hub
    model = torch.hub.load("ultralytics/yolov5", "yolov5s", pretrained=True)

    # Ensure the models directory exists
    os.makedirs(os.path.dirname(MODELS["yolov5s"]["path"]), exist_ok=True)

    # Save the model's state dictionary to the desired path
    model_state_path = MODELS["yolov5s"]["path"]
    torch.save(model.state_dict(), model_state_path)
    print(f"YOLOv5 model saved to {model_state_path}")

    # Move the yolov5s.pt model to the proper directory (if not already placed)
    if os.path.exists("yolov5s.pt"):
        print("Moving YOLOv5 model to the correct directory...")
        shutil.move("yolov5s.pt", model_state_path)
        print(f"YOLOv5 model moved to {model_state_path}")

# Download ImageNet class labels
def download_imagenet_classes():
    print("Downloading ImageNet class labels...")
    url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
    import requests
    response = requests.get(url)
    with open(MODELS["imagenet_classes"]["path"], "w") as f:
        f.write(response.text)

# Main function to download all models
def download_all_models():
    download_t5()
    download_ner()
    download_blip()
    download_yolov5()
    download_imagenet_classes()
    print("\nAll models downloaded successfully!")

if __name__ == "__main__":
    download_all_models()
