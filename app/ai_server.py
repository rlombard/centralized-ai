from fastapi import FastAPI, UploadFile, HTTPException
from transformers import BlipProcessor, BlipForConditionalGeneration, AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForTokenClassification, pipeline
from torchvision import models, transforms
from langdetect import detect
from keybert import KeyBERT
from pydantic import BaseModel
from PIL import Image
import torch
import cv2
import numpy as np
import io
import logging

# Logging configuration
logging.basicConfig(level=logging.INFO)

# Define input models
class SummarizationInput(BaseModel):
    text: str

class TextAnalysisInput(BaseModel):
    text: str

# Initialize FastAPI app
app = FastAPI()

# Global variables for models
text_model = None
tokenizer = None
image_model = None
blip_processor = None
blip_model = None
resnet_model = None
imagenet_classes = None
ner_pipeline = None
sentiment_pipeline = None
kw_model = None

# Preprocessing transformations for ResNet
preprocess = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

@app.on_event("startup")
async def load_models():
    """
    Load all required models during startup.
    """
    global text_model, tokenizer, image_model, blip_processor, blip_model, resnet_model, imagenet_classes, ner_pipeline, sentiment_pipeline, kw_model

    logging.info("Loading text summarization model...")
    tokenizer = AutoTokenizer.from_pretrained("./app/models/t5-small")
    text_model = AutoModelForSeq2SeqLM.from_pretrained("./app/models/t5-small")

    logging.info("Loading YOLOv5 image tagging model...")
    image_model = torch.hub.load('ultralytics/yolov5', 'custom', path='./app/models/yolov5s.pt')

    logging.info("Loading BLIP model for image captioning...")
    blip_processor = BlipProcessor.from_pretrained("./app/models/blip")
    blip_model = BlipForConditionalGeneration.from_pretrained("./app/models/blip")

    logging.info("Loading ResNet model for image classification...")
    resnet_model = models.resnet18(pretrained=True)
    resnet_model.eval()

    logging.info("Loading ImageNet class labels...")
    with open("./app/models/imagenet_classes.txt") as f:
        imagenet_classes = [line.strip() for line in f]

    logging.info("Loading NER pipeline...")
    ner_model = AutoModelForTokenClassification.from_pretrained("./app/models/ner")
    ner_tokenizer = AutoTokenizer.from_pretrained("./app/models/ner")
    ner_pipeline = pipeline("ner", model=ner_model, tokenizer=ner_tokenizer)

    logging.info("Loading sentiment analysis pipeline...")
    sentiment_pipeline = pipeline("sentiment-analysis")

    logging.info("Loading KeyBERT model for keyword extraction...")
    kw_model = KeyBERT()

@app.middleware("http")
async def log_requests(request, call_next):
    """
    Middleware to log incoming requests and responses.
    """
    logging.info(f"Request: {request.method} {request.url}")
    response = await call_next(request)
    logging.info(f"Response status: {response.status_code}")
    return response

@app.get("/health")
async def health():
    """
    Health check endpoint.
    """
    return {"status": "ok"}

@app.post("/tag-image")
async def tag_image(file: UploadFile):
    """
    Tag objects in an image using YOLOv5.
    """
    try:
        image_data = np.frombuffer(await file.read(), np.uint8)
        image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
        results = image_model(image)
        tags = results.pandas().xyxy[0][["name", "confidence"]].to_dict(orient="records")
        return {"tags": tags}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

@app.post("/describe-image")
async def describe_image(file: UploadFile):
    """
    Generate a caption for an image using BLIP.
    """
    try:
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        inputs = blip_processor(images=image, return_tensors="pt")
        outputs = blip_model.generate(**inputs)
        description = blip_processor.decode(outputs[0], skip_special_tokens=True)
        return {"description": description}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

@app.post("/classify-image")
async def classify_image(file: UploadFile):
    """
    Classify an image using ResNet.
    """
    try:
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        input_tensor = preprocess(image).unsqueeze(0)
        with torch.no_grad():
            outputs = resnet_model(input_tensor)
        _, predicted_idx = outputs.max(1)
        predicted_class = imagenet_classes[predicted_idx.item()]
        return {"class": predicted_class}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

@app.post("/analyze-text")
async def analyze_text(input: TextAnalysisInput):
    """
    Perform comprehensive text analysis including:
    - Sentiment analysis
    - Named entity recognition (NER)
    - Language detection
    - Keyword extraction
    - Text summarization
    """
    try:
        text = input.text

        # Sentiment Analysis
        sentiment = sentiment_pipeline(text)

        # Named Entity Recognition (NER)
        entities = ner_pipeline(text)

        # Language Detection
        language = detect(text)

        # Keyword Extraction
        keywords = kw_model.extract_keywords(text, top_n=5)

        # Text Summarization
        inputs = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=512, truncation=True)
        outputs = text_model.generate(inputs, max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
        summary = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Combine results
        result = {
            "sentiment": sentiment,
            "entities": [{"text": e['word'], "type": e['entity']} for e in entities],
            "language": language,
            "keywords": [kw[0] for kw in keywords],
            "summary": summary
        }

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing text: {str(e)}")
