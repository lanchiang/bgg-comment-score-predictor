# @app.post("/predict")

from fastapi import APIRouter
from pydantic import BaseModel
import torch

from ..utils import logs

from transformers import AutoTokenizer, AutoModelForSequenceClassification

log = logs.get_logger()

predict_router = APIRouter(include_in_schema=True, tags=["predict"])

tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained("distilbert/distilbert-base-uncased")

class TextRequest(BaseModel):
    text: str
    max_length: int = 512

@predict_router.post(
    path='/predict'
)
def predict(request: TextRequest):
    inputs = tokenizer(request.text, return_tensors="pt", max_length=request.max_length, truncation=True)
    model.eval()

    with torch.no_grad():
        outputs = model(**inputs)
    predicted_cls_idx = torch.argmax(outputs.logits, dim=-1).item()
    return {"label": predicted_cls_idx}