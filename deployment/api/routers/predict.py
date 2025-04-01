from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
import torch

from ..utils import logs

from ..utils.dependencies import get_tokenizer, get_model, get_device

log = logs.get_logger()

predict_router = APIRouter(include_in_schema=True, tags=["predict"])


class TextRequest(BaseModel):
    text: str
    max_length: int = 512,
    return_probabilities: Optional[bool] = False


@predict_router.post(
    path='/predict_rate'
)
def predict(request: TextRequest,
            tokenizer = Depends(get_tokenizer),
            model = Depends(get_model),
            device = Depends(get_device)
            ):
    try:
        texts = [request.text] if isinstance(request.text, str) else request.text

        inputs = tokenizer(
            request.text,
            padding=True,
            return_tensors="pt",
            max_length=request.max_length,
            truncation=True
        ).to(device)

        model.eval()

        with torch.no_grad():
            outputs = model(**inputs)

        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=1)
        predicted_class = torch.argmax(probabilities, dim=-1).cpu().numpy().tolist()
        # predicted_cls_idx = torch.argmax(outputs.logits, dim=-1).item()

        response = {
            'predictions': predicted_class,
            'status': 'success'
        }

        if request.return_probabilities:
            response['probabilities'] = probabilities.numpy().tolist()

        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
