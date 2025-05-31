from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import spacy

app = FastAPI()

class AbstractRequest(BaseModel):
    abstracts: List[str]

model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
model.eval()

nlp = spacy.load("en_core_sci_sm")

@app.post("/classify")
def classify(req: AbstractRequest):
    tokens = tokenizer(req.abstracts, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**tokens)
        probs = torch.softmax(outputs.logits, dim=-1)
    preds = torch.argmax(probs, dim=-1).tolist()
    confidences = probs.tolist()
    diseases = [
        list({ent.text for ent in nlp(a).ents if ent.label_.lower() in {"disease", "cancer"}})
        for a in req.abstracts
    ]
    return {
        "predictions": preds,
        "confidences": confidences,
        "diseases": diseases,
    }

