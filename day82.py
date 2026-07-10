# day 82:RoBERTa : Robustly Optimized BERT 
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch


#1. AutoTokenizer/AutoModel automatically loads the right class for any model name

roberta = pipeline("sentiment-analysis",
                   model="cardiffnlp/twitter-roberta-base-sentiment-latest")

texts = [
    "This product is absolutely amazing!",
    "Terrible experience, never buying again",
    "It is okay, nothing special",
]
for text in texts:
    result = roberta(text)[0]
    print(f"  '{text[:40]}' → {result['label']} ({result['score']:.2f})")

# 2. XLNet for Text Classification 
print("XLNet — Zero Shot Classification")
# Using zero-shot since fine-tuning XLNet needs more data/time
xlnet_zs = pipeline("zero-shot-classification",
                    model="typeform/distilbart-mnli-12-3")

result = xlnet_zs(
    "The new iPhone has an incredible camera and fast processor",
    candidate_labels=["technology", "sports", "politics", "food"]
)
print("Zero-shot results:")
for label, score in zip(result['labels'], result['scores']):
    print(f"  {label:<12} {score:.3f}")

# 3. Compare BERT vs RoBERTa 
print("BERT vs RoBERTa — Direct Comparison")
bert_pipe    = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
roberta_pipe = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest")

test_text = "The movie was not bad at all, surprisingly enjoyable"
bert_res    = bert_pipe(test_text)[0]
roberta_res = roberta_pipe(test_text)[0]

print(f"Text    : '{test_text}'")
print(f"BERT    : {bert_res['label']} ({bert_res['score']:.3f})")
print(f"RoBERTa : {roberta_res['label']} ({roberta_res['score']:.3f})")
