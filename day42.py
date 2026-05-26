#Semantic Analysis
import re 
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,classification_report,f1_score
from sklearn.pipeline import Pipeline

#1.sample dataset(positive/negative reviews)
reviews=[
    #positive
    "The product is very good",
    "High quality",
    "Fast delivery highly recommend",
    "Outstanding performance",

#negative
"Bad product broke after one day of purchase",
"Waste of money",
"Totally useless of time",
"Low quality"
]
lbl=[1]*4 +[0]*4 #1=pos,0=neg
def clean(text):
    text=text.lower()
    text=re.sub(r'[^a-z\s]', '',text)
    return re.sub(r'\s+',' ',text).strip()
reviews_clean=[clean(r) for r in reviews]
x_tr,x_te,y_tr,y_te=train_test_split(reviews_clean,lbl,test_size=0.25,random_state=42)

#2.Model Comparison
print("Semantic analysis-Model Comparison---")
models={
    'Logistic Regression':LogisticRegression(max_iter=1000),
    'LinearSVC':LinearSVC(max_iter=1000)
}

best_score,best_pipe=0,None
print(f"{'Model':<25} {'Accuracy':>6} {'F1':>6}")
print("-" *40)
for name,clf in models.items():
    pipe =Pipeline([
        ('tfidf',TfidfVectorizer(ngram_range=(1,2),sublinear_tf=True)),
        ('clf',clf)
    ])
    pipe.fit(x_tr,y_tr)
    y_pred=pipe.predict(x_te)
    acc=accuracy_score(y_te,y_pred)
    f1=f1_score(y_te,y_pred,average='macro')
    print(f"{name:<25} {acc:>6.3f} {f1:>6.3f}")
    if f1 > best_score:
        best_score, best_pipe = f1, pipe

#3.Predict custom reviews
print("\n Custom review predictions---")
custom=[
    'It is best product I have ever used',
    'Terrible quality broke immediately avoid',
    'It is okay but not great also not so bad'
]
preds=best_pipe.predict([clean(r) for r in custom])
lbl_map={1:"Positive 😊",0:"Negative 😢"}
for review,pred in zip(custom,preds):
    print(f" '{review[:45]}' -> {lbl_map[pred]}")

#4.Lexicon -based(no training needed)
print("Lexicon-based Sentiment(rule -based)---")
pos={'love','great','best','perfect','brillant','happy','high','good','fast','outstanding'}
neg={'terrible','bad','useless','avoid','low','waste','broke'}

def lexicon_sentiment(text):
    words=clean(text).split()
    p=sum(1 for w in words if w in pos)
    n=sum(1 for w in words if w in neg)
    if p>n:return "Positive 😊"
    elif n>p:return "Negative 😢"
    else: return "Neutral😐"
for review in custom :
    print(f" '{review[:45]}' -> {lexicon_sentiment(review)}") 
