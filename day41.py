#Text Classification
import re
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score,f1_score,classification_report

#data
cats=['sci.med','sci.space','rec.sport.hockey','talk.politics.guns']
train = fetch_20newsgroups(subset='train', categories=cats, remove=('headers','footers','quotes'))
test=fetch_20newsgroups(subset='test', categories=cats, remove=('headers','footers','quotes'))

def clean(text):
    text=text.lower()
    text=re.sub(r'\S+@\S+|http\S+|[^a-z\s]', '',text)
    return re.sub(r'\s+',' ',text).strip()

x_tr=[clean(t) for t in train.data]
x_te=[clean(t) for t in test.data]
y_tr,y_te=train.target,test.target
print(f"Train:{len(x_tr)}|Test:{len(x_te)}|Classes:{cats}")

#compare model
models={
    'Naive Bayes':MultinomialNB(),
    'Logistic Regression':LogisticRegression(max_iter=1000),
    'LinearSVC':LinearSVC(max_iter=2000)
}
print(f"{'Model':<25} {'Accuracy':>10} {'F1 macro':>10}")
print("-" *45)
best_score,best_pipe,best_name=0,None,''
for name,clf in models.items():
    pipe=Pipeline([
       ('tfidf', TfidfVectorizer(ngram_range=(1,2), sublinear_tf=True, max_features=50000)),
        ('clf',clf)

    ])
    pipe.fit(x_tr,y_tr)    
    y_pred=pipe.predict(x_te)
    acc=accuracy_score(y_te,y_pred)
    f1=f1_score(y_te,y_pred,average='macro')
    print(f"{name:<25} {acc:>10.3f} {f1:>10.3f}") 
    if f1>best_score:
        best_score,best_pipe,best_name=f1,pipe,name

# best model report
print(f"Best Model{best_name}")
y_pred_best=best_pipe.predict(x_te)
report=classification_report(y_te,y_pred_best,target_names=cats)
print('\n'.join([ l for l in report.split('\n') if 'accuracy' not in l]))

#predict custom text
print("Custom Predictions--")
samples=[
    "students code for change",
    "Gardern is beautiful",
    "Current noode is banned",
    "Admission open"
]
preds=best_pipe.predict([clean(t) for t in samples])
for text,pred in zip(samples,preds):
    print(f" '{text}' ->{cats[pred]}")

