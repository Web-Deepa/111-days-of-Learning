#Naive Bayes
import numpy as np

from sklearn.naive_bayes import GaussianNB,MultinomialNB,BernoulliNB
import seaborn as sns
from sklearn.datasets import load_iris,load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.metrics import accuracy_score,classification_report
from sklearn.feature_extraction.text import CountVectorizer

#1.GaussianNB on Iris
print("1.Gaussian Naive Bayes-Iris:")
iris=load_iris()
x,y=iris.data,iris.target
x_tr,x_te,y_tr,y_te=train_test_split(x,y,test_size=0.2,random_state=42)
gnb=GaussianNB()
gnb.fit(x_tr,y_tr)
y_pred=gnb.predict(x_te)
report=classification_report(y_te,y_pred,target_names=iris.target_names)
clean='\n'.join([l for l in report.split('\n') if 'accuracy' not in l])
print(clean)
print(f"Overall Accuracy:{accuracy_score(y_te,y_pred):.3f}")

#2.GaussianNB on Breast Cancer
print("\n 2.Gaussian Naive Bayes-Breat Cancer:")
bst=load_breast_cancer()
x,y=bst.data,bst.target
x_tr,x_te,y_tr,y_te=train_test_split(x,y,test_size=0.2,random_state=42)
gnb2=GaussianNB()
gnb2.fit(x_tr,y_tr)
y_pred2=gnb2.predict(x_te)
report2=classification_report(y_te,y_pred2,target_names=bst.target_names)
clean2='\n'.join([l for l in report2.split('\n') if 'accuracy' not in l])
print(clean2)
print(f"Overall Accuracy:{accuracy_score(y_te,y_pred2):.3f}")

#comparison
print("3.NB Types Comparison -Breast Cancer--")

#Multinomial requires non negative - MinMaxScaler
mms=MinMaxScaler()
x_tr_mm=mms.fit_transform(x_tr)
x_te_mm=mms.transform(x_te)
mnb1=MultinomialNB()
mnb1.fit(x_tr_mm,y_tr)
acc_m=accuracy_score(y_te,mnb1.predict(x_te_mm))

#BernoulliNB -Binary features
bnb=BernoulliNB()
bnb.fit(x_tr,y_tr)
acc_b=accuracy_score(y_te,bnb.predict(x_te))

print(f"{'Model':<20} {'Accuracy':>10}")
print("-" * 30)
print(f"{'GaussianNB':<20} : {accuracy_score(y_te,y_pred2):.3f} ")
print(f"{'MultinomalNB':<20}: {acc_m:>10.3f}")
print(f"{'BernoulliNB':<20} : {acc_b:>10.3f}")




#4.MultinomialNB
print("\n 4.Multinomial NB - Spam Detection:")
emails=[
    "free money win prize now",
    "click here to claim your reward",
    "you won a lottery congratulations",
    "free offer limited time win",
    "buy cheap pills online now",
    "meeting tomarrow at 9am please confirm",
    "your project report is due friday",
    "can we schedule a call this week",
    "lunch at 1pm today sounds good",
    "please review the attached document",

]
lbl=[1,1,1,1,1,0,0,0,0,0] #1=spam,0=no-spam/ham
vectorizer=CountVectorizer()
x_text=vectorizer.fit_transform(emails)
x_trn,x_tes,y_trn,y_tes=train_test_split(x_text,lbl,test_size=0.2,random_state=42)
mnb=MultinomialNB()
mnb.fit(x_trn,y_trn)
y_pred3=mnb.predict(x_tes)
print(f"Accuracy:{accuracy_score(y_tes,y_pred3):.3f}")
print(f"Predicted:{y_pred3.tolist()} (1=spam,0=ham)")

#test a email
new_emails = ["win free money now","see you at the meeting  tomarrow"]
new_vect=vectorizer.transform(new_emails)
preds=mnb.predict(new_vect)
probs=mnb.predict_proba(new_vect)
print("New email predictions:")
for email,pr,pb in zip(new_emails,preds,probs):
    label="SPAM" if pr==1 else "HAM"
    print(f" '{email}' ") 
    print(f" ->{label} (spam pb:{pb[1]:.3f})")


            
