#handling imbalanced data
import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, confusion_matrix,
                             f1_score, roc_auc_score)
from collections import Counter
from sklearn.utils import resample


#create imbalance data
print("Imbalanced data-as a problem")
x,y=make_classification(
    n_samples=1000,n_features=10,
    weights=[0.95,0.5],random_state=42
)
print(f"Class Distribution:{Counter(y)}")
print(f"Majority:{Counter(y)[0]/len(y)*100:.2f}% Minority:{Counter(y)[1]/len(y)*100:.2f}")

x_tr,x_te,y_tr,y_te=train_test_split(x,y,test_size=0.2,random_state=42,stratify=y)

#naive model with no balancing
naive=RandomForestClassifier(n_estimators=100,random_state=42)
naive.fit(x_tr,y_tr)
y_pred=naive.predict(x_te)
print(f"\n Naive model-No balancing:")
print(f"Accuracy:{(y_pred==y_te).mean():.3f}")
print(f"f1_score:{(y_te,y_pred)}")
report=classification_report(y_te,y_pred,target_names=['majority','minority'])
clean='\n'.join([l for l in report.split('\n') if 'accuracy' not in l])
print(clean)

#fix 
print("2.fix 1-class_weight='balanced")
#fix class wight
wt_model=RandomForestClassifier(n_estimators=100,  class_weight='balanced',random_state=42)
wt_model.fit(x_tr,y_tr)
y_pred_w=wt_model.predict(x_te)

print(f"Accuracy:{(y_pred_w==y_te).mean():.3f}")
print(f"f1_score:{(y_te,y_pred_w)}")
report_w=classification_report(y_te,y_pred_w,target_names=['majority','minority'])
clean_w='\n'.join([l for l in report.split('\n') if 'accuracy' not in l])
print(clean_w)

#fix SMOTE
print("\n Fix-SMOTE Oversampling:")
try:
    from imblearn.over_sampling import SMOTE
    smote=SMOTE(random_state=42)
    x_sm,y_sm=smote.fit_resample(x_tr,y_tr)
    print(f"Before SMOTE:{Counter(y_tr)}")
    print(f"After SMOTE:{Counter(y_sm)}")
    sm_model=RandomForestClassifier(n_estimators=100,  random_state=42)
    sm_model.fit(x_sm,y_sm)
    y_pred_sm=sm_model.predict(x_te)

    print(f"\n Naive model-No balancing:")
    print(f"Accuracy:{(y_pred_sm==y_te).mean():.3f}")
    print(f"f1_score:{(y_te,y_pred_sm)}")
    report_sm=classification_report(y_te,y_pred_sm,target_names=['majority','minority'])
    clean_sm='\n'.join([l for l in report_sm.split('\n') if 'accuracy' not in l])
    print(clean_sm)

except ImportError:
    print("Install Imbalanced-learn:pip install imbalanced-learn")
    print("Skipping SMOTE section....")

#undersampling
print("4.Random UnderSampling:")
#separate classes
x_maj=x_tr[y_tr==0]
y_maj=y_tr[y_tr==0]
x_min=x_tr[y_tr==1]
y_min=y_tr[y_tr==1]

#downsample majority to minority
x_maj_down,y_maj_down=resample(x_maj,y_maj,
                               replace=False,
                               n_samples=len(x_min),
                               random_state=42
                               )
x_under=np.vstack([x_maj_down,x_min])
y_under=np.hstack([y_maj_down,y_min])
print(f"After Undersampling:{Counter(y_under)}")

un_model=RandomForestClassifier(n_estimators=100,  random_state=42)
un_model.fit(x_under,y_under)
y_pred_u=un_model.predict(x_te)

print(f"Accuracy:{(y_pred_u==y_te).mean():.3f}")
print(f"f1_score:{(y_te,y_pred_u)}")
report_u=classification_report(y_te,y_pred_u,target_names=['majority','minority'])
clean_u='\n'.join([l for l in report_u.split('\n') if 'accuracy' not in l])
print(clean_u)

#compare all
print("\n Comparison:")
res={
    'No balancing':f1_score(y_te,y_pred),
    'Class weight':f1_score(y_te,y_pred_w),
    'Undersampling':f1_score(y_te,y_pred_u)
}

print(f"{'Method':<25} {'F1 score (minority)':>20}")
print("-" * 46)
for method, score in res.items():
    print(f"{method:<25} {score:>20.3f}")


