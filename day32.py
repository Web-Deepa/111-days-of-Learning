#Model Evaluation Metrics
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,roc_curve,roc_auc_score,mean_squared_error,r2_score
import numpy as np
import matplotlib.pyplot as plt

y_true=[1,0,0,1,1,0] #true levels
y_scores=[0.2,0.3,0.4,0,0.5,0.1] #predicted probablity
fpr,tpr,thresholds=roc_curve(y_true,y_scores) #calculate false positive rate,true positive rate and thresholds
auc_score=roc_auc_score(y_true,y_scores) #calculate auc-roc score
#plot the ROC Curves
plt.plot(fpr,tpr,label=f'ROC Curve (AUC={auc_score:.2f})')
plt.plot([0,1],[0,1],'k--')
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristics")
plt.legend(loc="lower right")
plt.savefig("roc.png")
plt.show()
print(f"AUC-ROC Scores:{auc_score :.3f}")
y_pred=[0,0,1,1,1,0] #predicted levels
accuracy=accuracy_score(y_true,y_pred) 
precision=precision_score(y_true,y_pred)
recall=recall_score(y_true,y_pred)
f1=f1_score(y_true,y_pred)
print(f"Accuracy:{accuracy:.3f}")
print(f"Precision:{precision:.3f}")
print(f"Recall:{recall:.3f}")
print(f"F1 score:{f1:.3f}")

y_t=[3,2,-0.4,6,0] #true value
y_p=[2.5,0,0.6,2.3,4.5] #predicted 
mse=mean_squared_error(y_t,y_p) #calculate MSE
r2=r2_score(y_t,y_p) #calculate R-squared
print(f"Mean Squared Error:{mse:.3f}")
print(f"R-squared:{r2:.3f}")
