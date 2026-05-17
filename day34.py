#Dimensionality Reduction and PCA
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import load_breast_cancer,load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#1.PCA
print("Basic PCA -Breast Cancer(30->2)---")
bst=load_breast_cancer()
x,y=bst.data,bst.target
sc=StandardScaler() #scaler
x_scaled=sc.fit_transform(x)
pca=PCA(n_components=2)
x_pca=pca.fit_transform(x_scaled)

print(f"Original shape:{x.shape}")
print(f"Compressed shape:{x_pca.shape}")
print(f"PC1 variance:{pca.explained_variance_ratio_[0]:.3f}")
print(f"PC2 variance:{pca.explained_variance_ratio_[1]:.3f}")
print(f"Total Explained:{pca.explained_variance_ratio_.sum():.3f}")

#plot
plt.figure(figsize=(7,5))
colors=['steelblue','coral']
for i , label in enumerate(bst.target_names):
    mask = y == i
    plt.scatter(x_pca[mask,0],x_pca[mask,1],c=colors[i],label=label,alpha=0.6,s=30)
plt.xlabel("Principal Component 1")  
plt.ylabel("Principle Component 2")
plt.title("PCA - Breast Cancer of 2 components")
plt.legend()
plt.savefig("pca.png")
plt.show()

#2.screeplot
print("SCree Plot - Find the Elbow ---")
pca_full=PCA()
pca_full.fit(x_scaled)
cumulative=np.cumsum(pca_full.explained_variance_ratio_)
print(f"{'Components':>12} {'Cumalative variance':>20}")
print("-" * 40)
for i,cv in enumerate(cumulative[:10],1):
    print(f"{i:>12} {cv:>20.3f}")
plt.figure(figsize=(8,4))
plt.subplot(1,2,1)
plt.plot(range(1,11),pca_full.explained_variance_ratio_[:10],'o-',color='steelblue')
plt.xlabel("Components")
plt.ylabel("Variance") 
plt.title("Sree Plot")   

plt.subplot(1,2,1)
plt.plot(range(1,11),pca_full.explained_variance_ratio_[:10],'s-',color='orange')
plt.axhline(y=0.95,color='red',linestyle='--',label='95% threshold')
plt.xlabel("Components")
plt.ylabel(" Cumulative Variance") 
plt.title("Cumulative Variance by Components")
plt.legend()
plt.savefig("scree.png")
plt.show()

n_95=np.argmax(cumulative >= 0.95)+1
print(f"Components needed for 95% variance:{n_95}")

#3.PCA as preprocessing
print("3.PCA as Preprocessing Before vs After --- ")
x_tr,x_te,y_tr,y_te=train_test_split(x_scaled,y,test_size=0.2,random_state=42)

#without PCA
m1=RandomForestClassifier(n_estimators=100,random_state=42)
m1.fit(x_tr,y_tr)
acc_full=accuracy_score(y_te,m1.predict(x_te))

#with PCA
pca_pre=PCA(n_components=0.95)
x_tr_pca=pca_pre.fit_transform(x_tr)
x_te_pca=pca_pre.transform(x_te)

m2=RandomForestClassifier(n_estimators=100,random_state=42)
m2.fit(x_tr_pca,y_tr)
acc_pca=accuracy_score(y_te,m2.predict(x_te_pca))

print(f"{'Model':<30} {'Features':>10} {'Accuracy':>10}")
print("-" * 55)
print(f"{'Without PCA':<30} {x_tr.shape[1]:>10} {acc_full:>10.3f}")
print(f"{'With PCA (95% var)':<30} {x_tr_pca.shape[1]:>10} {acc_pca:>10.3f}")

#4.Iris 3D ->2D
print("\n 4.Iris - 4 features -> 2D plot ---")
iris=load_iris()
x_iris=StandardScaler().fit_transform(iris.data)
pca_iris=PCA(n_components=2)
x_iris_2d=pca_iris.fit_transform(x_iris)
print(f"Variance explained:{pca_iris.explained_variance_ratio_.sum():.3f}")
plt.figure(figsize=(7,5))
colors=['steelblue','orange','green']
for i,name in enumerate(iris.target_names):
    mask=iris.target == i
    plt.scatter(x_iris_2d[mask,0],x_iris_2d[mask,1],c=colors[i],label=name,alpha=0.7,s=40)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("Iris - PCA 2D projection")
plt.legend()
plt.savefig("iris_pca.png")
plt.show()


