#linear function
import numpy as np
print("linear function--")
def linear(x,a,b):
    return a * x + b
books=np.array([1,2,3,4,5])
price=linear(books,500,50)
print("books:",books)
print("price of books:",price)
for b,p in zip(books,price):
    print(f"{b} books -> Rs.{p}")

#probablity
print("probablity--")
def sigmoid(x):
    return 1/(1+np.exp(-x))
val=np.array([1,-2,0,3,5])
prob=sigmoid(val)

print("values->probablity:")
for v,p in zip(val,prob):
    print(f"{v:3d}-> {p:.2%}")

#another example
scores=np.array([90,86,89,38,85])
scale=(scores-60)/10
prob=sigmoid(scale)
print("pass of probablity by scores:")
for s,p in zip(scores,prob):
    result = "PASS" if p >= 0.45 else "FAIL"
    print(f"scores {s} -> probablity {p:.2%}->{result}")

#data normalization
print("Normalization--")
sal=np.array([200,300,400,100,500])
print("original salary:",sal)
min_sal= np.min(sal)
max_sal= np.max(sal)
norm=(sal - min_sal) / (max_sal - min_sal)
print("Normalized(0 to 1):",np.round(norm,2))