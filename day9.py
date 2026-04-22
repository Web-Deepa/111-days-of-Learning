#vectors
import numpy as np 
#define
student_marks=np.array([90,85,88,80,94,87])
print("students marks vector:",student_marks)
humidity=np.array([20,30,40,47,60,50])
print("humidity vecctor:",humidity)

#vector math
a=np.array( [20,60,4])
b=np.array([34,8,65])
print(" Two vectors are:")
print("a:",a)
print("b:",b)
print("a+b:",a+b)
print("a-b:",a-b)
print("a*b:",a*b)
print("a/b:",a/b)

#dot product
dot=np.dot(a,b)
print("dot product of and b= a.b:",dot)

#example
marks=np.array([90,88,89,95])
weights=np.array([0.5,0.6,0.4,0.3])
score=np.dot(marks,weights)
print(f"final score is:",score)

#length:uses pythagoros theorem
v=np.array([56,12,2,1])
l = np.linalg.norm(v)
print("vector :",v)
print("length:",l)

#AI example
print("fruits array")
apple=np.array([20,30,25,15])
mango=np.array([40,25,19,23])
grapes=np.array([23,45,68,100])
fruits={"Apple": apple,"Mango": mango, "Grapes": grapes }

for name,sell_unit in fruits.items():
    avg=np.mean(sell_unit)
    total=np.sum(sell_unit)
    print(f"{name}:sell unit {sell_unit} ,average:{avg},total:{total}")




