#metrices
import numpy as np
#ccreate matrix
print("Creating matrix--")
m1=np.array([[1,2,3],
             [4,5,6],      
           [7,8,9]])
print("m1:",m1)
print("shape:",m1.shape)
print("Rows:",m1.shape[0])
print("Columns:",m1.shape[1])

#access elements
print("matrix elements:")
print("1st row:",m1[0])
print("2nd row:",m1[1])
print("3rd row ",m1[2])

print("1st element:",m1[0][0])
print("row 2 and col 3 element:",m1[1][2])

# matrix math
a=np.array([[0,1,5],[4,5,8],[9,4,3]])
b=np.array([[6,7,8],[8,0,3],[2,1,9]])
print("Matrix A:" )
print(a)
print("Matrix B:")
print(b)
print("A+B:",a+b)
print("A-B:",a-b)
print("A*B:",a*b) #element by multiply
print(" dot product:")
mul=np.dot(a,b)
print("a.b:",mul)


#matrix operations
print("Matrix C:")
c=np.array([[2,4,3],[3,7,8],[9,0,4]])
print(c)
print("Transpose of C:")
print(c.T)
print("sum of eacch row:",np.sum(c,axis=1))
print("sum of each column:",np.sum(c,axis=0))

#real example
dataset=np.array([[89,56,90],[78,90,78],[97,60,78]])
products=["Android","Samsung","Apple"]
market=["Dhangadhi","Pokhara","Kathmandu"]
print(" full dataset :")
print("Shape:",dataset.shape)
print("Average per products:")
for i,s in enumerate(products):
    avg=np.mean(dataset[i])
    print(f"{s}: {avg:1f}")

print("average per markets:")
for i,m in enumerate(market):
 avg=np.mean(dataset[:,i])
 print(f"{m} :{avg:.1f}")