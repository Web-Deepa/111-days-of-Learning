#numbers and arrays

import numpy as np
#array

num=np.array([10,3,56,89,90])
print("arrays:",num)
print("types:",type(num))

print("add 2 to all:",num+2)
print("multiply by 5 to all:", num*5)
print("square:", num**2)

#statistics
print("sum:",np.sum(num))
print("mean:",np.mean(num))
print("maximum:",np.max(num))
print("minimum:",np.min(num))

#2D array
marks=np.array([[80,90,88],[78,95,92], [86,98,79]])
print("marks:",marks)  
print("shape(rows,column)",marks.shape) 
print("average of all marks:",np.mean(marks)) 
print("average marks of per student:",np.mean(marks ,axis=1))          

#random data
random_data=np.random.randint(0, 100, size=4)      
print("random data:",random_data)           
                
