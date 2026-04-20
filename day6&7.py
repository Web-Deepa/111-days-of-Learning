#matplotlib
import matplotlib.pyplot as plt
import numpy as np

#1.simple line chart
sems=[2,4,6,8]
students=[48,45,44,37]
plt.plot(sems,students)
plt.title("Student by semester:")
plt.xlabel("sems")
plt.ylabel("students")
plt.show()

#2.bargraph
product=["mobile","laptop","pc","watch","ipad"]
sell=[40,50,30,25,18]
plt.bar(product,sell, color="yellow")
plt.title("sell unit of product")
plt.xlabel("product")
plt.ylabel("sell")

plt.show()

#histogram

populations=[200,500,300,450,600,500,300]
plt.hist(populations, bins=5, color="green", edgecolor="black")
plt.title("populations distribution")
plt.xlabel("Population value")
plt.ylabel("frequency")
plt.show()

#scatter plot
hour_studied=[2,3,4,4,5]
scores=[70,69,80,94,90]
plt.scatter(hour_studied,scores)
plt.title("scores according to study hours")
plt.xlabel("study hours")
plt.ylabel("exam scores")
plt.show()

#all plot at all
fig,axes=plt.subplots(2,2, figsize=(10,8))

#line 
axes[0,0].plot(sems,students,color="red")
axes[0,0].set_title("students my semester")
#bar
axes[0,1].bar(product,sell,color="yellow")
axes[0,1].set_title("sell unit of products")
#histogram
axes[1,0].hist(populations,bins=5,color="blue",edgecolor="black")
axes[1,0].set_title("population distribution")

#scatter
axes[1,1].scatter(hour_studied,scores, color="pink")
axes[1,1].set_title("scores in exam according to study hour")
plt.tight_layout()
plt.show()


