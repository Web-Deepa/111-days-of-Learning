#piechart
import matplotlib.pyplot as plt
import numpy as np
dist=["Kailali" ,"Kanchanpur" ,"Bardiya","Banke" , "Dang"]
prod=np.array([20,30,40,50,10])
print("Wheat production in five different districts:")
print(prod)
plt.pie(prod, labels=dist)
plt.title("Wheat production according to districts")
plt.show()


