#scipy and statistics:
from scipy import  stats
import numpy as np

print("basic statistics:")
size=np.array([3,4,5,3,6,7,9])
print("given data:",size)

print(f"Mean: {np.mean(size):.2f}") #mean
print(f"median: {np.median(size):.2f}") #median
mo=stats.mode(size) #mode
print(f"Mode: {mo.mode}")

print("describe all Stats--")
res=stats.describe(size)
print(f"count:{res.nobs}")
print(f"min: {res.minmax[0]}")
print(f"max: {res.minmax[1]}")
print(f"Mean: {res.mean:.2f}")
print(f"Standard deviation:{np.std(size):.2f}")
print(f"variance: {res.variance:.2f}")
#Z Score
print("Z Score--")
scores=np.array([90,96,93,89,87,80])
z_scores=stats.zscore(scores)
print("Scores:",scores)
print("Z Scores:",np.round(z_scores,1))
print("Z Scores:",np.round(z_scores,2))

#Analysis--
print("Analysis of data:")
for score, z in zip(scores,z_scores):
    if abs(z) > 1.5:
        print(f" Score {score} -> z={z:.2f} ,<- unusual!")
    else:
        print(f" Score {score} -> z={z:.2f} <- normal")

#correlation
day=np.array([1,2,4,6,8,8,1])
temp=np.array([20,30,24,27,29,27,23])
corr,pvalue=stats.pearsonr(day,temp)
print(f"day: {day}")
print("tempearture: {temp}")
print(f"correlation: {corr:.2f}")
print(f"P-value: {pvalue:.2f}")

if corr > 0.7:
    print("strong positive relations")
elif corr < -0.7:
    print("strong negative relationships")   

#real example:
print("shop sale analysis:")     
shop_sales=np.array([5000,8000,4000,3000])
r=stats.describe(shop_sales)
z=stats.zscore(shop_sales)

print(f"total week: {r.nobs}")
print(f"average sale: {r.mean}")
print(f"best sale week: {r.minmax[1]}")
print(f"worst week: {r.minmax[0]}")
print(f"Variance: {r.variance}")

print("weekly sale analysis")
for i,(sale,zscore) in enumerate (zip(shop_sales,z)):
    status="good week" if zscore > 0 else "slow week"
    print(f"Week {i+1}:Rs{sale}->{status}")
