#statistics
import numpy as np
print("Given data:")
price=[10,20,50,50,100,5,30]
data=np.array(price)
print(data)
print("data length:",len(data))
#mean
mean=np.mean(data)
print("Average of given data:",mean)
#median
median=np.median(data)
print("Median:",median)

from scipy import stats
mod=stats.mode(data) #mode
print(f"mode: {mod.mode}")

variance=np.var(data) #variance
print(f"variance: {variance:.3f}")
sd=np.std(data)
print(f"standard deviation: {sd:.2f}")

print(f"minimum: {np.min(data)}")
print(f"maximum: {np.max(data)}")

range=np.max(data)-np.min(data)
print(f"Range: {range}")

print("Percentiles--")
p0=np.percentile(data,0)
print(f"p0: {p0}")
p25=np.percentile(data,25)
print(f"p25: {p25}")
p50=np.percentile(data,50)
print(f"P50: {p50}")
p75=np.percentile(data,75)
print(f"p75: {p75}")
p100=np.percentile(data,100)
print(f"p100: {p100}")

#correlation
hr_study=np.array([2,3,4,5,1])
scores=np.array([80,90,89,86,94,])
corr=np.corrcoef(hr_study,scores)[0][1]
print(f"study hours: {hr_study}")
print(f"exam scores: {scores}")
print(f"Correlation: {corr: .2f}")
if corr> 0.7:
 print("strong correlation")
 print("more study= more marks")
elif corr < -0.7:
 print("strong negatiuve correlation") 
else:
 print("weak correlation") 

 #example
 print("real dataset analysis:")
 shop_data={
  "Dhangadhi":np.array([5000,6000,7000,4000]),
  "Nepalgunj":np.array([6000,5500,7300,8000]),
  "Dadeldhura":np.array([4000,2000,5000,6500])

 }
 print(f"{'city': <12} {'mean' :>8} {'std':>7} {'min':>7} {'max':>7}")
 print("-" *45)
 for city,sales in shop_data.items():
  print(f"{city:<12}"
        f"{np.mean(sales):>8.0f}"
        f"{np.std(sales):>8.0f}"
        f"{np.min(sales):>8}"
        f"{np.max(sales):>8}")
  
 #outliners
 temp=np.array([28,30,28,26,25,20,28])
 print("temperature in a week:",temp)
 mean=np.mean(temp)
 print(f"average temperature: {mean:.2f}")
 std=np.std(temp)
 print(f"std:{std:.2f}")
 print("outliners found:")
 for t in temp:
  if abs(t - mean)>2* std:
   print(f"{t} is an outliner")
  else:
   print(f"{t} is normal")


