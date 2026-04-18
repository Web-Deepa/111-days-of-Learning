import pandas as pd
#dataframe

data={
    "name":["Deepa" , "Deepak" ,"Paban" , "Sunti"],
    "sem":[6,5,3,7],
    "marks":[90,80,78,37],
    "passed":[True ,True ,True, False]
}
df=pd.DataFrame(data)
print("\n details:\n",df)
#explore data
print("\n data info:")
print("shape(rows,column)",df.shape)
print("lncolumns name",df.columns.tolist())

#print 1st 3 rows
print(df.head(3))

print(df.describe())
print(df["name"])
print(df["sem"])
print(df[["name","marks"]])
print(df[df["marks"]>80])

#adding grade
df["grade"]=df["marks"].apply(lambda x:"A" if x>=90 else "A-" if x>=85 else "others")
print(df)