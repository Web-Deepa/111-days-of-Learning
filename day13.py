# Data Cleaning 
import pandas as pd
import numpy as np

print("Messy dataset:")
messy_data = {
    "name":   ["Deepa","Pabitra","Bishal","Prem",None,"Deepa"],
    "age":    [21, 23, 20, None, -22, 21],
    "city":   ["Dhangadhi","Pokhara","Nepalganj",
               "Kathmandu","Dhangadhi","Dhangadhi"],
    "salary": [25000, 30000, 15000, 20000, 28000, 25000]
}
df = pd.DataFrame(messy_data)
print(df)
print("Data shape:", df.shape)

# Finding problems
print("\nFinding problems--")
print("Missing values:\n", df.isnull().sum())
print("Duplicate rows:", df.duplicated().sum())
print("Data types:\n", df.dtypes)
print("Dataset summary:\n", df.describe())

# Data Cleaning
print("\nCleaning dataset--")
df_clean = df.copy()

# 1. Replace negative age with NaN
df_clean.loc[df_clean["age"] < 0, "age"] = np.nan

# 2. Calculate mean values
mean_age = df_clean["age"].mean()
mean_salary = df_clean["salary"].mean()

# 3. Fill missing values (CORRECT WAY)
df_clean["age"] = df_clean["age"].fillna(mean_age)
df_clean["salary"] = df_clean["salary"].fillna(mean_salary)
df_clean["name"] = df_clean["name"].fillna("Unknown")

# 4. Fix unrealistic age (>100)
df_clean.loc[df_clean["age"] > 100, "age"] = mean_age

# 5. Remove duplicates and reset index
df_clean = df_clean.drop_duplicates().reset_index(drop=True)

# 6. Clean text data
df_clean["city"] = df_clean["city"].str.title()
df_clean["name"] = df_clean["name"].str.strip()

# Final Output
print("\nFinal clean dataset:")
print(df_clean)
print("Final shape:", df_clean.shape)
print("Any missing values left?", df_clean.isnull().sum().any())

# Save cleaned data
df_clean.to_csv("clean_data.csv", index=False)
print("\nClean data saved successfully!")
