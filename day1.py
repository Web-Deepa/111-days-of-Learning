#  Functions and Data Structures
#  Functions 
def calculate_avg(marks):
    total = sum(marks)
    count = len(marks)
    return total / count

scores = [70, 85, 90, 60, 95]
average = calculate_avg(scores)
print("Average score:", average)

# 2. If/else 
def grade(score):
    if score >= 90:
        return "A"
    elif score >= 85:
        return "A-"
    elif score >= 80:
        return "B+"
    else:
        return "others"

for score in scores:
    print(f"Score {score} → {grade(score)}")

# 3. Dictionary (like a mini database row in ML)
student = {
    "name": "Rakesh",
    "sem": 6,
    "sub": "ML",
    "score": 90
}

print("\n Student Info:")
for key, value in student.items():
    print(f"{key}: {value}")

# 4. List of dictionaries 
dataset = [
    {"name": "Paban", "score": 85, "passed ": True},
  
    {"name": "Deepak", "score": 40, "passed": False},
]

print(f"\n Dataset:")
for row in dataset:
    print(row)