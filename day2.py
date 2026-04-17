#function
def calculate_avg(marks):
    total=sum(marks)
    count=len(marks)
    return total/count
marks=[80,90,85,100,98]
average=calculate_avg(marks)
print("average marks:",average)

#decisions
def grade(mark):
    if mark >=90:
        return "A"
    elif mark >= 85:
        return "A-"  
    elif mark >= 80:
        return "B+"
    else: 
     "others"

for mark in marks:
    print (" Marks {mark}-> {grade(mark)}")
    #details
    student ={
        "name":"Rakesh",
        "sem":6,
        "sub":"ML",
        "score":90

    }
    print("\n student info:")

    for key, value  in  student.items():
        print("{key}:{value}")   
    
    