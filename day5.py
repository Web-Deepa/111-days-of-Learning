#1.list
students=["Juntara","Pradeep","Anish"]
print("All students:",students)
print("first student:",students[0])
print("last student:",students[-1])
print("total student:",len(students))

 #add students
students.append("Bishal")
students.append("Sabitri") 
print("new students:",students)

#remove students
students.remove("Pradeep")
print("students after remove:",students)

#looping through list
for s in students:
    print("student:",s)


#2.dictionary
customer={
    "name":"Nilam",
    "email":"nilam2@gmail.com",
    "buy":5000,
    "regular_customer ":True

}
print("Name of customer:",customer["name"])
print("email:",customer["email"])
print("buy:",customer["buy"])

#update value
customer["buy"]=10000
print("updated buy:",customer["buy"])
#add new key
customer["rate"]="***"
print("all of customers:",customer)


#3.CSV file
import csv
subscribers=[{"name":"Janak", "email":"janakp4@gmail.com","movie":"Mohar"},
             { "name":"Prativa", "email":"prati@gmail.com","movie":"Paralko Aago"},
             {"name":"Santosh" , "email":"santoshw@gmail.com","movie":"Boksiko ghar"}

]
print("details of subscribers:" ,subscribers)

#write in CSV file
with open ("subscribers.csv" ,"w" ,newline="") as file:
    writer=csv.DictWriter(file,fieldnames=["name","email","movie"])
    writer.writeheader()
    writer.writerows(subscribers)
    print("CSV file created successfully")

    #read  CSV file
with open("subscribers.csv","r") as file:
    reader=csv.DictReader(file)
    print("\n reading CSV ---")
    for row in reader:
            print(f"Name: {row['name']}, Email: {row['email']}, Movie:{row['movie']}")

#filter CSV
with open("subscribers.csv","r") as file:
     reader=csv.DictReader(file)
     print("\n filter data:")
     for row in reader:
          if str(row["movie"])=="Mohar":
            print(f"Name: {row['name']}, Movie: {row['movie']}")   

