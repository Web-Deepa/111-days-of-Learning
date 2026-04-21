 # classes
class MyClass:
    x = 6
    y=9
p1=MyClass()
print(p1.x)
del p1
p2=MyClass()
print(p2.x)
print(MyClass().y)

#class without init
class Customer:
    pass
p1=Customer()
p1.name="Deepak"
p1.contact="9848976539"
print(p1.name)
print(p1.contact)


#class with self
class Student:
 def __init__(self,roll,name,sem):
   self.roll=roll
   self.name=name
   self.sem=sem
 
p1=Student(12,"Deepa",6)
print("student details:")
print("Roll:",p1.roll)
print("Name:",p1.name)
print("Sem:",p1.sem)        
      
 #dog
class Dog:
   def __init__(self,name,age):
      self.name=name
      self.age=age
   def bark(self):
      print(f"{self.name} says ehoo")
   
d1=Dog("Punte",1)            
print(" Dog Name:",d1.name)
print("Age:",d1.age)
d1.bark()

#inheritance
class Parent:
   def __init__(self,f_name,l_name):
      self.f_name=f_name
      self.l_name=l_name
   def print_name(self):
      print(self.f_name,self.l_name)

class Child(Parent):
   pass
c=Child("Usha ","Tamang")   
c.print_name()

class Child2(Parent):
   def __init__(self,f_name,l_name):
      Parent.__init__(self,f_name,l_name)
c2=Child("Bibek","Shrestha")
c2.print_name()

# simple polymorphism
h="Hello! Python"
print("length of 'Hello! Python':",len(h))

write_tuple={"pen","notebook","dairy","pencil","marker"}
print(len(write_tuple))

#class polymorphism
class Sem_2:
   def __init__(self,course_name,course_code):
      self.course_name=course_name
      self.course_code=course_code
   def type(self):
      print("Theory:",self.course_name)

class Sem_4:
   def __init__(self,course_name,course_code):
      self.course_name=course_name
      self.course_code=course_code
   def type(self):
      print("Coding:",self.course_name)     

class Sem_6:
   def __init__(self,course_name,course_code):
      self.course_name=course_name
      self.course_code=course_code
   def type(self):   
      print("Numerical:",self.course_name)    

s2=Sem_2("DSA","CM02")
s4=Sem_4("Java","CM80")
s6=Sem_6("Image Prrocessing","CM85")

for x in(s2,s4,s6):
   x.type()

#inheritance polymorphism
class Animal:
   def __init__(self,name,habitant):
      self.name=name
      self.habitant=habitant
   def move(self):
      print("Move") 
class  Cow(Animal):
   def move(self):
      print("Walk")

class Horse(Animal):
     def move(self):
        print("Run")

class Pigeon(Animal):
   def move(self):
      print("fly")

c=Cow("Cow","Shed")      
h=Horse("Hourse","Stable")
p=Pigeon("Bird","Tree")  

for x in (c,h,p):
  print("Name:",x.name)
  print("Habiatnt:", x.habitant)
  print(f"move by:") 
  x.move()
 
  
