#calculus

import numpy as np
print("Derivation (slope):") #derivative
def f(x): #given function
    return x **2
def derivative(x):#derivative of f(x)
    return 2*x
x_values=np.array([2,3,5,6,-1,0])
print("  x    f(x)    slope")
print("-" * 25)
for x in x_values:
    print(f"{x:3d} {f(x):6.1f} {derivative(x):6.1f}")
print("meaning of slope:")
print("negative slope-> go right to reduce")
print("positive slope-> go left to reduce")   
print("Zero slope-> You are at the bottom")

#gradient descent
print("Gradient descent:")
print("finding minimum of y=x^2")
print("answer should be x=0")
print() #blank space
x=10 #starting value
learning_rate=0.1 #step size
epochs=50 #no of steps n
print(f"starting value : x={x}")
print(f"learning rate: {learning_rate}")
print(f"{'Step': >5} {'x':>10} {'slope':>10}")
print("-" *35)

for i in range(epochs):
    slope=derivative(x) #find slope
    x=x - slope*learning_rate #move opposite to slope
    if i % 10 == 0:
        print(f"{i: >5} {x:>10.4f} {slope:>10.4f}")
        print()
print(f"final x={x:.6f}")  
print(f"target=0.000000")      
print(f"f(x)={f(x):.8f}")

#learning rate effect
print("learning rate effect--")
def gradient_descent(start,lr,n):
    x=start
    for _ in range(n):
        x= x- lr * derivative(x)
    return x

lr_small=gradient_descent(10,0.01,50)
lr_medium=gradient_descent(10,0.1,50) #good
lr_large=  gradient_descent(10,0.99,50)  #big
print(f"learning rate:0.01 -> x= {lr_small:.4f} :very slow " )    
print(f"learning rate:0.1 -> x={lr_medium:.4f}:good just right")
print(f"learning rate:0.99 -> x={lr_large:.4f}:very fast")

#real example--
demand=np.array([8,7,6,5,3])
supply=np.array([5,5,6,6,7])    
m=0 #slope
b=0 #bias
lr= 0.001
print(f"starting :m= {m} ,b={b}")
print(f"{'epochs':>6} {'m':>8} {'b':>8} {'loss':>10}")
print("-" *40)
for epochs in range(100):
    pred=m * demand + b
    error= supply - pred
    dm= -2 * np.mean(demand * error)
    db= -2 * np.mean(error)
    lr=0.001
    m = m- lr * dm #update m
    b = b- lr * db #update b
    
    loss=np.mean(error ** 2) #calculate loss
    if epochs%40==0:
        print(f"{epochs:>6} {m:>8.3f} {b:>8.3f} {loss:>10}")
print(f"AI learned:")
print(f"m={m:.3f} :slope")  
print(f"b={b:.3f} :Intercept")  
print(f"prediction:")  
for d,actual  in zip(demand,supply):
    pred=m * d + b
    print(f"{d}->demand->predicted:{pred:.2f},actual:{actual}")

#chain rule:
print("chain rule used in deep neural networks")
def g(x):
    return 2 * x + 3
def f_chain(u):
    return u ** 2
def chain_derivative(x):
    return 4*(2 * x + 3)
x_test=np.array([1,0,2,-3])
print(f"{'x':>5} {'f(g(x))':>12} {'dy/dx':>10}")
print("-" * 40)
for x in x_test:
    u=g(x)
    y=f_chain(u)
    dy=chain_derivative(x)
    print(f"{x:>5} {y:>12.1f} {dy:>10.1f}")


