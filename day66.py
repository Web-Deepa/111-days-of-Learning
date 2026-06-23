#Day66-Attention Mechanism
import numpy as np

#1.Define data
np.random.seed(42)
words=['I','Love','Python','Code']
embeddings=np.random.rand(4,3) #shape(4 words,3dimensions)

#2.Create query,key and value matrices
Wq=np.random.rand(3,3) 
Wk=np.random.rand(3,3) #Key
Wv=np.random.rand(3,3) #Value

Q=embeddings @Wq #Query
K=embeddings @Wk #Key
V=embeddings @Wv  #Value

print("Query shape:",Q.shape)
print("Key shape:",K.shape)
print("Value shape:",V.shape)

#3.Ca;cua;te attention score
scores=Q@K.T
print(" Raw Attention score:",np.round(scores,2))

#4.Scale scores
d_k=K.shape[-1]
scaled_scores=scores/np.sqrt(d_k)

#5.Softmax
def softmax(x):
    e=np.exp(x-np.max(x,axis=-1,keepdims=True))
    return e/e.sum(axis=-1,keepdims=True)

attention_weights=softmax(scaled_scores)   
print("Attentions weights:",np.round(attention_weights,2))

#6.Multiply weights by value
output=attention_weights @ V 
print("Final attention output:",np.round(output,3))

#7.Show which word each word attend to most
for i ,word in enumerate(words):
    target=words[np.argmax(attention_weights[i])]
    print(f" '{word}->'{'targe'} (weight={attention_weights[i].max():.2f})")

                                          

