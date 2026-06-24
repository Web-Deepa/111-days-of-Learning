#Day67-Transformer Architecture
import numpy as np
np.random.seed(42)
seq_len,d_model=4,8 #4 words,each has 8 numbers
num_heads=2
d_k=d_model//num_heads #dimension per head

x=np.random.rand(seq_len,d_model) #input embeddings

#1.Positional Encoding
def positional_encoding(seq_len,d_model):
    pos=np.arange(seq_len)[:, None] #position :0,1,2,3../column vector
    i=np.arange(d_model)[None,:] #indices:0-7/row vector
    angle=pos/np.power(10000,(2*(i//2))/d_model) #frequency per dimension
    pe=np.zeros((seq_len,d_model))
    pe[:,0::2]=np.sin(angle[:,0::2]) #even=sine
    pe[:,1::2]=np.cos(angle[:,1::2]) #odd =cosine
    return pe
pe=positional_encoding(seq_len,d_model)
x=x+pe #add position info directly into the embeddings

#2.Multi-Head Self-Attention
def softmax(z):
    e=np.exp(z-np.max(z,axis=-1,keepdims=True))
    return e/e.sum(axis=-1,keepdims=True)

def attention_head(x,d_k):
    Wq,Wk,Wv=[np.random.rand(d_model,d_k) for _ in range(3)]
    Q,K,V=x@Wq,x@Wk,x@Wv
    scores=(Q@K.T)/np.sqrt(d_k)
    weights=softmax(scores)
    return weights @ V

#run multiple heads
heads=[attention_head(x,d_k) for _ in range(num_heads)]
multi_head_out=np.concatenate(heads,axis=-1) #stick all heads together

#3.Residual Coonection + Layer Norm
def layer_norm(z):
    mean=z.mean(axis=-1,keepdims=True)
    std=z.std(axis=-1,keepdims=True)
    return (z-mean)/(std+1e-6)
x=layer_norm(x+multi_head_out) #add +norm

#4.Feed Forward
W1,b1=np.random.rand(d_model,16),np.random.rand(16)
W2,b2=np.random.rand(16,d_model),np.random.rand(d_model)

def relu(z):return np.maximum(0,z)
ffn_out=relu(x @ W1 +b1) @ W2+b2
x=layer_norm(x+ffn_out)
print("Final encoder output shape:",x.shape)
print("Final encoder output:",np.round(x,2))