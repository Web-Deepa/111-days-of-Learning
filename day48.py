#LSTM(pytorch):upgrade from RNN ->Long Short Term Memory
import torch
from torch import nn,optim
import numpy as np
import matplotlib.pyplot as plt

#1.Reproducibility
np.random.seed(0)
torch.manual_seed(0)

#2.create synthetic sine wave data
t=np.linspace(0,100,1000) #1000 time steps
data=np.sin(t)

def create_sequences(data,seq_length):
    xs,ys=[],[]
    for i in range(len(data)-seq_length):
        x=data[i:(i+seq_length)]
        y=data[i+seq_length]
        xs.append(x)
        ys.append(y)
        return np.array(xs),np.array(ys)

SEQ_LEN=10
x,y=create_sequences(data,SEQ_LEN)  

#Convert to Pytorch Tensors
x_tr=torch.tensor(x[:,:,None],dtype=torch.float32)
y_tr=torch.tensor(y[:,None],dtype=torch.float32)

#3.Define LSTM Model
class LSTMModel(nn.Module):
    def __init__(self,input_dim,hidden_dim,layer_dim,output_dim):
        super(LSTMModel,self).__init__()
        self.hidden_dim=hidden_dim
        self.layer_dim=layer_dim
        self.lstm=nn.LSTM(input_dim,hidden_dim,layer_dim,batch_first=True)
        self.fc=nn.Linear(hidden_dim,output_dim)

    def forward(self,x,h0=None,c0=None):
        if h0 is None or c0 is None:
            h0=torch.zeros(self.layer_dim,x.size(0),self.hidden_dim).to (x.device)
            c0=torch.zeros(self.layer_dim,x.size(0),self.hidden_dim).to(x.device)
        out,(hn,cn)=self.lstm(x,(h0,c0))
        out=self.fc(out[:,-1,:])
        return out,hn,cn 

 #4.Initialize  model ,loss & optimizer
model=LSTMModel(input_dim=1,hidden_dim=100,layer_dim=1,output_dim=1)
criterion=nn.MSELoss()
optimizer=torch.optim.Adam(model.parameters(),lr=0.01)

 #5. Training Loop
epochs=100
h0,c0=None,None
for e in range(epochs):
     model.train()
     optimizer.zero_grad()

     outs,h0,c0=model(x_tr,h0,c0)
     loss=criterion(outs,y_tr)
     loss.backward()
     optimizer.step()

     h0,c0=h0.detach(),c0.detach()
     if(e+1)%10==0:
        print(f"Epoch[{e+1}/{epochs}],Loss:{loss.item():.3f}")

#6.Evaluation
model.eval()
with torch.no_grad():
    predicted, _,_ =model(x_tr)
    predicted=predicted.numpy()

#plot results
plt.figure(figsize=(10,4)) 
plt.plot(data,label="True Data")
plt.plot(range(SEQ_LEN,len(predicted)+SEQ_LEN),predicted,label="Predicted")
plt.legend()
plt.savefig("lstm.png")
plt.show()  


