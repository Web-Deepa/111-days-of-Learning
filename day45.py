##Text Sentiment Classification using RNN/LSTM
import torch
import torch.nn as nn
import torch.optim as optim

#1.Set Dummy dataset
#0=negative,1=positive
sentences=[
    "I love this movie",
    "This movie is terrible",
    "Waste of time and money",
    "Highly recommend this master piece",
    "Useless scene"

]
lbl=[1,0,0,1,0]

#2.Build vocabulary
voc=set([word for sen in sentences for word in sen.split()])
word_to_idx={word:idx + 1 for idx,word in enumerate(voc)} #indexed
word_to_idx["<pad>"]=0 #padding

#hyperparameters
vocab_size=len(word_to_idx)
embedding_dim=8
hidden_dim=16
output_dim=1

#sentences to padded tensors
def prepare_sequence(text,mapping,max_len=5):
    idxs=[mapping[w]for w in text.split()]
    if len(idxs)<max_len:
         idxs += [0]*(max_len - len(idxs))
    return torch.tensor(idxs[:max_len],dtype=torch.long)  
inputs=torch.stack([prepare_sequence(s,word_to_idx) for s in sentences])  
targets=torch.tensor(lbl,dtype=torch.float32).unsqueeze(1)      

# Define RNN/LSTM Model architecture 
class SentimentRNN(nn.Module):
     def __init__(self,vocab_size,embed_dim,hidden_dim,out_dim):
          super(SentimentRNN,self).__init__()
          self.embedding=nn.Embedding(vocab_size,embed_dim,padding_idx=0)
          self.rnn=nn.RNN(embed_dim,hidden_dim,batch_first=True)
          self.fc=nn.Linear(hidden_dim,out_dim)

     def forward(self,text):
        embedded=self.embedding(text)
        rnn_out,hidden=self.rnn(embedded)
        last_hidden=hidden.squeeze(0)
        return torch.sigmoid(self.fc(last_hidden))

#4.Instantiate and train model
model=SentimentRNN(vocab_size,embedding_dim,hidden_dim,output_dim)   
criterion=nn.BCELoss()
optimizer=optim.Adam(model.parameters(),lr=0.1)  
print("Sequential RNN Model----")
for epoch in range(20):
    model.train()
    optimizer.zero_grad()
    preds=model(inputs)
    loss=criterion(preds,targets)
    loss.backward()
    optimizer.step()
    if (epoch +1)% 5== 0:
        print(f"Epoch {epoch+1:02d}|Loss:{loss.item():.3f}")

#5. Make Predictions
model.eval()
with torch.no_grad():
    test_sen="I love this"
    test_tensor=torch.tensor([[word_to_idx.get(w,0)for w in test_sen.split()] + [0,0]],dtype=torch.long)
    pred=model(test_tensor).item()
    senti="Positive" if pred>0.5 else "Negative"
    print(f"Test Review :'{test_sen}' -> Predicted:{senti} ({pred:.4f})")




