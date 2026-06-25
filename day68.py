#Day-68:BERT(Bidirectional Encoder Representations from Transformers) Fine-Tunning
import torch
from transformers import BertTokenizer,BertForSequenceClassification,Trainer,TrainingArguments
from datasets import Dataset

#1.Load a pretrained BERT Model+tokenizer
model_name="bert-base-uncased"
tokenizer=BertTokenizer.from_pretrained(model_name)
model=BertForSequenceClassification.from_pretrained(model_name,num_labels=2)

#2.Sample dataset(1:pos,0:neg)
texts=[
    "I am fond of Python",
    "C is more complex",
    "Best time utilization",
    "Just English language",
    "Flow a strict rule but need as basic "
]
labels=[1,0,1,1,0]

#3.Tokenization
encodings=tokenizer(
    texts,padding='max_length',truncation=True,
    max_length=32,return_tensors='pt'
)

#4.Wrap into a Dataset object
class SimpleDataset(torch.utils.data.Dataset):
    def __init__(self,encodings,labels):
        self.encodings=encodings
        self.labels=labels
    def __getitem__(self,idx):
        item={k: v[idx] for k,v in self.encodings.items() }
        item['labels']=torch.tensor(self.labels[idx])
        return item
    def __len__(self):
        return len(self.labels)
train_dataset=SimpleDataset(encodings,labels)

#5.Define Training settings
training_args=TrainingArguments(
    output_dir='./bert_results',
    num_train_epochs=3,
    per_device_train_batch_size=2,
    logging_steps=1,
    save_strategy='no'
)

#6.Trainer handle
trainer=Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset
)
trainer.train()

#7.Predict on new text
def predict(text):
    device=model.device
    inputs=tokenizer(text,padding='max_length',truncation=True,
                     max_length=32,return_tensors='pt')
    inputs={k:v.to(device) for k,v in inputs.items()} #move inputs to sam edevice as CPU or GPU
    with torch.no_grad():
        outputs=model(**inputs)
        pred=torch.argmax(outputs.logits,dim=1).item()
    return "Positive" if pred==1 else "Negative"

print(predict("This is ineteresting"))
print(predict("It is very complex"))