#Day69-GPT and Text Generation
import torch 
from transformers import GPT2LMHeadModel,GPT2Tokenizer

#1.Load pretrained GPT-2
model_name="gpt2"
tokenizer=GPT2Tokenizer.from_pretrained(model_name)
model=GPT2LMHeadModel.from_pretrained(model_name)
model.eval()

#2.Covert prompt to token
prompt="Python Programming is"
input_ids=tokenizer.encode(prompt,return_tensors='pt')
print("Prompt tokens:",input_ids)

#3.Generate text by GREEDY decoding
greedy_output=model.generate(
    input_ids,
    max_length=25,
    num_return_sequences=1
)
print("greedy output:")
print(tokenizer.decode(greedy_output[0],skip_special_tokens=True))

#4.Generate using temperature and sampling
sampled_output=model.generate(
    input_ids,max_length=25,
    do_sample=True,
    temperature=0.8,
    top_k=50,
    top_p=0.95,
    num_return_sequences=3
)
print("Creative sample:")
for i,output in enumerate (sampled_output):
    text=tokenizer.decode(output,skip_special_tokens=True)
    print(f"{i+1}.{text}")

#5.Compare different temperature on same prompting
print("Comparing Temperatures:")
for temp in [0.2,0.3,1.3]:
    out=model.generate(input_ids,max_length=20,
                       do_sample=True,temperature=temp,top_k=50)
    text=tokenizer.decode(out[0],skip_special_tokens=True)
    print(f"temp:{temp}:{text}")