#day83: LoRA :Low-Rank Adaptation 
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType
import torch



# 1.Load a small base model 
model_name = "gpt2"
tokenizer  = AutoTokenizer.from_pretrained(model_name)
model      = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# 2.LoRA configuration

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=4,                         
    lora_alpha=16,               
    lora_dropout=0.1,
    target_modules=["c_attn"]    
)

# 3.Wrap model with LoRA 
lora_model = get_peft_model(model, lora_config)

# 4.Parameter training
total  = sum(p.numel() for p in lora_model.parameters())
trainable = sum(p.numel() for p in lora_model.parameters() if p.requires_grad)
print(f"Total parameters    : {total:,}")
print(f"Trainable parameters: {trainable:,}")
print(f"Trainable %         : {100*trainable/total:.2f}%")


# 5.Sample fine-tuning data
texts = [
    "Machine learning is the future of technology",
    "Neural networks learn from data automatically",
    "Deep learning powers modern AI applications",
    "Let read more"
]

# 6.Tokenize sample data
encodings = tokenizer(texts, padding=True, truncation=True,
                      max_length=32, return_tensors='pt')

# 7.Quick training demo
optimizer = torch.optim.AdamW(lora_model.parameters(), lr=1e-4)
lora_model.train()

for step, text in enumerate(texts):
    inputs = tokenizer(text, return_tensors='pt', max_length=32, truncation=True)
    inputs['labels'] = inputs['input_ids'].clone()  
    outputs = lora_model(**inputs)
    loss    = outputs.loss
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    print(f"Step {step+1} | Loss: {loss.item():.4f}")
