# Day 81:T5 :Text-To-Text Transfer Transformer


from transformers import T5Tokenizer, T5ForConditionalGeneration

# 1.Load pretrained T5 
tokenizer = T5Tokenizer.from_pretrained("t5-small")
model     = T5ForConditionalGeneration.from_pretrained("t5-small")
model.eval()

def generate(prompt, max_length=50):
    inputs  = tokenizer(prompt, return_tensors="pt", max_length=500, truncation=True)
    outputs = model.generate(inputs.input_ids, max_length=max_length, num_beams=4)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# 2. — Translation 
result = generate("translate English to French: Machine learning is powerful")
print(f"Translation : {result}")

# 3. — Summarization
long_text = "summarize: Machine learning is a branch of AI that enables systems \
to learn from data and improve from experience without being explicitly programmed. \
It focuses on developing computer programs that can access data and use it to learn."
result = generate(long_text, max_length=30)
print(f"Summary     : {result}")

# 4.  Grammar correction
result = generate("grammar: He go to school every days and learn many things")
print(f"Grammar fix : {result}")

# 5.  Question answering
result = generate("question: Who created Python? context: Python was created by Guido van Rossum in 1991")
print(f"QA answer   : {result}")
