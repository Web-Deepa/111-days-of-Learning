#Day70-HuggingFace Pipeline
from transformers import pipeline
#1.Sentiment Analysis
sentiment=pipeline("sentiment-analysis")
res=sentiment("I am fond of this ,so comfortable")
print("Sentiment:",res)

#2. Text Generation
generator=pipeline("text-generation",model="gpt2")
res=generator("Machine Learning is",max_length=20,num_return_sequences=1)
print("Generated texxt:",res[0]['generated_text'])

#3.Named Entity Recognition(NER)
ner=pipeline("ner",grouped_entities=True)
res=ner("Elon Musk SpaceX is bringing ipo for general public")
print("Entities found:")
for ent in res:
    print(f"{ent['word']}->{ent['entity_group']} (score=ent['score']):.2f")

#4.Question Answering
qa=pipeline("question-answering")
context="Python was created in 1990 by Guido van Rossum"
question="Who created python?"
res=qa(question=question,context=context)
print(f"Q:{question}")
print(f"A:{res['answer']}(confidence={res['score']:.2f})")

#5.Summarization
summarizer=pipeline("summarization")
long_text="Machine learning is a branch of AI which makes system learn from data,impoves from experience  perform predictions without explicitly programmed .It has increases important accross industries including healthcare,finance and transportion."
summary=summarizer(long_text,max_length=30,min_length=10,do_sample=False)
print("Summary:",summary[0]['summary_text'])

#6.Zero-Shot Classification
classifier=pipeline("zero-shot-classification")
res=classifier(
    "This new phone has an amazing camera and battery life.",
    candidate_labels=["technology","sports","cooking","politics"]
)
print("Zero-shot classification:")
for label,score in zip(res['labels'],res['scores']):
    print(f"{label:<12} {score:.3f}")

#7.Translation
translator=pipeline("translation_en_to_fr")
res=translator("Machine learning is amazing")
print("Translation(EN->FR):",res[0]['translation_text'])
