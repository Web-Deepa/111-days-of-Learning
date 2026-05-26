#Topic Modeling
import re
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.datasets import fetch_20newsgroups

#1.create data
cats=['sci.med','sci.space','rec.sport.hockey','talk.politics.guns']
data = fetch_20newsgroups(subset='train', categories=cats, remove=('headers','footers','quotes'))

def clean(text):
    text=text.lower()
    text=re.sub(r'\S+@\S+|http\S+|[^a-z\s]', '',text)
    return re.sub(r'\s+',' ',text).strip()
docs=[clean(t) for t in data.data[:200]] #use docs for speed
print(f"Documents:{len(docs)}")

#2.Vectorize(CountVectorizer for LDA)
cv=CountVectorizer(
    max_features=1000,
    min_df=3,
    max_df=0.85,
    stop_words='english'
)
x=cv.fit_transform(docs)
vocab=cv.get_feature_names_out()
print(f"Vocabulary size:{len(vocab)}")

#3.LDA Topic Modeling
print("LDA -4 Topics")
print("=" * 40)
lda=LatentDirichletAllocation(
    n_components=4,random_state=42,max_iter=20
)
lda.fit(x)

#4.Top words per topic
def print_topics(model,vocab,n_words=10):
    for i ,topic in enumerate(model.components_):
        top_words=[vocab[j] for j in topic.argsort()[-n_words:][::-1]]
        print(f"Topic {i+1}: {', '.join(top_words)}")
print_topics(lda,vocab)

#5.print("Topic assignment for New Documents--")
new_docs=[
    "NASA launched a new satellite",
    "Hockey team scored 3 goals",
    "Today is local holiday",
    "New rules are published"
]
new_vec=cv.transform([clean(d) for d in new_docs])
topic_dist=lda.transform(new_vec)
for doc,dist in zip(new_docs,topic_dist):
    top_topic=dist.argmax() + 1
    confidence=dist.max()
    print(f"  '{doc[:50]}'")
    print(f"   → Topic {top_topic}  (confidence: {confidence:.3f})")

#6.choosing number of topics(perplexity)
print("Choosing K topics-Perplexity Score---")
print(f"{'K topics':>10} {'Perplexity':>12}")
print("-" *30)
for k in [2,3,4,5,6]:
    m=LatentDirichletAllocation(n_components=k,random_state=42,max_iter=10)
    m.fit(x)
    print(f"{k:>10} {m.perplexity(x):>10.1f}")
print("(lower perplexity=better fit)")