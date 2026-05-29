#Word Embedding and Word to Vector (Word2Vec)
import nltk
from gensim.models import Word2Vec
from nltk.tokenize import sent_tokenize,word_tokenize

#Ensure nltk is downloaded
nltk.download('punkt',quiet=True)
nltk.download('punkt_tab',quiet=True)

#1.Sample Corpus
corpus=[
    "Word embeddings capture semantic meanings of words.",
    "4 days continuously holidays",
    "Let be a business woman",
    "Grocery and clothes ",
    "Blowing flowers attracts all",
    "Word2Vec is a popular algorithm for word embeddings.",
    "Natural language processing uses word embeddings"

]

#2.Preprocessing & Tokenization
data=[]
for sen in corpus:
    tokens=[word.lower() for word in word_tokenize(sen)]
    data.append(tokens)

#3.Trainig model
model=Word2Vec(sentences=data,vector_size=100,window=5,min_count=1,sg=1,epochs=10)

#4.Exploring embeddings
word="embeddings"
if word in model.wv:
    print(f"Top 3 most similar words to '{word}':")
    similar_words=model.wv.most_similar(word,topn=3)
    for similar_words,similarity in similar_words:
        print(f" -{similar_words} (score:{similarity:.4f})")

#Get vector representation
vect_re=model.wv[word]        
print(f"Shape of vector for '{word}': {vect_re.shape} ")

#5.Save and Load model
model.save("word2vec.model")
loaded_model=Word2Vec.load("word2vec.model")
print(loaded_model)





