import pandas as pd
import numpy as np
from nltk.stem import WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()
from nltk.corpus import stopwords
import string
string.punctuation
stop = stopwords.words('english')
from nltk.stem.porter import PorterStemmer
porter_stemmer = PorterStemmer()
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import torch
from transformers import BertTokenizer, BertForQuestionAnswering

pd.set_option('display.max_rows', 10)
pd.set_option('display.max_columns', 10)
pd.set_option('display.width', 1000)


movies = pd.read_csv("movies_metadata.csv", delimiter=';', encoding= "ISO-8859-1")
# Removing last 40000 rows
movies = movies.iloc[:-40000]

def remove_punctuation(text):
    punctuationfree="".join([i for i in text if i not in string.punctuation])
    return punctuationfree

#storing the puntuation free text
movies['overview']= movies['original_title'] + movies['overview']
movies['overview']= movies['overview'].astype(str).apply(lambda x:remove_punctuation(x))
movies['overview']= movies['overview'].apply(lambda x: x.lower())
movies['overview_without_stopwords'] = movies['overview'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
#movies["token"] = movies["overview_without_stopwords"].apply(word_tokenize)
print(movies.head())

#docs = movies['overview_without_stopwords']
docs = movies['overview_without_stopwords'].tolist()
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(docs)
Y = vectorizer.get_feature_names_out()
df1 = pd.DataFrame(Y)
print(df1)
#tfidf_matrix = tfidf_vectorizer.fit_transform(tuple(movies['overview_without_stopwords']))

def ask_question(question):
    query_vect = vectorizer.transform([question])
    similarity = cosine_similarity(query_vect, X)
    max_similarity = np.argmax(similarity, axis=None)

    print('Your question:', question)
    print('Similarity: {:.2%}'.format(similarity[0, max_similarity]))
    print('Answer:', movies.iloc[max_similarity]['overview'])
    print('Moviename:', movies.iloc[max_similarity]['original_title'])

ask_question('What happens at Lord of the Rings')