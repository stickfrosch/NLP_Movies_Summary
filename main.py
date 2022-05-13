import pandas as pd
from nltk import word_tokenize,sent_tokenize
from nltk.stem import WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()
from nltk.corpus import stopwords
import string
string.punctuation
stop = stopwords.words('english')
from nltk.stem.porter import PorterStemmer
porter_stemmer = PorterStemmer()

pd.set_option('display.max_rows', 10)
pd.set_option('display.max_columns', 10)
pd.set_option('display.width', 1000)

import nltk
import ssl

#try:
#    _create_unverified_https_context = ssl._create_unverified_context
#except AttributeError:
#    pass
#else:
#    ssl._create_default_https_context = _create_unverified_https_context

#nltk.download()

#nltk.download('punkt')



text = "Hello there! Welcome to the programming world."
print(word_tokenize(text))

movies = pd.read_csv("movies_metadata.csv", delimiter=';', encoding= "ISO-8859-1")
# Removing last 30000 rows
movies = movies.iloc[:-30000]

def remove_punctuation(text):
    punctuationfree="".join([i for i in text if i not in string.punctuation])
    return punctuationfree

#storing the puntuation free text
movies['overview']= movies['overview'].astype(str).apply(lambda x:remove_punctuation(x))
movies['overview']= movies['overview'].apply(lambda x: x.lower())
movies['overview_without_stopwords'] = movies['overview'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
movies["token"] = movies["overview_without_stopwords"].apply(word_tokenize)

#defining the function for lemmatization
def lemmatizer(text):
    lemm_text = [wordnet_lemmatizer.lemmatize(word) for word in text]
    return lemm_text
movies['overview_lemmatized']=movies['token'].apply(lambda x:lemmatizer(x))

#print(movies.head())

movies = movies[["original_title","overview_lemmatized"]]
print(movies.head())

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def get_top_k_articles(query, docs, k=2):
    # Initialize a vectorizer that removes English stop words
    vectorizer = TfidfVectorizer(analyzer="word", stop_words='english')

    # Create a corpus of query and documents and convert to TFIDF vectors
    query_and_docs = [query] + docs
    matrix = vectorizer.fit_transform(query_and_docs)

    # Holds our cosine similarity scores
    scores = []

    # The first vector is our query text, so compute the similarity of our query against all document vectors
    for i in range(1, len(query_and_docs)):
        scores.append(cosine_similarity(matrix[0], matrix[i])[0][0])

    # Sort list of scores and return the top k highest scoring documents
    sorted_list = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    top_doc_indices = [x[0] for x in sorted_list[:k]]
    top_docs = [docs[x] for x in top_doc_indices]

    return top_docs