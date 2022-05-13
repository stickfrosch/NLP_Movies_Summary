import pandas as pd
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
movies['overview']= movies['overview'].astype(str).apply(lambda x:remove_punctuation(x))
movies['overview']= movies['overview'].apply(lambda x: x.lower())
movies['overview_without_stopwords'] = movies['overview'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
#movies["token"] = movies["overview_without_stopwords"].apply(word_tokenize)
print(movies.head())

docs = movies['overview_without_stopwords']


def get_top_k_articles(query, docs, k=2):
    # Initialize a vectorizer that removes English stop words
    vectorizer = TfidfVectorizer(analyzer="word", stop_words='english')

    # Create a corpus of query and documents and convert to TFIDF vectors
    query_and_docs = [query] + docs
    matrix = vectorizer.fit_transform(query_and_docs)

    # Holds our cosine similarity scores
    scores = []
    print(scores)

    # The first vector is our query text, so compute the similarity of our query against all document vectors
    for i in range(1, len(query_and_docs)):
        scores.append(cosine_similarity(matrix[0], matrix[i])[0][0])

    # Sort list of scores and return the top k highest scoring documents
    sorted_list = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    top_doc_indices = [x[0] for x in sorted_list[:k]]
    top_docs = [docs[x] for x in top_doc_indices]

    return top_docs

model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')


def answer_question(question, answer_text):
    input_ids = tokenizer.encode(question, answer_text, max_length=512)

    # ======== Set Segment IDs ========
    # Search the input_ids for the first instance of the `[SEP]` token.
    sep_index = input_ids.index(tokenizer.sep_token_id)

    # The number of segment A tokens includes the [SEP] token istelf.
    num_seg_a = sep_index + 1

    # The remainder are segment B.
    num_seg_b = len(input_ids) - num_seg_a

    # Construct the list of 0s and 1s.
    segment_ids = [0] * num_seg_a + [1] * num_seg_b

    # There should be a segment_id for every input token.
    assert len(segment_ids) == len(input_ids)

    outputs = model(torch.tensor([input_ids]),  # The tokens representing our input text.
                    token_type_ids=torch.tensor([segment_ids]),
                    # The segment IDs to differentiate question from answer_text
                    return_dict=True)

    start_scores = outputs.start_logits
    end_scores = outputs.end_logits

    # ======== Reconstruct Answer ========
    # Find the tokens with the highest `start` and `end` scores.
    answer_start = torch.argmax(start_scores)
    answer_end = torch.argmax(end_scores)

    # Get the string versions of the input tokens.
    tokens = tokenizer.convert_ids_to_tokens(input_ids)

    # Start with the first token.
    answer = tokens[answer_start]

    # Select the remaining answer tokens and join them with whitespace.
    for i in range(answer_start + 1, answer_end + 1):

        # If it's a subword token, then recombine it with the previous token.
        if tokens[i][0:2] == '##':
            answer += tokens[i][2:]

        # Otherwise, add a space then the token.
        else:
            answer += ' ' + tokens[i]

    print('Answer: "' + answer + '"')

# Enter our query here
query = "Who Woody the toy?"
#query = "What else does the bassist for Death From Above play?"
#query = "What projects is Jesse Keeler involved in?"

# Segment our documents
#segmented_docs = segment_documents(docs, 450)

# Retrieve the top k most relevant documents to the query
candidate_docs = get_top_k_articles(query, docs, 3)

# Return the likeliest answers from each of our top k most relevant documents in descending order
for i in candidate_docs:
  answer_question(query, i)
  print ("Reference Document: ", i)

