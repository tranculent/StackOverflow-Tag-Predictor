try:
    import google.colab
    IN_COLAB = True
except:
    IN_COLAB = False

if IN_COLAB:
    ! wget https://raw.githubusercontent.com/hse-aml/natural-language-processing/master/setup_google_colab.py -O setup_google_colab.py
    import setup_google_colab
    setup_google_colab.setup_week1() 
    
import sys
sys.path.append("..")
from common.download_utils import download_week1_resources

download_week1_resources()

from grader import Grader

grader = Grader()

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

from ast import literal_eval
import pandas as pd
import numpy as np

def read_data(filename):
    data = pd.read_csv(filename, sep='\t')
    data['tags'] = data['tags'].apply(literal_eval)
    return data
    
train = read_data('data/train.tsv')
validation = read_data('data/validation.tsv')
test = pd.read_csv('data/test.tsv', sep='\t')

train.head()

X_train, y_train = train['title'].values, train['tags'].values
X_val, y_val = validation['title'].values, validation['tags'].values
X_test = test['title'].values

import re

REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
STOPWORDS = set(stopwords.words('english'))

def text_prepare(text):
    """
        text: a string
        
        return: modified initial string
    """
    text = text.lower() # lowercase text
    text = re.sub(REPLACE_BY_SPACE_RE, ' ', text) # replace REPLACE_BY_SPACE_RE symbols by space in text
    text = re.sub(BAD_SYMBOLS_RE, '', text) # delete symbols which are in BAD_SYMBOLS_RE from text
    text = text.split()
    temp = []
    for word in text:
      if word in STOPWORDS:
        temp.append(word)

    for word in temp:
      text.remove(word)

    return " ".join(text)
    
def test_text_prepare():
    examples = ["SQL Server - any equivalent of Excel's CHOOSE function?",
                "How to free c++ memory vector<int> * arr?"]
    answers = ["sql server equivalent excels choose function", 
               "free c++ memory vectorint arr"]
    for ex, ans in zip(examples, answers):
        if text_prepare(ex) != ans:
            return "Wrong answer for the case: '%s'" % ex
    return 'Basic tests are passed.'
    
prepared_questions = []
for line in open('data/text_prepare_tests.tsv', encoding='utf-8'):
    line = text_prepare(line.strip())
    prepared_questions.append(line)
text_prepare_results = '\n'.join(prepared_questions)

grader.submit_tag('TextPrepare', text_prepare_results)

X_train = [text_prepare(x) for x in X_train]
X_val = [text_prepare(x) for x in X_val]
X_test = [text_prepare(x) for x in X_test]

# Dictionary of all tags from train corpus with their counts.
tags_counts = {}
# Dictionary of all words from train corpus with their counts.
words_counts = {}

temp = {}
for posttag in train['tags']:
  for tag in posttag:
    if tag in temp:
      temp[tag] = temp.get(tag) + 1
    else:
      temp[tag] = 1

temp = sorted(temp.items(), key=lambda x: x[1], reverse=True)

i = 0
for tag in temp:
  if len(tags_counts) < 3:
    tags_counts[tag[0]] = temp[i][1]
    i+=1

temp = {}
for title in train['title']:
  for word in title.split():
    if word in temp:
      temp[word] = temp.get(word) + 1
    else:
      temp[word] = 1

temp = sorted(temp.items(), key=lambda x: x[1], reverse=True)

i = 0
for word in temp:
  if len(words_counts) < 3:
    words_counts[word[0]] = temp[i][1]
    i+=1

print(tags_counts)
print(words_counts)

most_common_tags = sorted(tags_counts.items(), key=lambda x: x[1], reverse=True)[:3]
most_common_words = sorted(words_counts.items(), key=lambda x: x[1], reverse=True)[:3]

grader.submit_tag('WordsTagsCount', '%s\n%s' % (','.join(tag for tag, _ in most_common_tags), 
                                                ','.join(word for word, _ in most_common_words)))
                                                
DICT_SIZE = 5000

WORDS_TO_INDEX = {}
counter = 0
for title in train['title']:
  for word in title.split():
    if word not in WORDS_TO_INDEX: 
      WORDS_TO_INDEX[word] = counter
      counter+=1

# print(WORDS_TO_INDEX)

INDEX_TO_WORDS = {} 
counter = 0
for title in train['title']:
  for word in title.split():
    if word not in INDEX_TO_WORDS.values():
      INDEX_TO_WORDS[counter] = word
      counter+=1

# print(INDEX_TO_WORDS)
  
ALL_WORDS = WORDS_TO_INDEX.keys()

def my_bag_of_words(text, words_to_index, dict_size):
    """
        text: a string
        dict_size: size of the dictionary
        
        return a vector which is a bag-of-words representation of 'text'
    """
    result_vector = np.zeros(dict_size)
    
    for word in text.split():
      if word in words_to_index.keys():
        if words_to_index[word] < dict_size:
          result_vector[words_to_index[word]] = 1

    print(result_vector)
    return result_vector
    
 def test_my_bag_of_words():
    words_to_index = {'hi': 0, 'you': 1, 'me': 2, 'are': 3}
    examples = ['hi how are you']
    answers = [[1, 1, 0, 1]]
    for ex, ans in zip(examples, answers):
        if (my_bag_of_words(ex, words_to_index, 4) != ans).any():
            return "Wrong answer for the case: '%s'" % ex
    return 'Basic tests are passed.'

from scipy import sparse as sp_sparse

X_train_mybag = sp_sparse.vstack([sp_sparse.csr_matrix(my_bag_of_words(text, WORDS_TO_INDEX, DICT_SIZE)) for text in X_train])
X_val_mybag = sp_sparse.vstack([sp_sparse.csr_matrix(my_bag_of_words(text, WORDS_TO_INDEX, DICT_SIZE)) for text in X_val])
X_test_mybag = sp_sparse.vstack([sp_sparse.csr_matrix(my_bag_of_words(text, WORDS_TO_INDEX, DICT_SIZE)) for text in X_test])
print('X_train shape ', X_train_mybag.shape)
print('X_val shape ', X_val_mybag.shape)
print('X_test shape ', X_test_mybag.shape)

row = X_train_mybag[10].toarray()[0]
non_zero_elements_count = 0

for x in row:
  if x == 0:
    non_zero_elements_count += 1

print(non_zero_elements_count)

grader.submit_tag('BagOfWords', str(non_zero_elements_count))

from sklearn.feature_extraction.text import TfidfVectorizer
def tfidf_features(X_train, X_val, X_test):
    """
        X_train, X_val, X_test — samples        
        return TF-IDF vectorized representation of each sample and vocabulary
    """
    # Create TF-IDF vectorizer with a proper parameters choice
    # Fit the vectorizer on the train set
    # Transform the train, test, and val sets and return the result
    
    tfidf_vectorizer= TfidfVectorizer(sublinear_tf=True, max_df=0.9, min_df=0.05, analyzer="word", stop_words=STOPWORDS)
    
    tfidf_vectorizer.fit(X_train)
    tfidf_vectorizer.transform(X_train)
    tfidf_vectorizer.transform(X_test)
    tfidf_vectorizer.transform(X_val)
    
    return X_train, X_val, X_test, tfidf_vectorizer.vocabulary_
    
X_train_tfidf, X_val_tfidf, X_test_tfidf, tfidf_vocab = tfidf_features(X_train, X_val, X_test)
tfidf_reversed_vocab = {i:word for word,i in tfidf_vocab.items()}
