import pandas
import spacy
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import namedtuple
import gensim.utils
import re
import string
import multiprocessing
from gensim.models import Doc2Vec
from gensim.models import TfidfModel
from gensim.corpora import Dictionary
from gensim import similarities 

def get_args():
    parser = argparse.ArgumentParser(description='Newton Find Jobs')
    parser.add_argument('--num_jobs', type=int, default=50000)
    parser.add_argument('--query', default='software developer')
    parser.add_argument('--use_pretrained_model',default=False)
    args = parser.parse_args()
    return args


#@Spacy
def load_spacy():
    nlp = spacy.load('en')
    nlp.remove_pipe('tagger')
    nlp.remove_pipe('parser')

#@Tokenize
def spacy_tokenize(string):
  tokens = list()
  doc = nlp(string)
  for token in doc:
    tokens.append(token)
  return tokens

#@Normalize
def normalize(tokens):
  normalized_tokens = list()
  for token in tokens:
    if (token.is_alpha or token.is_digit):
      normalized = token.text.lower().strip()
      normalized_tokens.append(normalized)
  return normalized_tokens

#@Tokenize and normalize
def tokenize_normalize(string):
  return normalize(spacy_tokenize(string))  

def read_data(max_jobs):
    
    # read data as its coming in to save memory
    data = []
    title_company=[]
    description=[]
    with open(file_name) as f:
    for line in f:
        if len(data)>max_jobs:
            break
        data.append(json.loads(line))    
    
    return data

def preprocess(data):
    
    df = pd.DataFrame(data).set_index('id')
    df['job'] = df["title"]+ df["description"]
    description = np.array(df.job)
    job_title = np.array(df.title)
    
    jobs_tuple = namedtuple('jobs_tuple', 'words tags title line_id')
    
    # list of all jobs 
    all_jobs = []
    # for each line in all job description, it will add a tuple to list
    for line_id, line in enumerate(description):
        tokens = tokenize_normalize(line)
        title = job_title[line_no]
        all_jobs.append(jobs_tuple(tokens, title, line_id))
    
    return all_jobs


def get_similar_jobs(all_jobs):
    # create dictionary, corpus, and build model
    dct = Dictionary(doc.words for doc in all_jobs) 
    corpus = [dct.doc2bow(line.words) for line in all_jobs]  
    
    index = 0
    tokens = tokenize_normalize("python engineer")
    index = similarities.MatrixSimilarity([dct.doc2bow(tokens)],num_features=len(dct))
    similar = np.zeros((len(all_jobs)))

    for id, doc in enumerate(all_jobs):
        similar[id] = index[dct.doc2bow(doc.words)]
        
    ranks=[]
    for i in range(len(top)):
        ranks.append(data[all_jobs[top[i]].original_number])
        
    df = pd.DataFrame(ranks).set_index('id')
    df = df.applymap(lambda x: x.replace('\n', ''))
    
    print(df)
        
        
if __name__ == "__main__":
    
    #spacy
    load_spacy()
    args = get_args()
    #read data
    data = read_data(args.num_jobs)
    # preprocess data
    all_jobs = preprocess(data)
    # build tfidf model
    get_similar_jobs(all_jobs)
