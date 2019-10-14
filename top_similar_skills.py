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


def get_args():
    parser = argparse.ArgumentParser(description='Newton Similar Skills')
    parser.add_argument('--num_jobs', type=int, default=50000)
    parser.add_argument('--query', default='python')
    parser.add_argument('--use_pretrained_model',default=True)
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
    # lower all cases and remove punctuations  
    if (token.is_alpha or token.is_digit):
      normalized = token.text.lower().strip()
      normalized_tokens.append(normalized)
  return normalized_tokens

#@Tokenize and normalize
def tokenize_normalize(string):
  return normalize(spacy_tokenize(string))  

def read_data(max_jobs):
    
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
        
def doc2vec_model(all_jobs):
    
    #create model
    doc2vec_model = Doc2Vec(dm=1, 
                size=300,
                window=10,
                seed=1,
                min_count=3,
                dbow_words=1,
                sample=1e-3,
                workers = 4)
    
    # build vocab
    doc2vec_model.build_vocab(jobs)
    
    #train_model
    doc2vec_model.train(jobs, 
                    total_examples=doc2vec_model.corpus_count, 
                    epochs=50, 
                    start_alpha=0.01, 
                    end_alpha=0.01)
    #save model
    doc2vec_model.save("doc2vec_model")

def get_similar_skills(skill):
    #load and run model
    doc2vec_model = Doc2Vec.load("model")
    #retrieve most similar words using cosine similarity
    most_similar = doc2vec_model.wv.most_similar_cosmul(positive = [skill])
    # get the first part of the tuple (strings of similar skills)
    top_skills = [x[0] for x in most_similar]
    print(top_skills)

if __name__ == "__main__":
    if use_pretrained_model:
        #spacy
        load_spacy()
        args = get_args()
        #read data
        data = read_data(args.num_jobs)
        # preprocess data
        all_jobs = preprocess(data)
        # doc2vec model if model is not saved
        doc2vec_model(all_jobs)
    # return top skills
    get_similar_skills(args.query)
