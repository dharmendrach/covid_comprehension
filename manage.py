import os
import tqdm
import textwrap
import json
import prettytable
import logging
import pickle
import torch
import warnings
warnings.simplefilter('ignore')

import pandas as pd
from sentence_transformers import SentenceTransformer
from transformers import pipeline

from covid.utils.process_data import generate_clean_csv
from covid.src.ranker import rank_with_bert
from covid.src.comprehension import comprehend_with_bert

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.DEBUG)

BIORXIV_PATH = 'data/biorxiv_medrxiv/biorxiv_medrxiv/'
COMM_USE_PATH = 'data/comm_use_subset/comm_use_subset/'
NONCOMM_USE_PATH = 'data/noncomm_use_subset/noncomm_use_subset/'
CUSTOM_PATH = 'data/custom_license/custom_license/'
METADATA_PATH = 'data/metadata.csv'

DATA_PATH = 'data'
MODELS_PATH = 'models'
RANK_MODE = 'bert'
RANK_USING = 'abstract'
MODEL_NAME = 'scibert-nli'
COMPREHENSION_MODEL="distilbert-base-uncased-distilled-squad"
COMPREHENSION_TOKENIZER="distilbert-base-uncased"
use_gpu = -1
CORPUS_PATH = os.path.join(DATA_PATH, 'corpus.pkl')
MODEL_PATH = os.path.join(MODELS_PATH, MODEL_NAME)
EMBEDDINGS_PATH = os.path.join(DATA_PATH, f'{MODEL_NAME}-{RANK_USING}-embeddings.pkl')

logging.info(f"Ranking with {RANK_MODE}, using model: {MODEL_NAME}")


def cache_corpus(mode):

    biorxiv_df = generate_clean_csv(BIORXIV_PATH, METADATA_PATH, 'biorxiv', DATA_PATH, mode)
    comm_use_df = generate_clean_csv(COMM_USE_PATH, METADATA_PATH, 'comm_use', DATA_PATH, mode)
    noncomm_use_df = generate_clean_csv(NONCOMM_USE_PATH, METADATA_PATH, 'noncomm_use', DATA_PATH, mode)
    custom_df = generate_clean_csv(CUSTOM_PATH, METADATA_PATH, 'custom', DATA_PATH, mode)

    corpus = pd.concat([biorxiv_df, comm_use_df, noncomm_use_df, custom_df], ignore_index=True)

    logging.info("Creating corpus by combining: biorxiv, comm_use, noncomm_use, custom data")
    with open(CORPUS_PATH, 'wb') as file:
        pickle.dump(corpus, file)
    return corpus


if __name__ == '__main__':
    if not os.path.exists(CORPUS_PATH):
        logging.info("If the RANK_USING mode is modified means delete the cache and recreate it.")
        logging.info("Caching the corpus for future use...")
        corpus = cache_corpus(RANK_USING)
    else:
        logging.info("Loading the corpus from", CORPUS_PATH, '...')
        with open(CORPUS_PATH, 'rb') as corpus_pt:
            corpus = pickle.load(corpus_pt)

    model = SentenceTransformer(MODEL_PATH)

    if RANK_USING == "abstract":
        rank_corpus = corpus['abstract'].values
    elif RANK_USING == "text":
        rank_corpus = corpus['text'].values
    elif RANK_USING == "title":
        rank_corpus = corpus['title'].values
    else:
        raise AttributeError("Ranking should be with abstract, text (or) title are only supported")

    if not os.path.exists(EMBEDDINGS_PATH):
        logging.info("Computing and caching model embeddings for future use...")
        embeddings = model.encode(rank_corpus, show_progress_bar=True)
        with open(EMBEDDINGS_PATH, 'wb') as file:
            pickle.dump(embeddings, file)
    else:
        logging.info("Loading model embeddings from", EMBEDDINGS_PATH, '...')
        with open(EMBEDDINGS_PATH, 'rb') as file:
            embeddings = pickle.load(file)


    comprehension_model = pipeline("question-answering", model=COMPREHENSION_MODEL, tokenizer=COMPREHENSION_TOKENIZER, device=use_gpu)
    while True:
        query = input('\nAsk your question: ')
        rank_results = rank_with_bert(query, model, corpus, embeddings)
        comprehend_results = comprehend_with_bert(comprehension_model, query, rank_results)
        print(rank_results)
        print('*' * 100)
        print(comprehend_results)
