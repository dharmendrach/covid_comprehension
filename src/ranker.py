import os
import tqdm
import textwrap
import json
import prettytable
import logging
import pickle
import warnings
warnings.simplefilter('ignore')

import pandas as pd
import scipy
from sentence_transformers import SentenceTransformer

from ..utils.process_data import generate_clean_csv

logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
logger = logging.getLogger(__name__)

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
CORPUS_PATH = os.path.join(DATA_PATH, 'corpus.pkl')
MODEL_PATH = os.path.join(MODELS_PATH, MODEL_NAME)
EMBEDDINGS_PATH = os.path.join(DATA_PATH, f'{MODEL_NAME}-{RANK_USING}-embeddings.pkl')

logging.info(f"Ranking with {RANK_MODE}, using model: {MODEL_NAME}")


def cache_corpus():

    biorxiv_df = generate_clean_csv(BIORXIV_PATH, METADATA_PATH, 'biorxiv', DATA_PATH)
    comm_use_df = generate_clean_csv(COMM_USE_PATH, METADATA_PATH, 'comm_use', DATA_PATH)
    noncomm_use_df = generate_clean_csv(NONCOMM_USE_PATH, METADATA_PATH, 'noncomm_use', DATA_PATH)
    custom_df = generate_clean_csv(CUSTOM_PATH, METADATA_PATH, 'custom', DATA_PATH)

    corpus = pd.concat([biorxiv_df, comm_use_df, noncomm_use_df, custom_df], ignore_index=True)

    logging.info(f"Creating corpus by combining: {biorxiv}, {comm_use}, {noncomm_use}, {custom} data")
    with open(CORPUS_PATH, 'wb') as file:
        pickle.dump(corpus, file)
    return corpus


def rank_with_bert(query, model, corpus, corpus_embed, top_k=5):
    queries = [query]
    query_embeds = model.encode(queries, show_progress_bar=False)
    for query, query_embed in zip(queries, query_embeds):
        distances = scipy.spatial.distance.cdist([query_embed], corpus_embed, "cosine")[0]
        distances = zip(range(len(distances)), distances)
        distances = sorted(distances, key=lambda x: x[1])
        results = []
        for count, (idx, distance) in enumerate(distances[0:top_k]):
            results.append([count + 1, corpus.iloc[idx]['abstract'].strip(), round(1 - distance, 4)])
    return results


def show_answers(results):
    table = prettytable.PrettyTable(
        ['Rank', 'Abstract', 'Score']
    )
    for res in results:
        rank = res[0]
        text = res[1]
        text = textwrap.fill(text, width=75)
        text = text + '\n\n'
        score = res[2]
        table.add_row([
            rank,
            text,
            score
        ])
    print('\n')
    print(str(table))
    print('\n')


if __name__ == '__main__':
    if not os.path.exists(CORPUS_PATH):
        print("Caching the corpus for future use...")
        corpus = cache_corpus()
    else:
        print("Loading the corpus from", CORPUS_PATH, '...')
        with open(CORPUS_PATH, 'rb') as corpus_pt:
            corpus = pickle.load(corpus_pt)

    model = SentenceTransformer(MODEL_PATH)

    if RANK_USING == "abstract":
        rank_corpus = corpus['abstract'].values
    elif RANK_USING == "text":
        rank_corpus = corpus['text'].values
    else:
        raise AttributeError("Ranking with abstract (or) text only supported")

    if not os.path.exists(EMBEDDINGS_PATH):
        print("Computing and caching model embeddings for future use...")
        embeddings = model.encode(rank_corpus, show_progress_bar=True)
        with open(EMBEDDINGS_PATH, 'wb') as file:
            pickle.dump(embeddings, file)
    else:
        print("Loading model embeddings from", EMBEDDINGS_PATH, '...')
        with open(EMBEDDINGS_PATH, 'rb') as file:
            embeddings = pickle.load(file)

    while True:
        query = input('\nAsk your question: ')
        results = rank_with_bert(query, model, corpus, embeddings)
        show_answers(results)
