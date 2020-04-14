from covid.src.comprehension import comprehend_with_bert, show_comprehension_results
from covid.src.ranker import rank_with_bert, show_ranking_results, paragraph_ranking
from covid.utils.process_data import generate_clean_csv
from transformers import pipeline
from sentence_transformers import SentenceTransformer
import pandas as pd
import os
import logging
import pickle
import warnings
warnings.simplefilter('ignore')


logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.DEBUG)

# path to datasets
DATA_PATH = 'data'
BIORXIV_PATH = 'data/biorxiv_medrxiv/biorxiv_medrxiv/'
COMM_USE_PATH = 'data/comm_use_subset/comm_use_subset/'
NONCOMM_USE_PATH = 'data/noncomm_use_subset/noncomm_use_subset/'
CUSTOM_PATH = 'data/custom_license/custom_license/'
METADATA_PATH = 'data/metadata.csv'

MODELS_PATH = 'models'
RANK_USING = 'abstract'     # rank using: abstract(default) / title / text
MODEL_NAME = 'scibert-nli'  # model: scibert-nli / biobert-nli
# model used for comprehension
COMPREHENSION_MODEL = "distilbert-base-uncased-distilled-squad"
# tokenizer for comprehension
COMPREHENSION_TOKENIZER = "distilbert-base-uncased"
# use the gpu
use_gpu = -1
# processed corpus path
CORPUS_PATH = os.path.join(DATA_PATH, 'corpus.pkl')
# path to the saved model
MODEL_PATH = os.path.join(MODELS_PATH, MODEL_NAME)
EMBEDDINGS_PATH = os.path.join(
    DATA_PATH, f'{MODEL_NAME}-{RANK_USING}-embeddings.pkl')         # path to save the computed embeddings

logging.info(f"Ranking with {RANK_USING}, using model: {MODEL_NAME}")


def cache_corpus(mode):
    """
    For each datapath, clean the data and cache the cleaned data.

    Parameters
    ----------
    mode: str
        A string indicating the mode of ranking. (abstract / title / text)
    """
    biorxiv_df = generate_clean_csv(
        BIORXIV_PATH, METADATA_PATH, 'biorxiv', DATA_PATH, mode)
    comm_use_df = generate_clean_csv(
        COMM_USE_PATH, METADATA_PATH, 'comm_use', DATA_PATH, mode)
    noncomm_use_df = generate_clean_csv(
        NONCOMM_USE_PATH, METADATA_PATH, 'noncomm_use', DATA_PATH, mode)
    custom_df = generate_clean_csv(
        CUSTOM_PATH, METADATA_PATH, 'custom', DATA_PATH, mode)

    corpus = pd.concat(
        [biorxiv_df, comm_use_df, noncomm_use_df, custom_df], ignore_index=True)

    logging.info(
        "Creating corpus by combining: biorxiv, comm_use, noncomm_use, custom data")
    with open(CORPUS_PATH, 'wb') as file:
        pickle.dump(corpus, file)
    return corpus


if __name__ == '__main__':
    if not os.path.exists(CORPUS_PATH):
        logging.info(
            "If the RANK_USING mode is modified means delete the cache and recreate it.")
        logging.info("Caching the corpus for future use...")
        corpus = cache_corpus(RANK_USING)
    else:
        logging.info("Loading the corpus from", CORPUS_PATH, '...')
        with open(CORPUS_PATH, 'rb') as corpus_pt:
            corpus = pickle.load(corpus_pt)

    # model used for document ranking and paragraph ranking
    # default model is scibert-nli
    model = SentenceTransformer(MODEL_PATH)

    if RANK_USING == "abstract":
        rank_corpus = corpus['abstract'].values
    elif RANK_USING == "text":
        rank_corpus = corpus['text'].values
    elif RANK_USING == "title":
        rank_corpus = corpus['title'].values
    else:
        raise AttributeError(
            "Ranking should be with abstract, text (or) title are only supported")

    # computing, caching and loading embeddings for ranking purpose
    if not os.path.exists(EMBEDDINGS_PATH):
        logging.info(
            "Computing and caching model embeddings for future use...")
        embeddings = model.encode(rank_corpus, show_progress_bar=True)
        with open(EMBEDDINGS_PATH, 'wb') as file:
            pickle.dump(embeddings, file)
    else:
        logging.info("Loading model embeddings from", EMBEDDINGS_PATH, '...')
        with open(EMBEDDINGS_PATH, 'rb') as file:
            embeddings = pickle.load(file)

    # model used for comprehension
    comprehension_model = pipeline("question-answering", model=COMPREHENSION_MODEL,
                                   tokenizer=COMPREHENSION_TOKENIZER, device=use_gpu)

    while True:
        query = input('\nAsk your question: ')

        # ranking the documents using pre-calculated embeddings and query
        document_rank_results = rank_with_bert(query, model, corpus, embeddings)

        # ranking the paragraphs of the top retrieved documents
        paragraph_rank_results = paragraph_ranking(query, model, document_rank_results)

        # comprehending the top paragraphs of the retrieved documents for finding answer
        comprehend_results = comprehend_with_bert(comprehension_model, query, paragraph_rank_results)

        show_ranking_results(document_rank_results)
        print('*' * 100)
        show_comprehension_results(comprehend_results["results"])
