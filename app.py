import os
import pickle
import logging

import pandas as pd

from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from transformers import pipeline

from covid.utils.process_data import generate_clean_csv
from covid.src.ranker import rank_with_bert, paragraph_ranking
from covid.src.comprehension import comprehend_with_bert

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


# path to datasets
DATA_PATH = 'data'
BIORXIV_PATH = 'data/biorxiv_medrxiv/biorxiv_medrxiv/'
COMM_USE_PATH = 'data/comm_use_subset/comm_use_subset/'
NONCOMM_USE_PATH = 'data/noncomm_use_subset/noncomm_use_subset/'
CUSTOM_PATH = 'data/custom_license/custom_license/'
METADATA_PATH = 'data/metadata.csv'

MODELS_PATH = 'models'
RANK_USING = 'abstract'     # rank using: abstract(default) / title / text
MODEL_NAME = 'scibert-nli'  # model: scibert-nli / biobert-nli / covidbert-nli
COMPREHENSION_MODEL = "distilbert-base-uncased-distilled-squad"     # model used for comprehension
COMPREHENSION_TOKENIZER = "distilbert-base-uncased"                 # tokenizer for comprehension
use_gpu = -1                                                        # use the gpu
CORPUS_PATH = os.path.join(DATA_PATH, 'corpus.pkl')                 # processed corpus path
MODEL_PATH = os.path.join(MODELS_PATH, MODEL_NAME)                  # path to the saved model
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
    logging.info("Computing and caching model embeddings for future use...")
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


class QueryRequest(BaseModel):
    """
    Request body for ranking and comprehension
    """
    query: str


@app.get("/")
def home(request: Request):
    """
    Displays the covid-19 comprehension homepage
    """
    num_docs = len(rank_corpus)
    return templates.TemplateResponse("home.html", {
        "request": request,
        "num_docs": '{:,}'.format(num_docs)
    })


@app.post("/query")
def post_query(request: Request, query_request: QueryRequest):
    query = query_request.query

    # ranking the documents using pre-calculated embeddings and query
    document_rank_results = rank_with_bert(query, model, corpus, embeddings)

    # ranking the paragraphs of the top retrieved documents
    paragraph_rank_results = paragraph_ranking(query, model, document_rank_results)

    # comprehending the top paragraphs of the retrieved documents for finding answer
    comprehend_results = comprehend_with_bert(comprehension_model, query, paragraph_rank_results)

    results = {
        'query': query,
        'results': comprehend_results
    }
    return {
        "code": "success",
        "response": results
    }
