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
COMPREHENSION_MODEL = "distilbert-base-uncased-distilled-squad"
COMPREHENSION_TOKENIZER = "distilbert-base-uncased"
use_gpu = -1
CORPUS_PATH = os.path.join(DATA_PATH, 'corpus.pkl')
MODEL_PATH = os.path.join(MODELS_PATH, MODEL_NAME)
EMBEDDINGS_PATH = os.path.join(
    DATA_PATH, f'{MODEL_NAME}-{RANK_USING}-embeddings.pkl')

logging.info(f"Ranking with {RANK_MODE}, using model: {MODEL_NAME}")


def cache_corpus(mode):

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

if not os.path.exists(EMBEDDINGS_PATH):
    logging.info("Computing and caching model embeddings for future use...")
    embeddings = model.encode(rank_corpus, show_progress_bar=True)
    with open(EMBEDDINGS_PATH, 'wb') as file:
        pickle.dump(embeddings, file)
else:
    logging.info("Loading model embeddings from", EMBEDDINGS_PATH, '...')
    with open(EMBEDDINGS_PATH, 'rb') as file:
        embeddings = pickle.load(file)

comprehension_model = pipeline("question-answering", model=COMPREHENSION_MODEL,
                               tokenizer=COMPREHENSION_TOKENIZER, device=use_gpu)


class QueryRequest(BaseModel):
    query: str


@app.get("/")
def home(request: Request):
    """
    displays the stock screener dashboard / homepage
    """
    return templates.TemplateResponse("home.html", {
        "request": request
    })


@app.post("/query")
def post_query(request: Request, query_request: QueryRequest):
    query = query_request.query

    document_rank_results = rank_with_bert(query, model, corpus, embeddings)
    paragraph_rank_results = paragraph_ranking(query, model, document_rank_results)
    comprehend_results = comprehend_with_bert(comprehension_model, query, paragraph_rank_results)

    # comprehend_results = []
    # each_answer = {
    #     "title": "title of the document",
    #     "document_score": 0.5,
    #     "document_rank": 1,
    #     "paragraphs": [
    #         "0 Lorem ipsum dolor sit amet, consectetur adipisicing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.",
    #         "1 Lorem ipsum dolor sit amet, consectetur adipisicing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.",
    #         "2 Lorem ipsum dolor sit amet, consectetur adipisicing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.",
    #         "3 Lorem ipsum dolor sit amet, consectetur adipisicing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.",
    #         "4 Lorem ipsum dolor sit amet, consectetur adipisicing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua."
    #     ],
    #     "paragraph_ranking": [
    #         {
    #             "paragraph_id": 3,
    #             "paragraph_score": 0.3
    #         },
    #         {
    #             "paragraph_id": 1,
    #             "paragraph_score": 0.1
    #         }
    #     ],
    #     "comprehension": [
    #         {
    #             "answer": "Lorem ipsum",
    #             "context": "3 Lorem ipsum dolor sit amet, consectetur adipisicing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.",
    #             "paragraph_id": 3,
    #             "probability": 0.23,
    #             "answer_start": 2,
    #             "answer_end": 13
    #         },
    #         {
    #             "answer": "Lorem",
    #             "context": "1 Lorem ipsum dolor sit amet, consectetur adipisicing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.",
    #             "paragraph_id": 1,
    #             "probability": 0.13,
    #             "answer_start": 2,
    #             "answer_end": 7
    #         }
    #     ],
    #     "cord_uid": 123,
    #     "publish_time": '20-20-2020',
    #     "authors": "1 Lorem ipsum dolor sit amet, consectetur adipisicing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.",
    #     "affiliations": "Infosys bangalore",
    #     "abstract": "this is the abstract of the document",
    #     "url": "http://google.com",
    #     "source": "github",
    #     "license": "open source"
    # }

    # comprehend_results.append(each_answer)
    # each_answer2 = each_answer.copy()
    # each_answer2["title"] = "title 2"
    # comprehend_results.append(each_answer2)
    results = {
        'query': query,
        'results': comprehend_results
    }
    return {
        "code": "success",
        "response": results
    }
