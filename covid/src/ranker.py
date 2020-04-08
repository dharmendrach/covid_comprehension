import logging
import warnings
warnings.simplefilter('ignore')

import scipy
import pandas as pd

def rank_with_bert(query, model, corpus, corpus_embed, top_k=5):
    queries = [query]
    query_embeds = model.encode(queries, show_progress_bar=False)
    for query, query_embed in zip(queries, query_embeds):
        distances = scipy.spatial.distance.cdist([query_embed], corpus_embed, "cosine")[0]
        distances = zip(range(len(distances)), distances)
        distances = sorted(distances, key=lambda x: x[1])
        results = []
        for count, (idx, distance) in enumerate(distances[0:top_k]):
            doc = corpus.iloc[idx]
            result = {}
            result['rank'] = count + 1
            result['score'] = round(1 - distance, 4)
            result['paper_id'] = doc['paper_id']
            result['cord_uid'] = doc['cord_uid']
            result['title'] = doc['title']
            result['publish_time'] = doc['publish_time']
            result['authors'] = doc['authors']
            result['affiliations'] = doc['affiliations']
            result['abstract'] = doc['abstract']
            result['text'] = doc['text']
            result['url'] = doc['url']
            result['source'] = doc['source']
            result['license'] = doc['license']
            results.append(result)
    return results


def show_ranking_results(results):
    result = {"title": [], "doc_rank": [], "doc_score": [], "abstract": []}
    for r in results:
        result['title'].append(r['title'])
        result['doc_rank'].append(r['rank'])
        result['doc_score'].append(r['score'])
        result['abstract'].append(r['abstract'])
    df = pd.DataFrame(result)
    print(df)
