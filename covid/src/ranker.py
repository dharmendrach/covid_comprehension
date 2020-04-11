import logging
import warnings
import scipy
import pandas as pd
warnings.simplefilter('ignore')


def paragraph_ranking(query, model, documents, top_k=15):
    for each_doc in documents:
        paras = each_doc["paragraphs"]
        para_embeds = model.encode(paras, show_progress_bar=False)
        query_embed = model.encode([query], show_progress_bar=False)
        distances = scipy.spatial.distance.cdist(query_embed, para_embeds, "cosine")[0]
        paragraph_ranking = []
        for para_idx, para_score in enumerate(distances):
            result = {}
            result['paragraph_id'] = paras[para_idx]
            result['paragraph_score'] = round(1 - para_score, 4)
            paragraph_ranking.append(result)
        sorted_paras = sorted(paragraph_ranking, key=lambda x: x['paragraph_score'], reverse=True)
        paragraph_ranking_results = sorted_paras[0:top_k]
        each_doc["paragraph_ranking"] = paragraph_ranking_results
    return documents


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
            paras = doc['text'].split('\n\n')
            paras = [para for para in paras if len(para) > 0]
            result = {}
            result['document_rank'] = count + 1
            result['document_score'] = round(1 - distance, 4)
            result['paper_id'] = doc['paper_id']
            result['cord_uid'] = doc['cord_uid']
            result['title'] = doc['title']
            result['publish_time'] = doc['publish_time']
            result['authors'] = doc['authors']
            result['affiliations'] = doc['affiliations']
            result['abstract'] = doc['abstract']
            result['paragraphs'] = paras
            result['url'] = doc['url']
            result['source'] = doc['source']
            result['license'] = doc['license']
            results.append(result)
    return results


def show_ranking_results(results):
    result = {"title": [], "doc_rank": [], "doc_score": [], "abstract": []}
    for r in results:
        result['title'].append(r['title'])
        result['doc_rank'].append(r['document_rank'])
        result['doc_score'].append(r['document_score'])
        result['abstract'].append(r['abstract'])
    df = pd.DataFrame(result)
    print(df)
