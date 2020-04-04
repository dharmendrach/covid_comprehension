import logging
import warnings
warnings.simplefilter('ignore')

import scipy
import pandas as pd
from tabulate import tabulate

def rank_with_bert(query, model, corpus, corpus_embed, top_k=5):
    queries = [query]
    query_embeds = model.encode(queries, show_progress_bar=False)
    for query, query_embed in zip(queries, query_embeds):
        distances = scipy.spatial.distance.cdist([query_embed], corpus_embed, "cosine")[0]
        distances = zip(range(len(distances)), distances)
        distances = sorted(distances, key=lambda x: x[1])
        results = []
        for count, (idx, distance) in enumerate(distances[0:top_k]):
            result = [count + 1, round(1 - distance, 4)]
            result.extend(corpus.iloc[idx].values)
            results.append(result)
    return results


def show_answers(results):
    
    cols = ['Rank', 'Score', 'paper_id', 'cord_uid', 'title', 'publish_time', 'authors',
                 'affiliations', 'abstract', 'text', 'url', 'source', 'license']

    df = pd.DataFrame(results, columns=cols)
    print(tabulate(df, headers='keys', tablefmt='fancy_grid', showindex=False))
