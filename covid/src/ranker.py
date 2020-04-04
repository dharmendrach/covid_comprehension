import textwrap
import prettytable
import logging
import warnings
warnings.simplefilter('ignore')

import scipy


logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
logger = logging.getLogger(__name__)


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
