import pandas as pd

N_BEST_PER_PASSAGE = 1
CONTEXT_SIZE = 50


def result_on_one_document(model, question, document, top_k=3):
    paragraphs = document["text"].split('\n\n')
    paragraphs = [para for para in paragraphs if len(para) > 0]

    answers = []
    for p in paragraphs:
        query = {"context": p, "question": question}
        predictions = model(query, topk=N_BEST_PER_PASSAGE)
        # assemble and format all answers
        for pred in predictions:
            if pred["answer"]:
                # context_start = max(0, pred["start"] - CONTEXT_SIZE)
                # context_end = min(len(p), pred["end"] + CONTEXT_SIZE)
                answers.append({
                    "answer": pred["answer"],
                    "context": p,
                    "offset_answer_start": pred["start"],
                    "offset_answer_end": pred["end"],
                    "probability": pred["score"],
                    "paper_id": document['paper_id']
                })

    # sort answers by their `probability` and select top-k
    answers = sorted(
        answers, key=lambda k: k["probability"], reverse=True
    )
    answers = answers[:top_k]

    return answers


def comprehend_with_bert(model, question, documents, top_k=5):
    answers = []
    for document in documents:
        doc_answers = result_on_one_document(model, question, document)
        for each_answer in doc_answers:
            each_answer['cord_uid'] = document['cord_uid']
            each_answer['title'] = document['title']
            each_answer['publish_time'] = document['publish_time']
            each_answer['authors'] = document['authors']
            each_answer['affiliations'] = document['affiliations']
            each_answer['abstract'] = document['abstract']
            each_answer['text'] = document['text']
            each_answer['url'] = document['url']
            each_answer['source'] = document['source']
            each_answer['license'] = document['license']
            each_answer['document_rank'] = document['rank']
            each_answer['document_rank_score'] = document['score']
            answers.append(each_answer)
    answers = sorted(
        answers, key=lambda k: k["probability"], reverse=True
    )
    answers = answers[:top_k]
    results = {"question": question,
            "answers": answers}
    return results


def show_comprehension_results(results):
    result = {"answer": [], "context": [], "title": [], "score": [], "doc_rank": [], "doc_score": []}
    for r in results:
        result['answer'].append(r['answer'])
        result['context'].append(r['context'])
        result['title'].append(r['title'])
        result['score'].append(r['probability'])
        result['doc_rank'].append(r['document_rank'])
        result['doc_score'].append(r['document_rank_score'])
    df = pd.DataFrame(result)
    print(df)
