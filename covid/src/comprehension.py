import pandas as pd

N_BEST_PER_PASSAGE = 1


def result_on_one_document(model, question, document, top_k=5):
    paragraphs = document["paragraphs"]
    paragraph_rank_results = document["paragraph_ranking"]
    answers = []
    for idx, para in enumerate(paragraph_rank_results):
        query = {"context": paragraphs[para.paragraph_id], "question": question}
        pred = model(query, topk=N_BEST_PER_PASSAGE)
        # assemble and format all answers
        # for pred in predictions:
        if pred["answer"]:
            answers.append({
                "answer": pred["answer"],
                "context": para,
                "offset_answer_start": pred["start"],
                "offset_answer_end": pred["end"],
                "probability": round(pred["score"], 4),
                "paragraph_id": para.paragraph_id
            })

    # sort answers by their `probability` and select top-k
    answers = sorted(
        answers, key=lambda k: k["probability"], reverse=True
    )
    answers = answers[:top_k]
    document["comprehension"] = answers
    return document


def comprehend_with_bert(model, question, documents):
    comprehend_results = []
    for document in documents:
        result = result_on_one_document(model, question, document)
        comprehend_results.append(result)
    return comprehend_results


def show_comprehension_results(results):
    result = {"answer": [], "context": [], "title": [], "score": [], "doc_rank": [], "doc_score": []}
    for r in results:
        result['answer'].append(r['answer'])
        result['context'].append(r['context'])
        result['title'].append(r['title'])
        result['score'].append(r['probability'])
        result['doc_rank'].append(r['document_rank'])
        result['doc_score'].append(r['document_score'])
    df = pd.DataFrame(result)
    print(df)
