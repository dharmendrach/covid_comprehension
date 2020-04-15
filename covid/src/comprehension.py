import pandas as pd
import spacy

spacy_nlp = spacy.load('en_core_web_sm')

N_BEST_PER_PASSAGE = 1


def get_full_sentence(para_text, start_index, end_index):
    """
    Get surrounding sentence

    Parameters
    ----------
    para_text: str
        Paragraph text in form of string
    start_index: int
        start index in the original text
    end_index: int
        end index in the original text

    Returns
    -------
        surrounding sentence for original text. it's start index and end index.
    """
    sent_start = 0
    sent_end = len(para_text)
    for sent in spacy_nlp(para_text).sents:
        if (sent.start_char <= start_index) and (sent.end_char >= start_index):
            sent_start = sent.start_char
        if (sent.start_char <= end_index) and (sent.end_char >= end_index):
            sent_end = sent.end_char
    sentence = para_text[sent_start:sent_end + 1]
    return sentence, sent_start, sent_end


def result_on_one_document(model, question, document, top_k=5):
    """
    Comprehension on one document

    For the top retrieved paragraphs in the document, answer will be comprehended
    on each paragraph using the model.

    Parameters
    ----------
    model: Any HuggingsFace Model.
        Model used for comprehension
    question: str
        A question string
    document: dict
        A document containing paragraphs

    Returns
    -------
    document: dict
        Updates the given document dictionary with comprehension results
    """
    paragraphs = document["paragraphs"]
    paragraph_rank_results = document["paragraph_ranking"]
    answers = []
    for idx, para in enumerate(paragraph_rank_results):
        para_idx = para["paragraph_id"]
        query = {"context": paragraphs[para_idx], "question": question}
        pred = model(query, topk=N_BEST_PER_PASSAGE, max_answer_length=300)
        # assemble and format all answers
        # for pred in predictions:
        if pred["answer"]:
            sent, start, end = get_full_sentence(query['context'], pred["start"], pred["end"])
            answers.append({
                "answer": sent,     # pred["answer"],
                "context": paragraphs[para_idx],
                "offset_answer_start": start,   # pred["start"],
                "offset_answer_end": end,   # pred["end"],
                "probability": round(pred["score"], 4),
                "paragraph_id": para_idx
            })

    # sort answers by their `probability` and select top-k
    answers = sorted(
        answers, key=lambda k: k["probability"], reverse=True
    )
    answers = answers[:top_k]
    document["comprehension"] = answers
    return document


def comprehend_with_bert(model, question, documents):
    """
    Document Comprehension

    For the top retrieved paragraphs in each document, answer will be comprehended
    on each paragraph using the model.

    Parameters
    ----------
    model: Any HuggingsFace Model.
        Model used for comprehension
    question: str
        A question string
    documents: list
        A list of documents containing paragraphs

    Returns
    -------
    comprehend_results: list
        Updates each document in the list with comprehension results

    """
    comprehend_results = []
    for document in documents:
        result = result_on_one_document(model, question, document)
        comprehend_results.append(result)
    return comprehend_results


def show_comprehension_results(results):
    """
    Prints the comprehension results

    Parameters
    ----------
    results: list
        List of comprehended documents
    """
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
