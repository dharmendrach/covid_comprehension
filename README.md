# Covid-19 Comprehension

**Covid-19 Comprehension** uses state-of-the-art language model to search relevant content and comprehend the documents present inside the [COVID-19 Open Research Dataset (CORD-19)](https://pages.semanticscholar.org/coronavirus-research) recently published by the White House and its research partners. The dataset contains over 51,000 scholarly articles about COVID-19, SARS-CoV-2 and related coronaviruses.


![Flow Img](https://github.com/graviraja/covid-comprehension/tree/master/static/flow.png)

Steps performed:

- Rank the documents with paper **abstract** (or) **title** (or) **content** using any of the above mentioned models.
- Rank the paragraphs of the retrieved documents using any of the above mentioned models.
- Comprehened the top paragraphs of the retrieved documents using **[`HuggingFace`](https://huggingface.co/)** question-answering pipeline.

Various models already fine-tuned on Natural Language Inference are available to perform the search:

We are using the models provided by **[`gsarti`](https://github.com/gsarti/covid-papers-browser)**  [7] for ranking purposes

- **[`scibert-nli`](https://huggingface.co/gsarti/scibert-nli)**, a fine-tuned version of AllenAI's [SciBERT](https://github.com/allenai/scibert) [1].

- **[`biobert-nli`](https://huggingface.co/gsarti/biobert-nli)**, a fine-tuned version of [BioBERT](https://github.com/dmis-lab/biobert) by J. Lee et al. [2]

- **[`covidbert-nli`](https://huggingface.co/gsarti/covidbert-nli)**, a fine-tuned version of Deepset's [CovidBERT](https://huggingface.co/deepset/covid_bert_base).

Both models are trained on [SNLI](https://nlp.stanford.edu/projects/snli/) [3] and [MultiNLI](https://www.nyu.edu/projects/bowman/multinli/) [4] using the [`sentence-transformers` library](https://github.com/UKPLab/sentence-transformers/) [5] to produce universal sentence embeddings [6]. Embeddings are subsequently used to perform semantic search on CORD-19.

For comprehension we finetuned the [CovidBERT](https://huggingface.co/deepset/covid_bert_base) model on [SQuAD](https://rajpurkar.github.io/SQuAD-explorer/) question answering data.

- **[`covidbert_squad`](https://huggingface.co/graviraja/covidbert_squad)**, a fine-tuned version of Deepset's [CovidBERT](https://huggingface.co/deepset/covid_bert_base) on SQuAD dataset.


![Model Img](https://github.com/graviraja/covid-comprehension/tree/master/static/model.png)



## Setup

Python 3.6 or higher is required to run the code. First, install the required libraries with `pip`, then download the `en_core_web_sm` language pack for spaCy:

```shell
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## Running the Code

First of all, download a model fine-tuned on NLI from HuggingFace's cloud repository.

```shell
python scripts/download_model.py --model scibert-nli
```

Second, download the data from the [Kaggle challenge page](https://www.kaggle.com/allen-institute-for-ai/CORD-19-research-challenge) and place it in the `data` folder and extract it.


Using a GPU is suggested since the creation of the embeddings for the entire corpus and comprehending the top retrieved documents might be time-consuming otherwise. Both the corpus and the embeddings are cached on disk after the first execution of the script, and execution is really fast after embeddings are computed.

Finally, simply run:

```shell
python manage.py
```

To enter the interactive demo, simply run:

```shell
uvicorn app:app --host 0.0.0.0
```

This run a server at http://localhost:8000/ 


## References

[1] Beltagy et al. 2019, ["SciBERT: Pretrained Language Model for Scientific Text"](https://www.aclweb.org/anthology/D19-1371/)

[2] Lee et al. 2020, ["BioBERT: a pre-trained biomedical language representation model for biomedical text mining"](http://doi.org/10.1093/bioinformatics/btz682)

[3] Bowman et al. 2015, ["A large annotated corpus for learning natural language inference"](https://www.aclweb.org/anthology/D15-1075/)

[4] Adina et al. 2018, ["A Broad-Coverage Challenge Corpus for Sentence Understanding through Inference"](http://aclweb.org/anthology/N18-1101)

[5] Reimers et al. 2019, ["Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks"](https://www.aclweb.org/anthology/D19-1410/)

[6] As shown in Conneau et al. 2017, ["Supervised Learning of Universal Sentence Representations from Natural Language Inference Data"](https://www.aclweb.org/anthology/D17-1070/)

[7] [gsarti covid-papers-browser](https://github.com/gsarti/covid-papers-browser)