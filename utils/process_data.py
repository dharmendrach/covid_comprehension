import os
import json
import logging
from pprint import pprint
from copy import deepcopy

import numpy as np
import pandas as pd
from tqdm import tqdm

logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
logger = logging.getLogger(__name__)


def load_files(path):
    filenames = os.listdir(path)
    logging.info(f"Number of files retrived from {path}: {len(filenames)}")
    all_files = []

    for each_file in tqdm(filenames, desc="Reading files: "):
        filename = path + each_file
        with open(filename, 'rb') as f:
            file = json.load(f)
        all_files.append(file)
    return all_files


def format_name(author):
    middle_name = " ".join(author['middle'])
    
    if author['middle']:
        return " ".join([author['first'], middle_name, author['last']])
    else:
        return " ".join([author['first'], author['last']])


def format_affiliation(affiliation):
    text = []
    location = affiliation.get('location')
    if location:
        text.extend(list(affiliation['location'].values()))
    
    institution = affiliation.get('institution')
    if institution:
        text = [institution] + text
    return ", ".join(text)


def format_authors(authors, with_affiliation=False):
    name_ls = []
    
    for author in authors:
        name = format_name(author)
        if with_affiliation:
            affiliation = format_affiliation(author['affiliation'])
            if affiliation:
                name_ls.append(f"{name} ({affiliation})")
            else:
                name_ls.append(name)
        else:
            name_ls.append(name)
    
    return ", ".join(name_ls)


def format_body(body_text):
    texts = [(di['section'], di['text']) for di in body_text]
    texts_di = {di['section']: "" for di in body_text}
    
    for section, text in texts:
        texts_di[section] += text

    body = ""

    for section, text in texts_di.items():
        body += text
        body += "\n\n"
    
    return body


def format_bib(bibs):
    if type(bibs) == dict:
        bibs = list(bibs.values())
    bibs = deepcopy(bibs)
    formatted = []
    
    for bib in bibs:
        bib['authors'] = format_authors(
            bib['authors'], 
            with_affiliation=False
        )
        formatted_ls = [str(bib[k]) for k in ['title', 'authors', 'venue', 'year']]
        formatted.append(", ".join(formatted_ls))

    return "; ".join(formatted)


def generate_clean_df(all_files, metadata):
    cleaned_files = []
    metadata_not_found = 0

    for file in tqdm(all_files):
        row = metadata.loc[metadata['sha'] == file['paper_id']]
        if row.empty:
            metadata_not_found += 1
            continue
        assert len(row) == 1, "Unique file not found"
        row = row.iloc[0]

        features = [
            file['paper_id'],
            row['cord_uid'],
            file['metadata']['title'],
            row['publish_time'],
            format_authors(file['metadata']['authors']),
            format_authors(file['metadata']['authors'], 
                           with_affiliation=True),
            format_body(file['abstract']),
            format_body(file['body_text']),
            row['url'],
            row['source'],
            row['license'],
            format_bib(file['bib_entries'])
        ]

        cleaned_files.append(features)

    col_names = ['paper_id', 'cord_uid', 'title', 'publish_time', 'authors',
                 'affiliations', 'abstract', 'text', 'url', 'source', 'license',
                 'bibliography']
    clean_df = pd.DataFrame(cleaned_files, columns=col_names)
    clean_df.head()
    
    return clean_df


def generate_clean_csv(datapath, metadata_path, source, datafolder):
    files = load_files(datapath)
    metadata = pd.read_csv(metadata_path)
    clean_df = generate_clean_df(files, metadata)
    logging.info(f"Saving the cleaned data into file: {datafolder}/clean_{source}.csv")
    clean_df.to_csv(f'{datafolder}/clean_{source}.csv', index=False)
    return clean_df

if __name__ == '__main__':
    pass
