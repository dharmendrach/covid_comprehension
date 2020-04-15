import os
import json
import logging

import pandas as pd
from tqdm import tqdm


def get_all_files(dir_name):
    """
    Get all the files in the given directory and all it's sub-directories

    Parameters
    ----------
    dir_name: str
        A directory path

    Returns
    -------
    all_files: list
        A list containing all the files in the given directory
    """
    list_of_file = os.listdir(dir_name)
    all_files = list()

    for entry in list_of_file:
        full_path = os.path.join(dir_name, entry)
        # If entry is a directory then get the list of files in this directory
        if os.path.isdir(full_path):
            all_files = all_files + get_all_files(full_path)
        else:
            if full_path.endswith('json'):
                all_files.append(full_path)

    return all_files


def load_files(path):
    """
    Load all the files in the given path

    Parameters
    ----------
    path: str
        path to a directory containing files

    Returns
    -------
    all_files: list
        A list containing the loaded files in json
    """
    filenames = get_all_files(path)
    logging.info(f"Number of files retrived from {path}: {len(filenames)}")
    all_files = []

    for each_file in tqdm(filenames, desc="Reading files: "):
        with open(each_file, 'rb') as f:
            file = json.load(f)
        all_files.append(file)
    return all_files


def format_name(author):
    """
    Formats the author name into a standard format
    """
    middle_name = " ".join(author['middle'])

    if author['middle']:
        return " ".join([author['first'], middle_name, author['last']])
    else:
        return " ".join([author['first'], author['last']])


def format_affiliation(affiliation):
    """
    Formats the affiliation details by location and institution
    """
    text = []
    location = affiliation.get('location')
    if location:
        text.extend(list(affiliation['location'].values()))

    institution = affiliation.get('institution')
    if institution:
        text = [institution] + text
    return ", ".join(text)


def format_authors(authors, with_affiliation=False):
    """
    Formats each author name with optional affiliation details
    """
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
    """
    Formats the sections of the paper into a single body
    """
    texts = [(di['section'], di['text']) for di in body_text]
    texts_di = {di['section']: "" for di in body_text}

    for section, text in texts:
        texts_di[section] += text

    body = ""

    for section, text in texts_di.items():
        body += text
        body += "\n\n"

    return body


def generate_clean_df(all_files, metadata, consider_without_mode=False, mode="abstract"):
    """
    Formats each file and links with it's metada.

    Parameters
    ----------
    all_files: list
        A list of json files
    metadata: pandas.DataFrame
        A pandas dataframe containing the metadata
    consider_without_mode: boolean
        To consider or not to consider the empty values of mode
    mode: str
        A string indicating the mode used for ranking

    Returns
    -------
    clean_df: pandas.DataFrame
        A pandas dataframe containing the cleaned data
    """
    cleaned_files = []
    metadata_not_found = 0
    mode_not_found = 0

    if mode == "abstract":
        mode_key = "abstract"
    elif mode == "text":
        mode_key == "body_text"
    elif mode_key == "title":
        mode_key = "title"
    else:
        raise AttributeError("Ranking should be with abstract, text (or) title are only supported")

    for file in tqdm(all_files):
        row = metadata.loc[metadata['sha'] == file['paper_id']]
        if row.empty:
            metadata_not_found += 1
            continue

        mode_data = ""
        if mode == "abstract" or mode == "text":
            mode_data = format_body(file['abstract'])
        elif mode == "title":
            mode_data = file['metadata']['title']
        if mode_data == "":
            mode_not_found += 1
            if consider_without_mode:
                continue

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
            row['source_x'],
            row['license']
        ]

        cleaned_files.append(features)

    logging.info(f"Metadata not found for {metadata_not_found} files")
    logging.info(f"{mode} is null for {mode_not_found} files")

    col_names = ['paper_id', 'cord_uid', 'title', 'publish_time', 'authors',
                 'affiliations', 'abstract', 'text', 'url', 'source', 'license']
    clean_df = pd.DataFrame(cleaned_files, columns=col_names)
    clean_df.head()

    return clean_df


def generate_clean_csv(datapath, metadata_path, source, datafolder, mode):
    """
    Formats each of the file in datapath and link it's metadata in metadata_path.

    Parameters
    ----------
    datapath: str
        A string containing the path to raw files
    metadata_path: str
        A string containing the path to metadata.csv file
    source: str
        A string indicating the source which is being processed
    datafolder: str
        A string containing the path to save the cleaned data
    mode: str
        A string indicating the mode of ranking

    Returns
    -------
    clean_df: pandas.DataFrame
        A pandas dataframe containing the cleaned data
    """
    files = load_files(datapath)
    metadata = pd.read_csv(metadata_path)
    clean_df = generate_clean_df(files, metadata, mode)
    logging.info(f"Saving the cleaned data into file: {datafolder}/clean_{source}.csv")
    clean_df.to_csv(f'{datafolder}/clean_{source}.csv', index=False)
    return clean_df
