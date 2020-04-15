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


def generate_clean_df(all_files, metadata, consider_empty=False, mode="abstract"):
    """
    Formats each file and links with it's metada.

    Parameters
    ----------
    all_files: list
        A list of json files
    metadata: pandas.DataFrame
        A pandas dataframe containing the metadata
    consider_empty: boolean
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

    for file in tqdm(all_files):
        mode_data = ""
        if mode == "abstract":
            try:
                if 'abstract' in file.keys():
                    mode_data = format_body(file['abstract'])
                elif 'abstract' in file['metadata'].keys():
                    mode_data = format_body(file['metadat']['abstract'])
                else:
                    mode_not_found += 1
                    continue
            except Exception as e:
                mode_not_found += 1
                continue
        elif mode == "text":
            mode_data = format_body(file['body_text'])
        elif mode == "title":
            mode_data = file['metadata']['title']
        if mode_data == "":
            mode_not_found += 1
            continue

        id_row = metadata.loc[metadata['sha'] == file['paper_id']]
        row = id_row
        metadata_found = True
        if id_row.empty:
            title_row = metadata.loc[metadata['title'] == file['metadata']['title']]
            if title_row.empty:
                metadata_not_found += 1
                metadata_found = False
            else:
                row = title_row
        if metadata_found:
            row = row.iloc[0]
            cord_uid = row['cord_uid']
            publish_time = row['publish_time']
            url = row['url']
            source = row['source_x']
            license = row['license']
        else:
            cord_uid = ''
            publish_time = ''
            url = ''
            source = ''
            license = ''

        try:
            paper_id = file['paper_id']
        except Exception as e:
            paper_id = ''

        try:
            title = file['metadata']['title']
        except Exception as e:
            title = 'Title not found'

        try:
            authors = format_authors(file['metadata']['authors'])
        except Exception as e:
            authors = 'Authors not found'

        try:
            affiliation = format_authors(
                file['metadata']['authors'], with_affiliation=True)
        except Exception as e:
            affiliation = ''

        try:
            if 'abstract' in file.keys():
                abstract = format_body(file['abstract'])
            elif 'abstract' in file['metadat'].keys():
                abstract = format_body(file['metadat']['abstract'])
            else:
                abstract = ''
        except Exception as e:
            abstract = ''

        try:
            body = format_body(file['body_text'])
        except Exception as e:
            body = ''

        features = [
            paper_id,
            cord_uid,
            title,
            publish_time,
            authors,
            affiliation,
            abstract,
            body,
            url,
            source,
            license
        ]

        cleaned_files.append(features)

    logging.info(f"Metadata not found for {metadata_not_found} files")
    logging.info(f"{mode} is null for {mode_not_found} files")
    logging.info(f"considered {len(cleaned_files)} files")
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
    clean_df = generate_clean_df(files, metadata,False, mode)
    logging.info(
        f"Saving the cleaned data into file: {datafolder}/clean_{source}.csv")
    clean_df.to_csv(f'{datafolder}/clean_{source}.csv', index=False)
    return clean_df
