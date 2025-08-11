import os
import requests
from pathlib import Path
from tqdm import tqdm
import shutil
import pandas as pd
from sentence_transformers import SentenceTransformer
from typing import List
from dotenv import load_dotenv
import argparse

load_dotenv()  # Load environment variables from .env file

EMBEDDING_MODEL_NAME = os.getenv('EMBEDDING_MODEL_NAME', 'all-MiniLM-L6-v2')

def download_phishing_dataset(
    output_dir="data", 
    filename="raw-phishing-email-dataset.zip",
    chunk_size=8192,
    overwrite=False
):
    """
    Download the phishing email dataset with progress bar.
    
    Args:
        output_dir (str): Directory to save the file
        filename (str): Name for the downloaded file
        chunk_size (int): Size of chunks to download at a time
    
    Returns:
        dict: Download information including path, size, etc.
    """
    # Setup paths
    output_path = Path(output_dir) if os.path.isabs(output_dir) else Path(os.getcwd()) / output_dir
    print(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    file_path = output_path / filename
    
    # Download URL
    url = "https://www.kaggle.com/api/v1/datasets/download/naserabdullahalam/phishing-email-dataset"
    
    # Start download with streaming
    response = requests.get(url, stream=True, allow_redirects=True)
    response.raise_for_status()
    
    # Get file size for progress bar
    total_size = int(response.headers.get('content-length', 0))

    #check if file already exists and size matches
    if file_path.exists() and file_path.stat().st_size == total_size:
        print(f'File {file_path} already exists')
        if not overwrite:
            print('Skipping download... \n to overwrite, use --overwrite flag')
            return {
        'path': str(file_path),
        'size_bytes': file_path.stat().st_size,
        'size_mb': file_path.stat().st_size / (1024 * 1024),
        'exists': file_path.exists()
    }
        if overwrite:
            print('Overwriting existing file...')
    
    # Download with progress bar
    with open(file_path, 'wb') as f, tqdm(
        desc=filename,
        total=total_size,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
                pbar.update(len(chunk))
    # Unzip the file
    print('Unzipping...')
    shutil.unpack_archive(filename=file_path, extract_dir=output_path)
    
    
    return {
        'path': str(file_path),
        'size_bytes': file_path.stat().st_size,
        'size_mb': file_path.stat().st_size / (1024 * 1024),
        'exists': file_path.exists()
    }
def csv_preprocessing(df, fname: str, 
                      columns_to_drop = ['receiver', 'urls', 'date']):
    existing_columns_to_drop = [col for col in columns_to_drop if col in df.columns]
    df = df.drop(columns=existing_columns_to_drop)
    df['source'] = fname
    return df

def embed_df_text(df, text_column='body', embedding_model=EMBEDDING_MODEL_NAME):
    """
    Embed df text column using specific sentence-transformer embedding model
    """
    # Example: df['embedding'] = df[text_column].apply(lambda x: some_embedding_function(x))
    model = SentenceTransformer(embedding_model)
    tqdm.pandas(desc=f"Embedding {text_column}")
    df['embedding'] = df[text_column].progress_apply(lambda x: model.encode(str(x)))
    # print('Embedding dtype:', type(df['embedding'][0]))
    return df

def combine_dfs_and_shuffle(dfs: List[pd.DataFrame]):
    combined_df = pd.concat(dfs, ignore_index=True)
    combined_df = combined_df.sample(frac=1).reset_index(drop=True)  # Shuffle the DataFrame
    return combined_df

def process_downloaded_files(input_dir,
                             interim_dir='data/interim',
                             processed_dir = 'data/processed',
                             files_to_use=None,
                             output_filename='combined_spam_ham_dataset.pkl',
                             overwrite=False):
    
    combined_output_path = f'{processed_dir}/{output_filename}'
    if os.path.exists(combined_output_path) and not overwrite:
        print(f'File {combined_output_path} already exists, skipping processing...')
        return
    dfs = []
    for file in tqdm(os.listdir(input_dir)):
        if files_to_use and file not in files_to_use:
            # skip files if not in specified list
            print(f'file {file} not found in approved list of files to use, skipping...')
            continue
        if file.endswith('.csv'):
            print(f'Processing file: {file}')
            df = pd.read_csv(f'{input_dir}/{file}')
            df = csv_preprocessing(df, file)
            df = embed_df_text(df)
            output_path = f'{interim_dir}/processed_{file}.pkl'
            print(f'Saving {file} to: {output_path}')
            df.to_pickle(output_path)
            dfs.append(df)
    combined = combine_dfs_and_shuffle(dfs)
    combined.to_pickle(combined_output_path)


def parse_args():
    parser = argparse.ArgumentParser(description="Download and preprocess phishing email dataset")
    parser.add_argument(
        '--files-to-use', 
        nargs='+', 
        default=['CEAS_08.csv', 'SpamAssasin.csv'],
        help='List of CSV files to process (default: CEAS_08.csv SpamAssasin.csv)'
    )
    parser.add_argument(
        '--overwrite', 
        action='store_true',
        default=False,
        help='Overwrite existing files (default: False)'
    )
    return parser.parse_args()

def main():
    #loads in CLI args, with defaults values already defined if not provided
    args = parse_args()
    result = download_phishing_dataset("data/raw", overwrite=args.overwrite)
    process_downloaded_files('data/raw',
                              files_to_use=args.files_to_use, 
                              overwrite=args.overwrite)

if __name__ == "__main__":
    main()