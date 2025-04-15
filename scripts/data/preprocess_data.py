import re

import click
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, PreTrainedTokenizerFast

from bgg_playground.database.db_config import get_db_connection


def fetch_from_postgres() -> Dataset:
    query = "select * from comments where (comment = '') IS NOT TRUE and comment_lang = 'ENGLISH'"

    with get_db_connection() as conn:
        return Dataset.from_pandas(pd.read_sql(sql=query, con=conn))


def preprocess(df: Dataset,
               min_comment_len = 15,
               tokenizer_model_name = 'google-bert/bert-base-cased') -> pd.DataFrame:
    df['comment'] = df['comment'].replace(to_replace=r'[^\w\s]', value='', regex=True)
    df['comment'] = df['comment'].replace(to_replace=r'\d', value='', regex=True)
    df['comment'] = df['comment'].apply(func=lambda text: re.compile(r'https?://\S+').sub('', text))

    df['comment'] = df['comment'].str.casefold()
    df = df[df['comment'].str.len() >= min_comment_len]

    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=tokenizer_model_name, use_fast=True)
    df['comment_tokens'] = df['comment'].apply(func=lambda comment : tokenizer(comment, padding='max_length', truncation=True))
    return df


def preprocess_comment(batch, tokenizer: PreTrainedTokenizerFast, min_length = 15):
    processed_texts = []
    for text in batch['comment']:
        text = re.sub(r'https?://\S+', '', text)
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\d', '', text)
        text = text.casefold()
        processed_texts.append(text)

    tokenized = tokenizer(
        processed_texts,
        padding=True,
        truncation=True,
        return_tensors='pt'
    )




@click.command()
@click.option(
    "--config_path",
    "-c",
    type=str,
    default="configs/data_config.yaml",
    help="Data config file path"
)
def main(config_path: str):
    dataset = fetch_from_postgres()
    df = preprocess(dataset)
    pass

if __name__ == '__main__':
    main()
