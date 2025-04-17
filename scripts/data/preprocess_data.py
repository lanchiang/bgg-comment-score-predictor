import os.path
import re

import click
import pandas as pd
import yaml
from datasets import Dataset

from bgg_playground.database.db_config import get_db_connection


def fetch_from_postgres() -> pd.DataFrame:
    query = "select * from comments where (comment = '') IS NOT TRUE and comment_lang = 'ENGLISH'"

    with get_db_connection() as conn:
        return pd.read_sql(sql=query, con=conn)


def preprocess(df: pd.DataFrame, min_comment_len = 30) -> pd.DataFrame:
    # remove all non-white space characters, numbers, and urls
    df['comment'] = df['comment'].replace(to_replace=r'[^\w\s]', value='', regex=True)
    df['comment'] = df['comment'].replace(to_replace=r'\d', value='', regex=True)
    df['comment'] = df['comment'].apply(func=lambda text: re.compile(r'https?://\S+').sub('', text))

    # lower case all characters
    df['comment'] = df['comment'].str.casefold()
    # keep only data whose comments are longer than the minimal length
    df = df[df['comment'].str.len() >= min_comment_len]

    return df


def train_test_split(df: pd.DataFrame, test_size=0.2, random_state=42):
    test_df = df.sample(frac=test_size, random_state=random_state)
    train_df = df.drop(test_df.index)
    return Dataset.from_pandas(train_df), Dataset.from_pandas(test_df)


@click.command()
@click.option(
    "--config_path",
    "-c",
    type=str,
    default="configs/data_config.yaml",
    help="Data config file path"
)
def main(config_path: str):
    # refactoring idea: first read env variables from the CI/CD workflow, then fallback to config file
    with open(config_path, 'r') as f:
        yaml_dict = yaml.safe_load(f)

    dataset = fetch_from_postgres()

    dp_variables = yaml_dict['data_preprocessing']

    df = preprocess(dataset)
    train_dataset, eval_dataset = train_test_split(df, dp_variables['test_size'], dp_variables['random_state'])

    output_dir = dp_variables['output_dir']
    train_dataset.save_to_disk(os.path.join(output_dir, 'train.dataset'))
    eval_dataset.save_to_disk(os.path.join(output_dir, 'eval.dataset'))


if __name__ == '__main__':
    main()
