import csv
import glob
import os.path
from io import StringIO
from pathlib import Path

import click
import pandas as pd
import yaml
from lingua import Language, LanguageDetectorBuilder
from psycopg2.extras import execute_batch

from bgg_playground.database.db_config import get_db_connection
from bgg_playground.database.db_queries import get_ratings_with_comments
from bgg_playground.utils import logs

log = logs.get_logger()

languages = [
        Language.ENGLISH,
        Language.FRENCH,
        Language.SPANISH,
        Language.GERMAN,
        Language.CHINESE,
        Language.DUTCH,
        Language.ITALIAN,
        Language.JAPANESE,
        Language.KOREAN,
        Language.RUSSIAN
    ]
detector = LanguageDetectorBuilder.from_languages(*languages).build()


def load_csv_to_db(path: str, csv_columns: list, table_columns: list, table_name: str, debug_mode: bool):
    """
    Loads the data from a csv file to a table in the database.
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute(f'TRUNCATE TABLE {table_name} RESTART IDENTITY CASCADE')
    conn.commit()
    log.info(f'Table {table_name} cleared successfully.')

    if Path(path).is_dir():
        csv_files = glob.glob(os.path.join(path, '*.csv'))

        dfs = []

        for file in csv_files:
            df = pd.read_csv(file)
            df = df[csv_columns]
            dfs.append(df)

        df = pd.concat(dfs, ignore_index=True)
    else:
        df = pd.read_csv(path)
        df = df[csv_columns]

    log.info(f"Debug mode set to: {debug_mode}")
    if debug_mode:
        seed = 42
        subset_size = 3000
        df = df.sample(n=subset_size, random_state=seed)

    log.info(f"Number of loaded comments: {len(df)}")

    output = StringIO()
    df.to_csv(output, sep='\t', quoting=csv.QUOTE_ALL, header=False, index=False)
    output.seek(0)

    # Quote all column names in the list, so that they do not conflict with reserved keywords
    quoted_columns = [f"\"{col}\"" for col in table_columns]

    sql = f"COPY {table_name} ({', '.join(quoted_columns)}) FROM STDIN WITH (FORMAT csv, DELIMITER E'\t', QUOTE E'\"')"

    cursor.copy_expert(sql, output)
    conn.commit()
    cursor.close()
    conn.close()

    log.info(f'{path} loaded to {table_name} successfully')


@click.command()
@click.option(
    '--config_file',
    '-c',
    type=str,
    default='configs/data_config.yml',
    help='Database config file path'
)
@click.option(
    '--debug_mode',
    '-d',
    type=bool,
    default=False,
    help='Load a subset of data into the database. For debug purpose only.'
)
def populate_db_with_config(config_file: str, debug_mode: bool):
    """
    Populates the database with the data from the csv files specified in the config file.
    """
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    for csv_config in config['csv_files']:
        file_path = csv_config['file_path']
        csv_columns = csv_config['csv_columns']
        table_columns = csv_config['table_columns']
        table_name = csv_config['table_name']

        load_csv_to_db(file_path, csv_columns, table_columns, table_name, debug_mode)


def parallel_language_detection(ratings_with_comments):
    """
    Detects the language of the comments in parallel.
    """
    text_list = [rating[3] for rating in ratings_with_comments]
    results = detector.detect_languages_in_parallel_of(text_list)
    results = [(rating_with_comment[0], lang) for rating_with_comment, lang in zip(ratings_with_comments, results)]
    return results


def populate_comment_lang():
    """
    Populates the comment_lang column in the comments table with the detected language of the comment text.
    """
    ratings_with_comments = get_ratings_with_comments()
    
    results = parallel_language_detection(ratings_with_comments)

    with get_db_connection() as conn:
        with conn.cursor() as cursor:
            execute_batch(
                cursor,
                "UPDATE comments SET comment_lang = %s WHERE id = %s",
                [(lang.name, id) if lang is not None else ('', id) for id, lang in results]
            )

        conn.commit()


if __name__ == '__main__':
    populate_db_with_config()
