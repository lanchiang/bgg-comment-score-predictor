import os.path
from pathlib import Path

import click
import numpy as np
import pandas as pd


@click.command()
@click.option(
    '--file_path',
    '-f',
    type=str,
    help='Path to the file to be split'
)
@click.option(
    '--num_files',
    '-n',
    type=int,
    default=10,
    help='Number of file splits'
)
@click.option(
    '--output_dir',
    '-o',
    type=str,
    help='Output directory for the file splits'
)
def split(
    file_path: str,
    num_files: int,
    output_dir: str
):
    df = pd.read_csv(file_path)
    df_list = np.array_split(df, num_files)
    file_name = Path(file_path).stem

    for i, df_part in enumerate(df_list):
        df_part.to_csv(os.path.join(output_dir, f"{file_name}_part_{i}.csv"))

if __name__ == '__main__':
    split()