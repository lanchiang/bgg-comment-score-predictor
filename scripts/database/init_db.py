import click

from bgg_playground.database.db_config import get_db_connection


@click.command()
@click.option(
    '--schema_path',
    '-s',
    type=str,
    default='schema.sql',
    help='Path to the schema file'
)
def create_table(schema_path: str):
    conn = get_db_connection()
    cursor = conn.cursor()

    with open(schema_path, 'r') as f:
        sql_commands = f.read()
        cursor.execute(sql_commands)

    conn.commit()
    cursor.close()
    conn.close()


if __name__ == '__main__':
    create_table()
