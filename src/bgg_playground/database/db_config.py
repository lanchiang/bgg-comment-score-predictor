import os
from dotenv import load_dotenv
import psycopg2

from ..utils import logs

log = logs.get_logger()

load_dotenv()


def get_db_connection(**kwargs):
    """
    Get a connection to the database.
    If the database does not exist, create it.
    """
    try:
        return psycopg2.connect(
            dbname=kwargs.get('dbname') or os.getenv('DB_NAME'),
            user=os.getenv('DB_USER'),
            password=os.getenv('DB_PASSWORD'),
            host=os.getenv('DB_HOST'),
            port=os.getenv('DB_PORT'),
        )
    except psycopg2.OperationalError as e:
        if f"database \"{os.getenv('DB_NAME')}\" does not exist" in str(e):
            log.warning('Database does not exist. Creating it...')
            create_database(db_name=os.getenv('DB_NAME'))
            return get_db_connection()
        else:
            raise e


def create_database(db_name):
    """Create the database if it does not exist."""
    # Connect to default database (postgres) to create the new one
    conn = get_db_connection(dbname="postgres")
    conn.autocommit = True  # Set autocommit to True to disable transactions and execute the CREATE DATABASE command
    cursor = conn.cursor()
    
    cursor.execute(f"CREATE DATABASE {db_name};")
    log.info(f"Database '{db_name}' created successfully.")
    
    conn.commit()
    cursor.close()
    conn.close()