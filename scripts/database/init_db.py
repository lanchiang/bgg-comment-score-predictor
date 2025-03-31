from src.database.db_config import get_db_connection


def create_table(sql_file: str):
    conn = get_db_connection()
    cursor = conn.cursor()

    with open(sql_file, 'r') as f:
        sql_commands = f.read()
        cursor.execute(sql_commands)

    conn.commit()
    cursor.close()
    conn.close()


if __name__ == '__main__':
    create_table('schema.sql')
    print('Table created successfully')