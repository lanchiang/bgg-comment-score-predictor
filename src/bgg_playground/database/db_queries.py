from bgg_playground.database.db_config import get_db_connection


def get_ratings_with_comments(sample_size=-1):
    with get_db_connection() as conn:
        with conn.cursor() as cursor:
            if sample_size > 0:
                cursor.execute("SELECT * FROM comments as c where c.comment <> \'\' ORDER BY RANDOM() LIMIT %s", (sample_size,))
            else:
                cursor.execute("SELECT * FROM comments as c where c.comment <> \'\'")
            ratings_with_comments = cursor.fetchall()

            return ratings_with_comments