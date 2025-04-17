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


def detect_comment_language():
    populate_comment_lang()
    log.info('Comment language detected successfully')


if __name__ == '__main__':
    detect_comment_language()