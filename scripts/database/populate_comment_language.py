from scripts.database.populate_db import populate_comment_lang


def detect_comment_language():
    populate_comment_lang()
    print('Comment language detected successfully')
    


if __name__ == '__main__':
    detect_comment_language()