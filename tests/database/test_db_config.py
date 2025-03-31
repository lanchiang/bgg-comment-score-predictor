import unittest



class TestDBConfig(unittest.TestCase):
    
    def test_get_db_config(self):
        from src.database.db_config import get_db_connection

        with get_db_connection() as db_connection:
            self.assertIsNotNone(db_connection)
            self.assertEqual(db_connection.info.dbname, 'bgg_comments')
            self.assertEqual(db_connection.info.user, 'lan')
            self.assertEqual(db_connection.info.host, 'localhost')
            self.assertEqual(db_connection.info.port, 5432)
