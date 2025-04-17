import unittest

from bgg_playground.database.db_config import get_db_connection


class TestDBConfig(unittest.TestCase):
    
    def test_get_db_config(self):
        with get_db_connection() as db_connection:
            self.assertIsNotNone(db_connection)
            self.assertEqual(db_connection.info.dbname, 'testdb')
            self.assertEqual(db_connection.info.user, 'postgres')
            self.assertEqual(db_connection.info.host, 'localhost')
            self.assertEqual(db_connection.info.port, 5432)
