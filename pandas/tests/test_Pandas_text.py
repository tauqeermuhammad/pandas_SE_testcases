import pandas as pd
import unittest
from my_api import count_words
from my_api import remove_stopwords
from my_api import calculate_similarity

class TestCountWords(unittest.TestCase):

    def test_word_count(self):
        df = pd.DataFrame({'text_column': ['This is a test sentence', 'This is another test sentence']})
        df_result = count_words(df, 'text_column')
        self.assertEqual(df_result['word_count'][0], 5)
        self.assertEqual(df_result['word_count'][1], 5)
        
        
class TestRemoveStopwords(unittest.TestCase):

    def test_stopword_removal(self):
        df = pd.DataFrame({'text_column': ['This is a test sentence', 'This is another test sentence']})
        df_result = remove_stopwords(df, 'text_column')
        self.assertEqual(df_result['text_without_stopwords'][0], 'test sentence')
        self.assertEqual(df_result['text_without_stopwords'][1], 'another test sentence')

class TestCalculateSimilarity(unittest.TestCase):

    def test_similarity_score(self):
        df = pd.DataFrame({'text_column1': ['This is a test sentence', 'This is another test sentence'], 
                           'text_column2': ['This is a test sentence', 'This is a different test sentence']})
        df_result = calculate_similarity(df, 'text_column1', 'text_column2')
        self.assertAlmostEqual(df_result['similarity_score'][0], 1.0, places=2)
        self.assertAlmostEqual(df_result['similarity_score'][1], 0.70, places=2)



if __name__ == '__main__':
    unittest.main()
