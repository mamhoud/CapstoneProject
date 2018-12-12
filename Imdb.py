import pandas as pd
import re  # regular expressions
from nltk.corpus import stopwords # Import the stop word list
from collections import Counter
from nltk.stem.snowball import SnowballStemmer
from sklearn.utils import shuffle


class Imdb:
    """
    Class which abstracts IMDB dataset and some useful utilities on that data
    author: rai.skumar@gmail.com
    """

    # class or static variable
    special_chars = re.compile("[^A-Za-z]+")  # characters which are invalid in review/text

    def __init__(self, data_file):
        """
        Constructor which acts as initializer for data
        :param data_file: local path of input file
        """
        self.data = pd.read_csv(data_file, header=0, delimiter="\t", quoting=3)  # pandas dataframe
        self.data = shuffle(self.data)
        self.labels = self.data['sentiment'].values   # put sentiment in numpy array
        self.inputs = self.data['review'].values      # put review values in numpy array
        self.cleaned_inputs = []                      # cleaned input array
        print('IMDB dataset loaded successfully!')

    def print_meta_data(self):
        print('Total Number Of Records = {}'.format(len(self.data)))
        print('Column Names ={}'.format(self.data.columns.values))
        print('\n Print first 5 rows-')
        print(self.data.head(5))
        print('Sentiment Distribution :\n')

    def get_unique_words(self, cleaned_text=True):
        """
        Returns the set of unique words in the dataset.
        If the cleaned_text is true then the operation will be performed on self.cleaned_inputs else on the self.inputs
        :param cleaned_text:
        :return: set of unique words
        """
        input_arr = self.inputs
        words = []

        if cleaned_text:
            input_arr = self.cleaned_inputs

        for review in input_arr:
            for r in review.split():
                words.append(r)

        unique_words = set(words)
        print('Total words count ={}'.format(len(words)))
        print('Unique words count ={}'.format(len(unique_words)))
        return unique_words

    def get_number_of_words(self):
        """
        Calculate number of words in cleaned text
        :return:
        """
        number_of_words = []
        for review in self.cleaned_inputs:
            number_of_words.append(len(review.split()))
        return number_of_words

    @staticmethod
    def get_all_text(input_arr):
        """
        Join together all the text of the input arr to form a single text
        :return:
        """
        return " ".join([r for r in input_arr])

    @staticmethod
    def count_text(text):
        """
        Split the text by whitespace and then create Counter
        :param text:
        :return:
        """
        words = re.split("\s+", text)
        return Counter(words)   # Count up the occurrence of each word.

    def get_review_count(self):
        """
        Performs count on positive and negative reviews.
        The counting is performed on the dataframe object
        :return:
        """
        positive = len([sentiment for sentiment in self.data['sentiment'] if sentiment == 1])
        negative = len([sentiment for sentiment in self.data['sentiment'] if sentiment == 0])
        print('Positive Review Count = {}'.format(positive))
        print('Negative Review Count = {}'.format(negative))
        return positive, negative

    def clean_all_reviews(self, remove_stopwords=False, stemmerize=False, remove_small_words=False):
        """
        Cleans the review data
        :return: cleaned input values
        """
        for val in self.inputs:
            cleaned = Imdb.clean_text(val)

            if remove_stopwords:
                cleaned = Imdb.remove_stop_words(cleaned)
            if stemmerize:
                cleaned = Imdb.stemmerize(cleaned)
            if remove_small_words:
                cleaned = Imdb.remove_small_words(cleaned)

            self.cleaned_inputs.append(cleaned)

        return self.cleaned_inputs

    @staticmethod
    def clean_text(review):
        """
        Remove punctuation, numbers and special characters. Also convert all chars to lower case.
        :param review:
        :return:
        """
        review = review.lower()

        # remove all non alphabetic chars with space
        clean_text = re.sub(Imdb.special_chars, " ", review)
        return clean_text

    @staticmethod
    def remove_stop_words(review):
        """
        Remove stop words from the given text/sentence(s)
        NLTK(Natural Language Toolkit) in python has a list of stopwords stored in 16 different languages.
        :param review:
        :return:
        """
        stops = set(stopwords.words("english"))  # set of stop words

        review = review.split()
        words = [w for w in review if not w in stops]

        review = " ".join(words) # joins the words array back to text
        return review

    @staticmethod
    def remove_small_words(review):
        """
        Remove all words of length 2
        :param review:
        :return:
        """
        review = review.split()
        words = [w for w in review if len(w) > 2]

        review = " ".join(words) # joins the words array back to text
        return review

    @staticmethod
    def stemmerize(review):
        """
        STEMMER helps to reduce the vocabulary size drastically
        After Stemmer process words like loves, love, loving will be treated as one word instead of 3 words!
        :param review:
        :return:
        """
        stemmer = SnowballStemmer("english")  # we are dealing with english words/reviews

        review = review.split()
        words = [stemmer.stem(w) for w in review]

        review = " ".join(words)  # joins the words array back to text
        return review

    @staticmethod
    def get_tokens(reviews):
        """
        Generate token from review text
        :param reviews:
        :return:
        """
        return [review.split() for review in reviews]