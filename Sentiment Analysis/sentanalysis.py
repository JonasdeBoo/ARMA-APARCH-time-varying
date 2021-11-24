import pandas as pd
import numpy as np
from fs_sentaggregation import weighted_sentiment_calculator
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from nltk.tokenize import sent_tokenize
from dateutil import parser
import ast


class TwitterSentimentAnalysis:
    """
    This class performs the sentiment analysis and aggregation. The TwitterSentimentAnalysis class has several methods that
    first process the given Twitter dataset, then use this adjusted dataset to perform calculations, classifications
    of sentiment scores and lastly also aggregates sentiment.
    """
    def __init__(self, tweets: pd.DataFrame, text_col: str, metrics_col: str, created_col: str, trading_dates: list):
        self.tweets = tweets
        self.text_col = text_col
        self.metrics_col = metrics_col
        self.created_col = created_col
        self.trading_dates = trading_dates

        # Create list of public metrics
        self.metrics = list(ast.literal_eval(tweets[metrics_col].iloc[0]).keys())
        self.tweets = self.seperate_dates()

    def seperate_metrics(self):
        """
        This function stores private or public metrics as seperate columns, in doing so changing the tweets DataFrame.
        """
        tweets = self.tweets.copy()

        if not set(self.metrics).issubset(self.tweets.columns):
            tweets[self.metrics_col] = tweets[self.metrics_col].apply(lambda x: eval(x))
            df_public_metrics = pd.json_normalize(tweets[self.metrics_col])

            tweets = tweets.join(df_public_metrics)

        return tweets

    def seperate_dates(self):
        """
        This method seperates times and dates of each tweets. Original created at attributes of tweets are in ISO 8604 format.
        """
        tweets = self.tweets.copy()

        tweets[self.created_col] = tweets[self.created_col].apply(lambda date: parser.parse(date))

        tweets['date'] = [d.date() for d in tweets[self.created_col]]
        tweets['time'] = [d.time() for d in tweets[self.created_col]]

        tweets.drop(self.created_col, axis=1)

        return tweets


    def sentiment_classification(self):
        """
        This method classifies tweets based on the sentiment each separate sentence in the tweet has.
        Within each tweet, the sentiment across tweets is averaged for all sentences.
        """

        tweets = self.seperate_metrics()

        # Instantiate Sentiment Intensity Analyzer
        analyzer = SentimentIntensityAnalyzer()

        # Update lexicon
        analyzer.lexicon.update(new_words)

        compounded_sentiment_series = []

        for tweet in tweets[self.text_col]:
            tokenized_tweets = sent_tokenize(tweet)

            compound_sentiment_tokenized_tweets = []
            for sentence in tokenized_tweets:
                compound_sentiment_tokenized_tweets += [analyzer.polarity_scores(sentence)['compound']]

            compounded_sentiment_series.append(np.mean(compound_sentiment_tokenized_tweets))

        tweets['compounded sentiment'] = compounded_sentiment_series

        # Calculate the direction of the sentiment, if compounded sentiment > 0.05, direction is positive, if compounded
        # sentiment < -.05 the direction is negative, otherwise the direction is neutral
        senti_direction_series = []
        for i in range(len(tweets)):
            if tweets['compounded sentiment'].iloc[i] > 0.05:
                senti_direction_series.append('pos')
            elif tweets['compounded sentiment'].iloc[i] < -0.05:
                senti_direction_series.append('neg')
            else:
                senti_direction_series.append('neu')

        tweets['sentiment direction'] = senti_direction_series

        return tweets

    def calculate_daily_sent(self):
        """
        Method that aggregates each tweet per day (or also per time period (i.e. market open vs. market close)
        """
        tweets = self.sentiment_classification().copy()

        # Define dates where tweets were sent and days when markets where open
        trading_dates = self.trading_dates
        unique_dates = tweets['date'].unique().tolist()

        # Define polarities
        polarities =['neg', 'pos', 'neu']

        # Instantiate empty DataFrame
        df_weighted_sentiment = pd.DataFrame()
        df_weekend = pd.DataFrame()

        # Create fisher score for each unique date in the tweets dataframe
        for i in range(len(unique_dates)-1):
            if unique_dates[i] in trading_dates:
                # Only if yesterday was a trading day as well, otherwise the Tweets belong to the changes attributed to
                # bthe returns over the weekend
                if unique_dates[i-1] in trading_dates:
                    bool_array = unique_dates[i] == tweets['date']
                    df_tweets = tweets[bool_array]

                    # Empty weekend DataFrame
                    df_weekend = pd.DataFrame()

                    weighted_sentiment = weighted_sentiment_calculator(df_tweets, self.metrics, 'sentiment direction',
                                                                           polarities, 'compounded sentiment', unique_dates[i], 'open')
                    df_weighted_sentiment = df_weighted_sentiment.append(weighted_sentiment)



            # Now, add weekend days to DataFrame df_weekend, which will be emptied as soon as a date from unique_dates
            # is in trading_days again
            else:
                bool_array = unique_dates[i] == tweets['date']
                daily_tweets = tweets[bool_array]

                df_weekend = df_weekend.append(daily_tweets)

                # Append premarket tweets the day after the weekend to the weekend tweet DataFrame
                if unique_dates[i+1] in trading_dates:
                    bool_array_2 = unique_dates[i+1] == tweets['date']
                    tweets_after_weekend = tweets[bool_array_2]
                    # Append premarket tweets after weekend to df_weekend
                    df_weekend = df_weekend.append(tweets_after_weekend)
                    df_tweets = df_weekend

                    # Calculate weighted sentiment over the weekend and assert the score to premarket score of day
                    # after the weekend.
                    weighted_sentiment = weighted_sentiment_calculator(df_tweets, self.metrics, 'sentiment direction',
                                                                       polarities, 'compounded sentiment',
                                                                       unique_dates[i+1], 'open')

                    df_weighted_sentiment = df_weighted_sentiment.append(weighted_sentiment)

        df_weighted_sentiment = df_weighted_sentiment.fillna(0).reset_index(drop=True)

        return df_weighted_sentiment


# Create dictionary of new words (Adjusted VADER)
new_words = {'sustainable': 1.5,
             'innovation': 1.5,
             'global warming': -1.5,
             'pollution': -1.5,
             'human': 1.5,
             'responsible': 1.5,
             'restore': 1.5,
             'environmental damage': -1.5,
             'layoff': -1.5,
             'un-green': -1.5,
             'ecofriendly': 1.5,
             'ESG': 1.5,
             'lawsuit': -1.5,
             'sued': -1.5,
             'allegation': -1.5,
             'discrimination': -1.5,
             'environmental': 1.5,
             'unequality': -1.5,
             'unequal': -1.5,
             'greenhouse gas': -1.5,
             'emmission': -1.5,
             'oil': -1.5,
             'favoritism': -1.5,
             'disrespectful': -1.5,
             'asocial': -1.5,
             'social': 1.5,
             'trash': -1.5,
             'wasteful': -1.5,
             'garbage': -1.5,
             'plastic waste': -1.5,
             'lose': -1.5,
             'abuse': -1.5,
             'impoverished': -1.5,
             'toxic': -1.5,
             'dumping': -1.5,
             'obesity': -1.5,
             'clean': 1.5,
             'cleanenergy': 1.5,
             'renewable': 1.5,
             'impact investing': 1.5,
             'CSR': 1.5,
             'human rights': 1.5,
             'recycling': 1.5,
             'renewables': 1.5,
             'green': 1.5,
             'ungreen': -1.5,
             'environmentally friendly': 1.5,
             'sexism': -1.5,
             'ethical': 1.5,
             'climate change': -1.5,
             'climatechange': -1.5,
             'climate disaster': -1.5,
             'climate disruption': -1.5,
             'polluting': -1.5,
             'harassment': -1.5,
             'unsafe': -1.5,
             'insecure': -1.5,
             }