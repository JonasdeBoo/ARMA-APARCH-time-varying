import pandas as pd
import numpy as np


def weighted_sentiment_calculator(df_tweets: pd.DataFrame, metrics: list, direction_col: str, polarities: list,
                                  sentiment_score_col: str, date, close_or_open: str):
    """
    This function calculates per 'block' of tweets the fisher score, based on the provided metrics. This function
    is used in the sentanalysis.py file. Output is a DataFrame containing the date, and the open or close specification
    each trading day has one pre-market close period and open period, if a trading day comes after the weekend,
    all tweets sent in the period since the last market closing time are sent.
    """
    fisher_scores = {}
    # Calculate FS score for every metric available
    for metric in df_tweets[metrics]:
        mean = df_tweets[metric].mean()
        std = df_tweets[metric].std()
        # Set std to very small number if it was zero
        if std == 0:
            std = 1

        score = []
        # Calculcate fisher score for all polarity classes
        for polarity in polarities:
            mu_j_s = df_tweets[df_tweets[direction_col] == polarity][metric].mean()
            n_s = df_tweets[df_tweets[direction_col] == polarity]['text'].count()
            score.append(n_s * ((mu_j_s - mean) ** 2))

        fisher_scores[metric] = sum(score) / (std ** 2)

    # calculate weight of each tweet sent on each day
    tweet_weight = sum(
        [df_tweets[metric] * fisher_scores[metric] for metric in df_tweets[metrics]]) + 1

    # Normalize tweet weight by dividing by the summed tweet weight per day
    normalized_tweet_weight = tweet_weight / tweet_weight.sum()

    # Daily sentiment score is the product of the twitter weights by their respective sentiment scores. Also calculate
    # number of interactions and number of tweets
    period_sentiment_score = sum(normalized_tweet_weight * df_tweets[sentiment_score_col])
    period_interactions = df_tweets[metrics].sum(axis=1).sum()
    n_tweets = len(df_tweets)

    # Store all metrics in a DataFrame
    df_sentiscores = pd.DataFrame({'date': [date],
                                   'close/open': close_or_open,
                                   'sentiment': period_sentiment_score,
                                   'n_interactions': period_interactions,
                                   'n_tweets': n_tweets
                                    })

    return df_sentiscores