import os
import requests
import json
import pandas as pd
import time
from dotenv import load_dotenv

load_dotenv()  # Load variables from dotenv


class CollectTwitterData:
    """
    This is a class that returns twitter data in a DataFrame. Specify the number of pages (one page can host at most
    max_results), so tweets in DataFrame is at its maximum n_pages*max_results.

    Fill in company name and company ticker (can be from file) to search for ony company, specify start and end date to
    indicate window in which to look for tweets.

    This class automatically filters out retweets and searches for english tweets.

    """
    def __init__(self, n_pages: int, max_results: int, start_date: str, end_date: str, critical_date, company_name: str,
                 official_comp_name: str, company_ticker: str, counter: int, negation_filter: str=None):
        self.n_pages = n_pages
        self.max_results = max_results
        self.start_date = start_date
        self.end_date = end_date
        self.critical_date = critical_date
        self.company_name = company_name
        self.official_company_name = company_name
        self.company_ticker = company_ticker
        self.negation_filter = negation_filter
        self.counter = counter


        # To set your environment variables in your terminal run the following line:
        self.bearer_token = os.environ.get("BEARER_TOKEN")
        self.search_url = os.environ.get("SEARCH_URL")

        # Optional params: start_time,end_time,since_id,until_id,max_results,next_token,
        # expansions,tweet.fields,media.fields,poll.fields,place.fields,user.fields
        company_name = f"(\{company_name}\ OR \{official_comp_name}\ OR ${company_ticker} )"
        esg_dict = "(eco-friendly OR harassment OR fraud OR sustainable OR governance OR responsible  OR policy OR innovation OR \"plastic waste\" OR " \
                   "beneficial OR unhealthy OR pollution OR \"climate change\" OR \"clean energy\" OR carbon OR ethical " \
                   "OR \"green energy\" OR \"green ambitions\" OR \"greenhouse gas\" OR exclusions OR negative OR \"human-rights\" OR \"human rights\" OR " \
                   "\"impact investing\" OR slavery OR renewable OR \"energy transition\" OR \"sustainable transition\" " \
                   "OR SDG OR unfair OR employee OR employees OR discrimination OR sexism OR ESG OR \"Corporate Social Responsibility\"" \
                   " OR \"green development\" OR inclusion OR waste OR favoritism OR impoverished OR obesity OR scarcity" \
                   "OR layoff OR unemployment OR renewables OR inclusion \"employee bonus\")"
        attribute_filter = "(-is:retweet -is:reply -is:quote -is:nullcast)"
        company_negation_filter = self.negation_filter
        language_filter = "(lang:en)"

        if company_negation_filter is None:
            self.query = self.company_name + esg_dict + attribute_filter + language_filter

        else:
            self.query = company_name + esg_dict + attribute_filter + company_negation_filter + language_filter


    @staticmethod
    def create_headers(bearer_token):
        headers = {"Authorization": "Bearer {}".format(bearer_token)}
        return headers


    @staticmethod
    def connect_to_endpoint(url, headers, params):
        response = requests.request("GET", url, headers=headers, params=params)

        if response.status_code != 200:
            print("Error, sleep for 15 seconds, error is {}".format(response.text))
            time.sleep(15)

        # Each time we get a 200 response, lets exit the function and return the response.json
        else:
            return response.json()


    def find_tweets(self, end_date, counter):
        """
        This method generates tweets and stores them on n_pages. If the connection is lost, the connection will be
        re-established and if a segment of tweets is empty,
        """

        tweets = pd.DataFrame()

        i = 0

        while i <= self.n_pages:
            counter += 1

            if counter >= 299:
                time.sleep(900)
                counter = 0

            if i == 0:
                query_params = {
                    'query': self.query,
                    'start_time': self.start_date,
                    'end_time': end_date,
                    'max_results': self.max_results,
                    'expansions': 'author_id',
                    'tweet.fields': 'created_at,public_metrics',
                    'user.fields': 'username'
                }

            else:
                if len(data_dict['meta']) >= 4:
                    next_token = data_dict['meta']['next_token']
                    query_params = {
                        'query': self.query,
                        'start_time': self.start_date,
                        'end_time': end_date,
                        'max_results': self.max_results,
                        'expansions': 'author_id',
                        'tweet.fields': 'created_at,public_metrics',
                        'user.fields': 'username',
                        'next_token': next_token
                        }

            time.sleep(1)
            headers = self.create_headers(self.bearer_token)

            try:
                json_response = self.connect_to_endpoint(self.search_url, headers, query_params)
                json_data = json.dumps(json_response, indent=4, sort_keys=True)
                data_dict = json.loads(json_data)

            except requests.ConnectionError:
                continue

            print(data_dict['meta']['result_count'])

            if data_dict['meta']['result_count'] != 0:
                tweets = tweets.append(pd.DataFrame.from_dict(data_dict['data']))
                print(tweets.iloc[-1])
                time.sleep(1) # As only 1 request per second can be made.
                if data_dict['meta']['result_count'] < (self.max_results - 20):
                    i = self.n_pages + 1
                else:
                    i += 1
            else:
                break

        return tweets, counter

    def get_tweets(self):
        """
        This method iterates the method find_tweets as long as the latest tweet previously retrieved is before the
        defined end date
        """

        tweets, counter = self.find_tweets(self.end_date, self.counter)

        while tweets.iloc[-1].created_at > self.critical_date:
            new_tweets, counter = self.find_tweets(tweets.iloc[-1].created_at, counter)

            tweets = tweets.append(new_tweets)

        return tweets, counter


