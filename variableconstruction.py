import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller

class VariableSelection:
    """
    The goal of this class is to check for all return, sentiment and control data whether the input is stationary and ergodic.
    Then, based on the return data, several DataFrames are created, concatenating the different DataFrames based on the
    dates of the return dataframe.
    """
    def __init__(self, df_returns: pd.DataFrame, df_sentiment: pd.DataFrame, df_control: pd.DataFrame,
                 date_col_ret: str, date_col_sent: str, date_col_control: str, return_cols: list, sentiment_cols: list,
                 control_cols: list):
        """
        The goal of this function is to merge all DataFrames by data and to check whether all inputs satsify the
        stationarity conditions.
        """
        self.df_returns = df_returns
        self.df_sentiment = df_sentiment
        self.df_control = df_control

        self.date_col_ret = date_col_ret
        self.date_col_sent = date_col_sent
        self.date_col_control = date_col_control

        # Check cols and store as attributes
        self.returns_col = self.check_list(return_cols)
        self.sentiment_cols = self.check_list(sentiment_cols)
        self.control_cols = self.check_list(control_cols)

        # Check if cols are in the DataFrames
        self.check_input(df_returns, [date_col_ret] + self.returns_col)
        self.check_input(df_sentiment, [date_col_sent] + self.sentiment_cols)
        self.check_input(df_control, [date_col_control] + self.control_cols)


    @staticmethod
    def check_list(input_list: list):
        """
        Function that checks group list, if the input list is a group_list and empty, create empty list.
        :param input_list: list of strings
        :raises ValueError: Cannot parse input_list to type list. If the input can not be made into a list of strings.
        :return: input_list: Input list made into a list of strings, if it was empty, empty list
        """
        if isinstance(input_list, type(None)):
            input_list = []
        elif isinstance(input_list, str):
            input_list = [input_list]
        elif isinstance(input_list, list):
            input_list = input_list
        else:
            raise ValueError(f"Cannot parse '{input_list}' {type(input_list)} to type list.")
        return input_list


    @staticmethod
    def check_input(df, cols):
        """
        Check if the columns are actually in the DataFrame
        """
        if (cols is not None) and not all(x in df.columns.tolist() for x in cols):
            bool_error_cols = [x not in df.columns.tolist() for x in cols]
            raise ValueError("Input columns' values should be None or list of column names from df, {}".format(
                [x for x, y in zip(cols, bool_error_cols) if y == True]))


    def merge_dfs(self):
        """
        This function is created to merge the DataFrames to one DataFrame, only consisting of the relevant columns.
        """
        # Get identical column names for the date columns
        self.df_returns = self.df_returns.rename(columns={self.date_col_ret: 'date'})
        self.df_sentiment = self.df_sentiment.rename(columns={self.date_col_sent: 'date'})
        self.df_control = self.df_control.rename(columns={self.date_col_control: 'date'})

        # Check if input dates are strings, otherwise convert to strings
        if not isinstance(self.df_returns['date'].iloc[0], str):
            self.df_returns['date'] = self.df_returns['date'].dt.strftime('%Y-%m-%d')

        if not isinstance(self.df_sentiment['date'].iloc[0], str):
            self.df_sentiment['date'] = self.df_sentiment['date'].dt.strftime('%Y-%m-%d')

        if not isinstance(self.df_control['date'].iloc[0], str):
            self.df_control['date'] = self.df_control['date'].dt.strftime('%Y-%m-%d')

        # Now merge all columns, only keep overlapping dates
        df_total = pd.merge(self.df_sentiment, self.df_returns, on='date', how='right')
        df_total = pd.merge(df_total, self.df_control, on='date', how='left')

        # only select relevant columns
        cols_to_keep = ['date'] + self.returns_col + self.control_cols + self.sentiment_cols
        df_total = df_total[cols_to_keep]

        return df_total


    def test_stationarity(self):
        """
        This method tests the stationarity of all the variables that are included in this research.
        """
        # Load the merged DataFrame to check stationarity
        df_total = self.merge_dfs()
        cols = self.sentiment_cols

        for col in cols:
            df_total[col] = (df_total[col] - np.mean(df_total[col])) / np.std(df_total[col])

        # Apply the Dickey-Fuller test for testing stationarity
        for col in cols:
            result = adfuller(df_total[col].fillna(0))
            # If p-value of result is larger than 0.05, try taking the difference, and see if that works
            if result[1] > 0.05:
                print(f'time series is not stationary, logarithmic difference of {col} is taken')
                result_dif = adfuller(np.log(df_total[col]).diff().dropna())
                # If difference is still non-stationary, raise ValueError, other, replace data in column with difference
                if result_dif[1] > 0.05:
                    raise ValueError(f"Differencing has no effect, time series of {col} is not stationary")
                else:
                    df_total[f'delta_{col}'] = np.log(df_total[col]).diff().fillna(0)
                    print(f'{col} is not stationary, however, the log difference is, standardized log difference included in regression')
                    df_total = df_total.drop(columns = [col])

            # Standardize new variables
            df_total[col] = (df_total[col] - df_total[col].mean()) / df_total[col].std()

        # Standardize control variables
        for col in self.control_cols:
            df_total[col] = (df_total[col] - df_total[col].mean()) / df_total[col].std()

        return df_total


