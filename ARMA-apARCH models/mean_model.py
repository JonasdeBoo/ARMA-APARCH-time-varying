import pandas as pd
import statsmodels.tsa.api as sm
import numpy as np


class MeanModel:
    """
    This class defines the mean model of the GARCH specification. The mean model can also take exogenous regressors
    with the number of lags specified by x_lags. If no parameters are provided, this class will estimate them with help
    of the sm.SARIMAX package.

    This class serves as helper function to calculate the behaviour of the conditional mean model when investigating
    the GARCH models.
     """
    def __init__(self, df: pd.DataFrame, y_col: str, lags: list = None, params: list = None, x_cols: list = None,
                 x_lags: list = None):
        """
        Instantiate class of MeanModel, data and lags provided are mandatory, X and lags_X are optional parameters.
        """
        # Set attributes
        self.df = df
        self.y_col = y_col
        self.y = df[y_col]
        self.P = lags[0]
        self.Q = lags[1]
        self.params = params
        self.pnum = self.P + self.Q + 1

        # Store as attributes, but can be None, as some columns reoccur, also asign values to them eventhough they are
        # None
        if x_cols is not None:
            self.X = df[x_cols]
            self.x_cols = x_cols
            self.x_lags = self.check_list(x_lags)
        else:
            self.X = None
            self.x_lags = [0]
            self.x_cols = None

        # Define AR lags
        lags = pd.concat([self.y.shift(i+1) for i in range(self.P)], axis=1)
        lags.columns = ['Lag' + str(i + 1) for i in range(self.P)]
        self._lags = lags.fillna(0.0001).values

        # Define exogenous lags
        if x_cols is not None:
            df_xlag = pd.DataFrame()
            for lag, col in zip(x_lags, x_cols):
                if lag > 0:
                    x_lagged = pd.concat([df[col].shift(j + 1) for j in range(lag)], axis=1)
                    x_lagged.columns = [f'Lag {col} {j+1}' for j in range(lag)]
                    x_lagged = x_lagged.fillna(df[col].mean())
                    df_xlag = pd.concat([df_xlag, x_lagged], axis=1)

            # Store exogenous lags as attribute
            self.x_lagged = df_xlag.values

        # Get starting values of the Regression with help of the statsmodels.tsa.api.SARIMAX package
        if params is None:
            if x_cols is not None:
                model = sm.SARIMAX(endog=self.y, trend='c', exog=df_xlag, order=(self.P, 0, self.Q)).fit()
                self.params = model.params
            else:
                model = sm.SARIMAX(endog=self.y, trend='c', order=(self.P, 0, self.Q)).fit()
                self.params = model.params

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
        elif isinstance(input_list, int):
            input_list = [input_list]
        elif isinstance(input_list, float):
            input_list = [input_list]
        elif isinstance(input_list, list):
            input_list = input_list
        else:
            raise ValueError(f"Cannot parse '{input_list}' {type(input_list)} to type list.")
        return input_list

    def conditional_mean(self):
        """
        This method calculates the conditional mean process. It returns the residuals and the calculated conditional
        mean.
        """
        # Create y and y_lag, containing for each y the lags of order P
        y = self.y.values
        y_lag = self._lags

        if self.x_cols is not None:
            x_lagged = self.x_lagged

        # Split parameters into ARMA parameters and parameters denoting exogenous variables
        lag_x_params = self.params[1:1+np.sum(self.x_lags)]
        arma_params = self.params[1+np.sum(self.x_lags):]

        # If no lags are inserted, then the intercept is used as y_hat
        y_hat = np.ones((len(y),)) * self.params[0]

        if self.P > 0:
            # We need to calculate AR lags
            ar_params = arma_params[:self.P]
            y_hat = y_hat + y_lag @ ar_params

        # If X is not None, include in the regression
        if self.x_cols is not None:
            y_hat = y_hat + x_lagged @ lag_x_params

        if self.Q > 0:
            ma_params = arma_params[self.P:self.P+self.Q]
            ylags = np.ones(self.Q) * self.y.values[0]
            yhat_lags = np.ones(self.Q) * self.params[0]
            lagind = np.arange(0, len(ylags))
            lagind = np.roll(lagind, 1)

            # Iteratively add the lagged error to y_hat, and then create new values of y_hat_lags
            for t in range(len(y)):
                etlag = ylags - yhat_lags
                ma_incr = etlag @ ma_params
                y_hat[t] = y_hat[t] + ma_incr
                ylags = ylags[lagind]
                ylags[0] = y[t]
                yhat_lags = yhat_lags[lagind]
                yhat_lags[0] = y_hat[t]

        # Reshape to get the same array two times
        shape = y_hat.shape
        y = np.reshape(y, shape)

        return y_hat, y - y_hat
