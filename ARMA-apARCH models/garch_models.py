import pandas as pd
import numpy as np
from mean_model import MeanModel


class ArmaApARCHX:
    """
    This class can be called to compute the parameters of an ARMA-apARCH-X model.User can specify exogenous columns,
    such as control_cols and sentiment_cols. These columns are first checked to be part of the DataFrame, and whether
    they are in the right format (i.e. a list).

    Model that will be used is ARMA(P,Q)-apARCH(1,1)-X(L_k). Lags of conditional mean process are determined by
    different class, as are the optimal lags of the exogenous variables, please consider bic_find_lags.py.

    Mean model follows the specification as given in mean_model.MeanModel
    """

    def __init__(self, df: pd.DataFrame, returns_col: str, lag_arma: list = [1, 1], exog_cols: list = None,
                 lag_exog:  list = None, params: list = None):
        """
        Initialization method of ARMA-apARCH-X class. All arguments are stored as attributed and input columns are
        checked to be in the input DataFrame. Otherwise, a ValueError is raised.
        """
        self.df = df
        self.T = len(df)
        self.returns_col = returns_col
        self.exog_cols = self.check_list(exog_cols)
        self.k = len(self.exog_cols)  # Calculate length of list of control variables
        self.params = params

        if lag_arma is None:
            self.lag_arma = [0,0]
        else:
            self.lag_arma = self.check_list(lag_arma)

        if lag_exog is None:
            self.lag_exog = [0]
        else:
            self.lag_exog = self.check_list(lag_exog)

        # Check if columns (control_columns and sentiment_columns) are actually in df
        if (self.exog_cols is not None) and not all(x in self.df.columns.tolist() for x in self.exog_cols):
            bool_error_cols = [x not in self.df.columns.tolist() for x in self.exog_cols]
            raise ValueError("Input columns' values should be None or list of column names from df, {}".format(
                [x for x, y in zip(self.exog_cols, bool_error_cols) if y]))

        # If params is None, use params from mean model and generate params for the rest, to estimate the model with
        # random params
        if params is None:
            arma_params = MeanModel(df=df,
                                    y_col=returns_col,
                                    lags=lag_arma).params
            aparch_params = self.generate_random_params((4 + np.sum(self.lag_exog) * 2))
            self.params = np.concatenate([arma_params.values[:-1], aparch_params])

    @staticmethod
    def generate_random_params(n_params):
        """
        Method that generates random parameter for initialization.
        """
        # Calculate initial parameters, must satisfy nonnegativity constraint.
        params = [0.005] * n_params

        return params

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

    def conditional_volatility(self):
        """
        Function that defines the epsilon and the sigma2 after inserting the variables. The residuals
        of the mean equation are retrieved via running the class MeanModel with no exogenous variables.
        """
        # Split params in conditional mean params (ARMA) and apARCH-X params
        arma_params = self.params[:1+np.sum(self.lag_arma)]
        aparchx_params = self.params[1+np.sum(self.lag_arma):]

        # First, retrieve the residuals of the mean model
        y_hat, et = MeanModel(df=self.df,
                              y_col=self.returns_col,
                              lags=self.lag_arma,
                              params=arma_params).conditional_mean()

        # Give name to params that occur every time
        omega = aparchx_params[0]
        beta = aparchx_params[1]
        alpha = aparchx_params[2]
        phi = aparchx_params[3]

        # Construct empty result arrays (and insert initial value, hence the + 1)
        sigma2 = np.ones(self.T,) * omega
        et_lagged = pd.Series(et).shift(1).fillna(et.mean()).values

        # Now, for each element in the ARMA-apARCH-X function, add it to the conditional volatility function sigma2
        sigma2 = sigma2 + alpha * ((abs(et_lagged) - phi * et_lagged) ** 2)

        # Define exogenous lags
        if len(self.exog_cols) > 0:
            df_xlag = pd.DataFrame()
            for lag, col in zip(self.lag_exog, self.exog_cols):
                if lag > 0:
                    x_lagged = pd.concat([self.df[col].shift(j + 1) for j in range(lag)], axis=1)
                    x_lagged.columns = [f'Lag {col} {j + 1}' for j in range(lag)]
                    x_lagged = x_lagged.fillna(self.df[col].mean())
                    df_xlag = pd.concat([df_xlag, x_lagged], axis=1)

            x_lagged = df_xlag.values

            # Define params describing the exogenous variables
            phi_ik = aparchx_params[4:4 + np.sum(self.lag_exog)]
            pi_ik = aparchx_params[4 + np.sum(self.lag_exog):]

            # Define zeta_kt, which captures the asymmetric effects
            zeta_kt = (abs(x_lagged) - x_lagged * phi_ik) ** 2

            # Add exogenous variables
            sigma2 = sigma2 + zeta_kt @ pi_ik

        # add lagged volatility
        sigmalag = np.zeros(1)

        for t in range(self.T):
            sigma_incr = beta * sigmalag
            sigma2[t] = sigma2[t] + sigma_incr
            sigmalag = sigma2[t]

        return sigma2, et


class ArmaXapARCH:
    """
    This class can be called to compute the parameters of an ARMA-X-apARCH model. Here, the exogenous variables enter
    the mean equation instead of the variance equation.

    Model that will be used is ARMA(P,Q)-X(L_k)-apARCH(1,1). Lags of conditional mean process are determined by
    different class. The calibration of the optimal lag of possible exogenous variables will be done via the
    AIC criterion in this class.
    """

    def __init__(self, df: pd.DataFrame, returns_col: str, lag_arma: list = [1, 1], exog_cols: list = None,
                 lag_exog:  list = None, params: list = None):
        """
        Initialization method of ARMA-apARCH-X class. All arguments are stored as attributed and input columns are
        checked to be in the input DataFrame. Otherwise, a ValueError is raised.
        """
        self.df = df
        self.T = len(df)
        self.returns_col = returns_col
        self.exog_cols = self.check_list(exog_cols)
        self.k = len(self.exog_cols)  # Calculate length of list of control variables
        self.params = params

        if lag_arma is None:
            self.lag_arma = [0]
        else:
            self.lag_arma = self.check_list(lag_arma)

        if lag_exog is None:
            self.lag_exog = [0]
        else:
            self.lag_exog = self.check_list(lag_exog)

        # Check if columns (control_columns and sentiment_columns) are actually in df
        if (self.exog_cols is not None) and not all(x in self.df.columns.tolist() for x in self.exog_cols):
            bool_error_cols = [x not in self.df.columns.tolist() for x in self.exog_cols]
            raise ValueError("Input columns' values should be None or list of column names from df, {}".format(
                [x for x, y in zip(self.exog_cols, bool_error_cols) if y]))

        # If params is None, use params from mean model and generate params for the rest, to estimate the model with
        # random params
        if params is None:
            arma_params = MeanModel(df=df,
                                    y_col=returns_col,
                                    lags=lag_arma,
                                    x_cols=exog_cols,
                                    x_lags=lag_exog).params
            aparch_params = self.generate_random_params(4)

            self.params = np.concatenate([arma_params.values[:-1], aparch_params])

    @staticmethod
    def generate_random_params(n_params):
        """
        Method that generates random parameter for initialization.
        """
        # Calculate initial parameters, must satisfy nonnegativity constraint.
        params = [0.005] * n_params

        return params

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


    def conditional_volatility(self):
        """
        Function that defines the epsilon and the sigma2 after inserting the variables. The residuals
        of the mean equation are retrieved via running the class MeanModel with no exogenous variables.
        """
        # Split params in conditional mean params (ARMA) and apARCH-X params
        arma_params = self.params[:1+np.sum(self.lag_arma)+np.sum(self.lag_exog)]
        aparchx_params = self.params[1+np.sum(self.lag_arma)+np.sum(self.lag_exog):]

        # First, retrieve the residuals of the mean model
        y_hat, et = MeanModel(df=self.df,
                              y_col=self.returns_col,
                              lags=self.lag_arma,
                              params=arma_params,
                              x_cols=self.exog_cols,
                              x_lags=self.lag_exog).conditional_mean()

        # Give name to params that occur every time
        omega = aparchx_params[0]
        beta = aparchx_params[1]
        alpha = aparchx_params[2]
        phi = aparchx_params[3]

        # Construct empty result arrays (and insert initial value, hence the + 1)
        sigma2 = np.ones(self.T,) * omega
        et_lagged = pd.Series(et).shift(1).fillna(et.mean()).values

        # Now, for each element in the ARMA-apARCH-X function, add it to the conditional volatility function sigma2
        sigma2 = sigma2 + alpha * ((abs(et_lagged) - phi * et_lagged) ** 2)

        # add lagged volatility
        sigmalag = np.zeros(1)

        for t in range(self.T):
            sigma_incr = beta * sigmalag
            sigma2[t] = sigma2[t] + sigma_incr
            sigmalag = sigma2[t]

        return sigma2, et


class ArmaApArchXGarch:
    """
    This class models the returns as an ARMA(P,Q)-apARCH(1,1)-X process, where some of the exogenous regressors are
    allowed to follow a GARCH specification as well. Thus, each model takes for each exogenous variable the lagged
    level as well as the lagged volatility into the GARCH specification of the returns.
    """

    def __init__(self, df: pd.DataFrame, returns_col: str, lag_arma: list = [1, 1], exog_cols: list = None,
                 lag_exog_level: list = None, params: list = None, xgarch_cols: list=None, lag_exog_sigma: list = None):
        """
        Initialization method of ARMA-apARCH-X class. All arguments are stored as attributed and input columns are
        checked to be in the input DataFrame. Otherwise, a ValueError is raised.
        """
        self.df = df
        self.T = len(df)
        self.returns_col = returns_col
        self.exog_cols = self.check_list(exog_cols)
        self.k = len(self.exog_cols)  # Calculate length of list of control variables

        if lag_arma is None:
            self.lag_arma = 0
        else:
            self.lag_arma = self.check_list(lag_arma)
        if lag_exog_level is None:
            self.lag_exog_level = [0]
        else:
            self.lag_exog_level = self.check_list(lag_exog_level)

        if lag_exog_sigma is None:
            self.lag_exog_sigma = [0]
        else:
            self.lag_exog_sigma = self.check_list(lag_exog_sigma)

        self.params = params
        self.xgarch_cols = xgarch_cols

        # Check if columns (control_columns and sentiment_columns) are actually in df
        if (len(self.exog_cols) > 0) and not all(x in self.df.columns.tolist() for x in self.exog_cols):
            bool_error_cols = [x not in self.df.columns.tolist() for x in self.exog_cols]
            raise ValueError("Input columns' values should be None or list of column names from df, {}".format(
                [x for x, y in zip(self.exog_cols, bool_error_cols) if y]))

        if (len(self.xgarch_cols) > 0) and not all(x in self.df.columns.tolist() for x in self.xgarch_cols):
            bool_error_cols = [x not in self.df.columns.tolist() for x in self.xgarch_cols]
            raise ValueError("Input columns' values should be None or list of column names from df, {}".format(
                [x for x, y in zip(self.xgarch_cols, bool_error_cols) if y]))

        # If params is None, use params from mean model and generate params for the rest, to estimate the model with
        # random params
        if params is None:
            arma_params = MeanModel(df=df,
                                    y_col=returns_col,
                                    lags=lag_arma).params

            aparch_params = self.generate_random_params(4 + np.sum(self.lag_exog_sigma) +
                                                        2 * np.sum(self.lag_exog_level))

            self.params = np.concatenate([arma_params.values[:-1], aparch_params])

    @staticmethod
    def generate_random_params(n_params):
        """
        Method that generates random parameter for initialization.
        """
        # Calculate initial parameters, must satisfy nonnegativity constraint.
        params = [0.005] * n_params

        return params

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

    def conditional_volatility(self):
        """
        Function that defines the epsilon and the sigma2 after inserting the variables. The residuals
        of the mean equation are retrieved via running the class MeanModel with no exogenous variables.
        """
        # Split params in conditional mean params (ARMA) and apARCH-X and X-GARCH params
        arma_params = self.params[:1 + np.sum(self.lag_arma)]
        aparchx_params = self.params[1 + np.sum(self.lag_arma):]

        # First, retrieve the residuals of the mean model
        y_hat, et = MeanModel(df=self.df,
                              y_col=self.returns_col,
                              lags=self.lag_arma,
                              params=arma_params).conditional_mean()

        # Give name to params that occur every time
        omega = aparchx_params[0]
        beta = aparchx_params[1]
        alpha = aparchx_params[2]
        phi = aparchx_params[3]

        # Construct empty result arrays (and insert initial value, hence the + 1)
        sigma2 = np.ones(self.T, ) * omega
        et_lagged = pd.Series(et).shift(1).fillna(et.mean()).values

        # Now, for each element in the ARMA-apARCH-X function, add it to the conditional volatility function sigma2
        sigma2 = sigma2 + alpha * ((abs(et_lagged) - phi * et_lagged) ** 2)

        # split aparchx_params
        pi_ik = aparchx_params[4:(4 + np.sum(self.lag_exog_level) + np.sum(self.lag_exog_sigma))]
        phi_ik = aparchx_params[(4 + np.sum(self.lag_exog_level) + np.sum(self.lag_exog_sigma)):]

        # Define exogenous volatility lags
        if len(self.xgarch_cols) > 0:
            df_sigma_x_lag = pd.DataFrame()
            for lag, col in zip(self.lag_exog_sigma, self.xgarch_cols):
                if lag > 0:
                    x_lagged = pd.concat([np.sqrt(self.df[col].shift(j + 1)) for j in range(lag)], axis=1)
                    x_lagged.columns = [f'Lag {col} {j + 1}' for j in range(lag)]
                    x_lagged = x_lagged.fillna(0)
                    df_sigma_x_lag = pd.concat([df_sigma_x_lag, x_lagged], axis=1)

        # Define exogenous lags
        if len(self.exog_cols) > 0:
            df_x_lag = pd.DataFrame()
            for lag, col in zip(self.lag_exog_level, self.exog_cols):
                if lag > 0:
                    x_lagged = pd.concat([(self.df[col].shift(j + 1)) ** 2 for j in range(lag)], axis=1)
                    x_lagged.columns = [f'Lag {col} {j + 1}' for j in range(lag)]
                    x_lagged = x_lagged.fillna(self.df[col].mean())
                    df_x_lag = pd.concat([df_x_lag, x_lagged], axis=1)

            x_lagged = df_x_lag.values

            # Define zeta_kt, which captures the asymmetric effects
            zeta_kt = (abs(x_lagged) - x_lagged * phi_ik) ** 2

        # Add exogenous variables
        if len(self.exog_cols + self.xgarch_cols) > 0:
            # Combine exogenous level and exogenous volatility into one df
            exog_lagged = pd.concat([pd.DataFrame(zeta_kt), df_sigma_x_lag], axis=1).values

            sigma2 = sigma2 + exog_lagged @ pi_ik

        # add lagged volatility
        sigmalag = np.zeros(1)

        for t in range(self.T):
            sigma_incr = beta * sigmalag
            sigma2[t] = sigma2[t] + sigma_incr
            sigmalag = sigma2[t]

        return sigma2, et


class ExogGarch:
    """
    This class calculates the behaviour of an Exogenous variable following a standard ARMA(1,1)-GARCH(1,1) trajectory.
    """
    def __init__(self, df: pd.DataFrame, col: str, params: list = None):
        """
        Initialization method of ARMA-GARCH class. All arguments are stored as attributed and input columns are
        checked to be in the input DataFrame. Otherwise, a ValueError is raised.

        Col can be any column that an ARMA-GARCH structure is to be imposed on.
        """
        self.df = df
        self.T = len(df)
        self.col = col
        self.params = params

        # Check if columns (control_columns and sentiment_columns) are actually in df
        if (len(self.col) > 0) and not all(x in self.df.columns.tolist() for x in self.col):
            bool_error_cols = [x not in self.df.columns.tolist() for x in self.col]
            raise ValueError("Input columns' values should be None or list of column names from df, {}".format(
                [x for x, y in zip(self.col, bool_error_cols) if y]))

        # If params is None, use params from mean model and generate params for the rest, to estimate the model with
        # random params
        if params is None:
            arma_params = MeanModel(df=df,
                                    y_col=self.col,
                                    lags=[1, 1]).params
            aparch_params = self.generate_random_params(3)

            self.params = np.concatenate([arma_params.values[:-1], aparch_params])

    @staticmethod
    def generate_random_params(n_params):
        """
        Method that generates random parameter for initialization.
        """
        # Calculate initial parameters, must satisfy nonnegativity constraint.
        params = [0.005] * n_params

        return params

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

    def conditional_volatility(self):
        """
        This method structures the exogenous variables as a GARCH process themselves. It is supposed that
        x_t = nu sigma_x, where sigma_x follows a GARCH(1,1) process.
        """
        # split params into ARMA params and garch_x params
        armax_params = self.params[:3]
        garch_params_x = self.params[3:]

        x_hat, et_x = MeanModel(df=self.df,
                                y_col=self.col,
                                lags=[1, 1],
                                params=armax_params).conditional_mean()

        # Give name to params that occur every time
        omega_x = garch_params_x[0]
        beta_x = garch_params_x[1]
        alpha_x = garch_params_x[2]

        # Construct empty result arrays (and insert initial value, hence the + 1)
        sigma2_x = np.ones(self.T, ) * omega_x
        et_lagged = pd.Series(et_x).shift(1).fillna(et_x.mean()).values

        # Add sigma
        sigma2_x = sigma2_x + alpha_x * (et_lagged ** 2)

        # add lagged volatility
        sigmalag = np.zeros(1)

        for t in range(self.T):
            sigma_incr = beta_x * sigmalag
            sigma2_x[t] = sigma2_x[t] + sigma_incr
            sigmalag = sigma2_x[t]

        return sigma2_x, et_x