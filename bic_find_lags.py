import pandas as pd
import numpy as np
from qml_estimation import QuasiMaximumLikelihoodEstimator


class LagsSelection:
    """
    This class is used to find the optimal number of lags to be included in the model, with the help of the BIC.
    That is, the model, with parameters optimizing the quasi likelihood has a BIC of: w * log(n) - 2 * log L. With L
    denoting the likelihood, n the sample size and w the number of parameters. By the model with the lowest BIC is optimal
    as the BIC punishes inclusion of too many parameters.

    This class has three methods, one to estimate the lags of the ARMA process, and two to estimate the optimal lags of
    the exogenous columns. One unconditional on other included lags of exogenous variables, others conditional on the
    lags of other exogenous variables.

    This class exploits the QuasiMaximumLikelihoodEstimator to find the optimal lags. The model evaluated is determined
    by the parameter 'model_type'
    """
    def __init__(self, df: pd.DataFrame, returns_col: str, arma_lags: list = None, exog_cols: list = None,
                 model_type: str = 'asym', x_garch_cols: list = None):
        """
        Initialize class LagsSelection
        """
        self.df = df
        self.T = len(df)
        self.returns_col = returns_col
        self.exog_cols = self.check_list(exog_cols)
        self.model_type = model_type
        self.arma_lags = self.check_list(arma_lags)
        self.x_garch_cols = self.check_list(x_garch_cols)

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


    def find_arma_lags(self):
        """
        This function is created to find the optimal number of lags used in the ARMA(P,Q) mean model. The selection
        will be done on the basis of the BIC selection criterion. This method returns the P and Q lags in a list.

        The exogenous variables are not included in this selection algorithm, as any inclusion provides an extension
        to the normal ARMA-apARCH model.
        """
        dict_ = {}

        # Try parameters up to a week of trading days lag.
        for P in range(1, 4):
            for Q in range(1, 4):
                minimization_result, psi_hat, likelihood, df_params = \
                    QuasiMaximumLikelihoodEstimator(df=self.df,
                                                    returns_col=self.returns_col,
                                                    lags_arma=[P, Q]).optimize_likelihood()
                bic = len(psi_hat) * np.log(self.T) - (2 * self.T * likelihood)
                dict_[P, Q] = bic
                print(dict_)

        # Get key of the ARMA(P,Q) model with the lowest BIC value
        lag_arma_tup = min(dict_, key=dict_.get)
        bic = dict_[lag_arma_tup]
        lag_arma = list(lag_arma_tup)

        return lag_arma, bic


    def find_exog_lags(self):
        """
        This method is created to find for all exogenous columns the optimal number of lags to be included in the model,
        in order to maximize the MLE.

        The selection of the lags happens on the basis of the BIC criterion and is done conditional on the other lag
        parameters. That is, the algorithm iteratively finds the best lags for every variable in exog_cols, when the
        best value is found for lag i, this lag is included when finding the lag i + 1.
        """
        # First, find optimal ARMA(P,Q) lags
        if self.arma_lags is None:
            arma_lags, bic = self.find_arma_lags()
        else:
            arma_lags = self.arma_lags
            minimization_result, psi_hat, likelihood, df_params = QuasiMaximumLikelihoodEstimator(df=self.df,
                                                                                                  returns_col=self.returns_col,
                                                                                                  lags_arma=arma_lags).optimize_likelihood()
            bic = len(psi_hat) * np.log(self.T) - (2 * self.T * likelihood)

        exog_lags = [0] * len(self.exog_cols)
        x_garch_lags = [0] * len(self.x_garch_cols)

        for q in range(1, 4):
            # Take into account statistics at first iteration
            exog_lags_init = exog_lags

            if q == 1:
                bic_init = bic
            else:
                bic_init = dict_[tuple(exog_lags_init)]

            dict_ = {}
            dict_[tuple(exog_lags_init)] = bic_init
            for i, col in enumerate(self.exog_cols):
                # If second iteration and optimal lag was 1
                if exog_lags[i] < q-1:
                    exog_lags[i] = exog_lags[i]
                    dict_[tuple(exog_lags)] = bic
                else:
                    exog_lags[i] = exog_lags[i] + 1

                    # Calculate the optimal parameters given lag of exogenous column i == q, while the rest is equal to zero.
                    minimization_result, psi_hat, likelihood, df_params = \
                        QuasiMaximumLikelihoodEstimator(df=self.df,
                                                        returns_col=self.returns_col,
                                                        lags_arma=arma_lags,
                                                        exog_cols=self.exog_cols,
                                                        lags_exog=exog_lags,
                                                        model_type=self.model_type,
                                                        x_garch_cols=self.x_garch_cols,
                                                        x_garch_lags=x_garch_lags).optimize_likelihood()

                    # Calculate the BIC
                    bic = len(psi_hat) * np.log(self.T) - (2 * self.T * likelihood)
                    dict_[tuple(exog_lags)] = bic
                    print(dict_)

            # Find lags that minimze the BIC
            exog_lags = list(min(dict_, key=dict_.get))
            if q == 1 and 0 in exog_lags:
                exog_lags = [1] * len(exog_lags)
            print(exog_lags)

        # In case the x-garch model is called, also x-garch lags
        if self.model_type == 'x-garch':
            for q in range(1, 4):
                # Take into account statistics at first iteration
                x_garch_lags_init = x_garch_lags
                if q == 1: bic_init = dict_[tuple(exog_lags)]
                else: bic_init = dict_[tuple(x_garch_lags_init)]

                dict_ = {}
                dict_[tuple(x_garch_lags_init)] = bic_init
                for i, col in enumerate(self.x_garch_cols):
                    # If second iteration and optimal lag was 1
                    if x_garch_lags[i] < q - 1:
                        x_garch_lags[i] = x_garch_lags[i]
                        dict_[tuple(x_garch_lags)] = bic
                    else:
                        x_garch_lags[i] = x_garch_lags[i] + 1

                        # Calculate the optimal parameters given lag of exogenous column i == q, while the rest is equal to zero.
                        minimization_result, psi_hat, likelihood, df_params = \
                            QuasiMaximumLikelihoodEstimator(df=self.df,
                                                            returns_col=self.returns_col,
                                                            lags_arma=arma_lags,
                                                            exog_cols=self.exog_cols,
                                                            lags_exog=exog_lags,
                                                            model_type=self.model_type,
                                                            x_garch_cols=self.x_garch_cols,
                                                            x_garch_lags=x_garch_lags).optimize_likelihood()

                        # Calculate the BIC
                        bic = len(psi_hat) * np.log(self.T) - (2 * self.T * likelihood)
                        dict_[tuple(x_garch_lags)] = bic
                        print(dict_)

                # Find lags that minimze the BIC
                x_garch_lags = list(min(dict_, key=dict_.get))
                if q == 1 and 0 in x_garch_lags:
                   x_garch_lags = [1] * len(x_garch_lags)
                print(x_garch_lags)

            # Concatenate x_garch_lags to exog_lags
            exog_lags = exog_lags + x_garch_lags

        return exog_lags
