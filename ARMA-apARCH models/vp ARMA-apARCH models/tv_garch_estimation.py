import pandas as pd
import numpy as np
import scipy.optimize as opt
from tv_garch_models import tvArmaApArchXGarch, tvArmaApARCHX, tvArmaXapARCH


class QuasiMaximumLikelihoodEstimator:
    """
    This class finds for every model in garch_models the parameters that minimize (maximum) the Quasi Log Likelihood,
    as defined by Franq and Thieu (2019).

    This class takes as input all the variables and the model_type, which indicates for which model the parameters
    should be optimized. Furthermore, this class sets parameter constraints for each model in the folder garch_models.

    Lastly, this class has a method that calculates the variance covariance matrix numerically, making use of the
    quasi_log_likelihood method.
    """
    def __init__(self, df: pd.DataFrame, returns_col: str, dep_cols: str, h: int=25, lags_arma: list= [1, 1], exog_cols: list = None,
                 lags_exog: list = None, params: list = None, model_type: str = 'asym', x_garch_cols: list = None,
                 x_garch_lags: list = None, n_iter: int = 400):
        """
        Initialize class QuasiMaximumLikelihoodEstimator
        """
        self.df = df
        self.T = len(df)
        self.returns_col = returns_col
        self.exog_cols = self.check_list(exog_cols)
        self.k = len(self.exog_cols)  # Calculate length of list of control variables

        self.dep_cols = dep_cols
        self.h = h

        # Check input of lag parameters
        if lags_arma is None:
            self.lags_arma = [0]
        else:
            self.lags_arma = self.check_list(lags_arma)

        if lags_exog is None:
            self.lags_exog = [0]
        else:
            self.lags_exog = self.check_list(lags_exog)

        self.params = params
        self.model_type = model_type
        self.x_garch_cols = x_garch_cols

        # Check input of x_garch_lag parameters
        if x_garch_lags is None:
            self.x_garch_lags = [0]
        else:
            self.x_garch_lags = self.check_list(x_garch_lags)

        self.n_iter = n_iter

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

        if not input_list == []:
            if not isinstance(input_list[0], str):
                if not isinstance(input_list[0], int):
                    input_list = [int(i) for i in input_list]
            else:
                input_list = input_list
        else:
            input_list = input_list

        return input_list


    def quasi_log_likelihood(self, params, out=False):
        """
        Here the gaussian likelihood given observations epsilon and sigma2 is created.
        """
        sigma2, et = self._calc_estimates(params)

        # QMLE from Franq and Thieu
        lls = np.log(sigma2) + ((et ** 2) / sigma2)
        lls = np.nan_to_num(lls)       # Replace NaNs with 0
        logLik = np.sum(lls) / self.T

        if logLik == np.inf:
            logLik = 1000
        elif logLik == 0:
            logLik = 1000

        if out is True:
            return logLik, lls
        else:
            return logLik

    def optimize_likelihood(self):
        """
        This method exploits scipy.optimize.minimize to find the parameter values that minimize (maximize) the negative
        likelihood defined in quasi_log_likelihood. Bounds are set dependent on the type of model.
        """
        # Retrieve parameter names and bounds
        params, param_names = self._init_params()
        bnds, cons = self._set_bounds()

        ftol = 1e-07
        minimization_result = opt.minimize(fun=self.quasi_log_likelihood, x0=params, method='SLSQP', jac=self._jac_func,
                                           bounds=bnds, constraints=cons, options={'ftol': ftol, 'maxiter': self.n_iter})

        # Make loop that always continues until solution is reached, that finds new parameters and multiplies number of
        # function evaluations by a factor 3 and reduce the tolerance level. Furthermore, take old parameter guess as
        # starting values
        if not minimization_result.success:
            for i in range(10):
                if not minimization_result.success:
                    if ftol < 0.1:
                        ftol = ftol * 10
                        print(f'tolerance level is changed to {ftol}')
                    minimization_result = opt.minimize(self.quasi_log_likelihood, x0=minimization_result.x,
                                                       method='SLSQP', jac=self._jac_func, bounds=bnds, constraints=cons,
                                                       options={'ftol': ftol, 'maxiter': self.n_iter * 3})

            print(minimization_result)

        # Optimal parameters
        psi_hat = list(minimization_result.x)
        likelihood = -self.quasi_log_likelihood(psi_hat)

        df_params = pd.DataFrame(
            {'param names': param_names,
             'psi_hat': psi_hat})

        return minimization_result, psi_hat, likelihood, df_params

    def _jac_func(self, params):
        """
        This function calculates the Jacobian, which helps improve the speed of the minimization result. The Jacobian
        is calculated using finite differencing.
        """
        # Compute Jacobian with respect to set of parameters x
        x = params

        # Calculate the likelihood and the length of the parameters
        n = len(x)

        # Define step sizes
        eps = 1e-07
        scores = np.zeros((self.T, n))

        for i in range(n):
            delta = np.zeros(n)
            delta[i] = eps

            loglik, logliksplus = self.quasi_log_likelihood(x + delta, out=True)
            loglik, logliksminus = self.quasi_log_likelihood(x - delta, out=True)

            scores[:, i] = (logliksplus - logliksminus) / (2 * eps)

        jac = scores.mean(axis=0)

        return jac

    def _hessian_2sided(self, params):
        """
        This method calculates the 2 sided Hessian matrix.
        """
        if params is None:
            params, param_names = self._init_params()

        # Evaluate Hessian at function value f
        f = self.quasi_log_likelihood(params)
        x = params

        # Define step sizes
        h = 1e-05 * np.abs(x)

        xh = x + h
        h = xh - h

        # K is number of params, define h, function plus and function minus values
        K = np.size(x, 0)
        h = np.diag(h)
        fp = np.zeros(K)
        fm = np.zeros(K)

        # Create matrices
        hh = np.diag(h)
        hh = hh.reshape((K, 1))
        hh = hh @ hh.T

        # Create empty arrays to compute double backward and forward steps
        H = np.zeros((K, K))
        fpp = np.zeros((K, K))
        fmm = np.zeros((K, K))

        for i in range(K):
            fp[i] = self.quasi_log_likelihood(x + h[i])
            fm[i] = self.quasi_log_likelihood(x - h[i])
            for j in range(i, K):
                fpp[i, j] = self.quasi_log_likelihood(x + h[i] + h[j])
                fpp[j, i] = fpp[i, j]
                fmm[i, j] = self.quasi_log_likelihood(x - h[i] - h[j])
                fmm[j, i] = fmm[i, j]

        for i in range(K):
            for j in range(i, K):
                H[i, j] = (fpp[i, j] - fp[i] - fp[j] + f + f - fm[i] - fm[j] + fmm[i, j]) / hh[i, j] / 2
                H[j, i] = H[i, j]

        return H


    def vcov(self, params):
        """
        This method calculates the variance-covariance matrix with respect to parameter values params. To calculate the
        Jacobian, the function jac_func is used, the Hessian is calculated using finite differences.

        The vcov matrix Sigma is found by Sigma = inv(J)*I*inv(J), where J denotes the Hessian and I the dot product of
        the Jacobian.
        """
        if params is None:
            params, param_names = self._init_params()

        x = params
        n = len(x)

        # Define step sizes
        eps = 1e-05
        step = [i * eps for i in x]

        scores = np.zeros((self.T, n))

        for i in range(n):
            h = step[i]
            delta = np.zeros(n)
            delta[i] = h

            loglik, logliksplus = self.quasi_log_likelihood(x + delta, out=True)
            loglik, logliksminus = self.quasi_log_likelihood(x - delta, out=True)

            scores[:, i] = (logliksplus - logliksminus) / (2 * h)

        I = (scores.T @ scores) / self.T
        I = np.nan_to_num(I)
        J = self._hessian_2sided(x)

        Jinv = np.mat(np.linalg.inv(J))

        vcv = np.asarray((Jinv * np.mat(I) * Jinv) / self.T)
        vcv = np.asarray(vcv)

        return vcv

    # Methods used to set constraints and initiate parameter values
    def _calc_estimates(self, params):
        """
        This method calculates the estimates of the conditional volatility process and the residuals of the conditional
        mean process
        """
        # Compute sigma2 and epsilon for all the different types of exogenous volatility models
        if self.model_type == 'mean-x':
            model = tvArmaXapARCH(df=self.df,
                                  returns_col=self.returns_col,
                                  dep_cols=self.dep_cols,
                                  h=self.h,
                                  lag_arma=self.lags_arma,
                                  exog_cols=self.exog_cols,
                                  lag_exog=self.lags_exog,
                                  params=params)

            sigma2, et = model.conditional_volatility()

        elif self.model_type == 'x-garch':
            model = tvArmaApArchXGarch(df=self.df,
                                       returns_col=self.returns_col,
                                       dep_cols=self.dep_cols,
                                       h=self.h,
                                       lag_arma=self.lags_arma,
                                       exog_cols=self.exog_cols,
                                       lag_exog_level=self.lags_exog,
                                       params=params,
                                       xgarch_cols=self.x_garch_cols,
                                       lag_exog_sigma=self.x_garch_lags)

            sigma2, et = model.conditional_volatility()

        else:
            model = tvArmaApARCHX(df=self.df,
                                  returns_col=self.returns_col,
                                  dep_cols=self.dep_cols,
                                  h=self.h,
                                  lag_arma=self.lags_arma,
                                  exog_cols=self.exog_cols,
                                  lag_exog=self.lags_exog,
                                  params=params)

            sigma2, et = model.conditional_volatility()

        return sigma2, et


    def _set_bounds(self):
        """
        This method sets bounds and constraints that make sure the parameter estimates are consistent.
        """
        e = 1e-10
        if self.model_type == 'mean-x':
            # Instantiate bounds given the parameter constraints of the ARMA-X-apARCH(1,1) model
            bnds = ((-10, 10),)                                                 # bounds on intercept
            bnds += ((-(1 - e), 1 - e),) * np.sum(self.lags_arma)               # bounds on ARMA
            bnds += ((-10, 10),) * np.sum(self.lags_exog)                       # bounds on exog cols
            bnds += ((e, 2 * np.var(self.df[self.returns_col])),) * 2           # bounds on omega
            bnds += ((e, 1 - e),) * 2                                           # bounds on beta
            bnds += ((e, 10),) * 2                                              # bounds on alpha
            bnds += ((-(1 - e), 1 - e),) * 2                                    # bonds on phi

        elif self.model_type == 'x-garch':
            # Instantiate bounds given the parameter constraints of the ARMA-apARCH(1,1)-XGARCH model
            bnds = ((-10, 10),)                                                 # bounds on intercept
            bnds += ((-(1 - e), 1 - e),) * (np.sum(self.lags_arma))             # bounds on ARMA
            bnds += ((e, 2 * np.var(self.df[self.returns_col])),) * 2           # bounds on omega
            bnds += ((e, 1 - e), ) * 2                                          # bounds on beta
            bnds += ((e, 10), ) * 2                                             # bounds on alpha
            bnds += ((-(1 - e), 1 - e),) * 2                                    # bonds on phi
            bnds += ((e, 10),) * 2 * \
                    (np.sum(self.lags_exog) + np.sum(self.x_garch_lags))        # exogenous bounds
            bnds += ((-(1 - e), 1 - e),) * np.sum(self.lags_exog) * 2           # phi of exogenous level

        else:
            # Instantiate bounds given the parameter constraints of the ARMA-apARCH-X model
            bnds = ((-10, 10),)                                                 # intercept
            bnds += ((-(1 - e), 1 - e),) * (np.sum(self.lags_arma))             # bounds on ARMA params
            bnds += ((e, 2 * np.var(self.df[self.returns_col])),) * 2           # bounds on omega
            bnds += ((e, (1 - e)),) * 2                                         # bounds on beta
            bnds += ((e, 10),) * 2                                              # bounds on alpha
            bnds += ((-(1 - e), 1-e),) * 2                                      # bounds on phi
            bnds += ((-(1 - e), 1-e),) * (np.sum(self.lags_exog)) * 2           # bounds on phi_ik
            bnds += ((e, 10),) * 2 * (np.sum(self.lags_exog))                  # bounds on pi_ik

        params, param_names = self._init_params()

        # Define constraints, when alpha or beta occurs in the param_names, the sum of these params must be less than 1
        alpha_ind = [i for i, elem in enumerate(param_names) if 'alpha' in elem]
        beta_ind = [i for i, elem in enumerate(param_names) if 'beta' in elem]
        omega_ind = [i for i, elem in enumerate(param_names) if 'omega' in elem]

        alpha_ = [i for i, elem in enumerate(param_names) if elem == 'alpha']
        beta_ = [i for i, elem in enumerate(param_names) if elem == 'beta']
        phi_ind = [i for i, elem in enumerate(param_names) if elem == 'phi']
        phi_star_ind = [i for i, elem in enumerate(param_names) if elem == 'phi_star']
        phi_ik_ind = [i for i, elem in enumerate(param_names) if 'phi' in elem]
        pi_ik_ind = [i for i, elem in enumerate(param_names) if 'pi' in elem]

        # Define constraints
        cons1 = {'type': 'ineq', 'fun': lambda params: 0.99999 - (np.sum(params[alpha_ind]) *
                                                                 (1 + (params[phi_ind] + params[phi_star_ind]) ** 2)
                                                                 + np.sum(params[beta_ind]))}
        cons2 = {'type': 'ineq', 'fun': lambda params: 0.99999 - (params[alpha_] * (1 + params[phi_ind] ** 2)
                                                                 + params[beta_])}
        cons3 = {'type': 'ineq', 'fun': lambda params: 0.99999 - np.abs(np.sum(params[phi_ind]))}
        cons4 = {'type': 'ineq', 'fun': lambda params: 0.99999 - np.sum(params[beta_ind])}
        cons5 = {'type': 'ineq', 'fun': lambda params: np.sum(params[beta_ind]) - 0.0001}
        cons6 = {'type': 'ineq', 'fun': lambda params: np.sum(params[alpha_ind]) - 0.0001}
        cons7 = {'type': 'ineq', 'fun': lambda params: np.sum(params[omega_ind]) - 0.0001}
        cons8 = {'type': 'ineq', 'fun': lambda params: 0.99999 -
                                                       np.max([np.abs(params[phi_ik_ind[i]]) for i in range(0, len(phi_ik_ind), 2)])}
        cons9 = {'type': 'ineq', 'fun': lambda params: np.max([np.abs(params[pi_ik_ind[i]]) for i in range(0, len(pi_ik_ind), 2)])}

        # All combinations phi, phi_star and pi, pi_star that are present in the data
        chunks_phi_ik = [phi_ik_ind[i:i + 2] for i in range(0, len(phi_ik_ind), 2)]
        chunks_pi_ik = [pi_ik_ind[i:i + 2] for i in range(0, len(pi_ik_ind), 2)]

        # The biggest absolute value phi + phi_star must be smaller than 1, than all values are
        cons10 = {'type': 'ineq', 'fun': lambda params: 0.99999 - np.max([np.abs(np.sum(params[chunks_phi_ik[i]])) for i in
                                                                         range(len(chunks_phi_ik))])}

        # The smallest value of pi + pi_star must be greater than zero, then all values are
        cons11 = {'type': 'ineq', 'fun': lambda params: [np.sum(params[chunks_pi_ik[i]]) - 0.0001 for
                                                               i in range(len(chunks_pi_ik))]}

        # Define ARMA constraints
        gamma_ind = [i for i, elem in enumerate(param_names) if 'gamma' in elem]
        cons12 = {'type': 'ineq', 'fun': lambda params: 0.99999 - np.sum(params[gamma_ind])}

        if self.model_type != 'mean-x':
                cons = (cons1, cons2, cons3, cons4, cons5, cons6, cons7, cons8, cons9, cons10, cons11, cons12)
        else:
            cons = (cons1, cons2, cons3, cons4, cons5, cons6, cons7, cons10, cons12)

        return bnds, cons

    def _init_params(self):
        """
        This method initalizes parameters by making use of the parameter initialization from the garch_models.
        Furthermore, lists with the parameter names are created.
        """
        if self.model_type == 'mean-x':
            model = tvArmaXapARCH(df=self.df,
                                  returns_col=self.returns_col,
                                  dep_cols=self.dep_cols,
                                  lag_arma=self.lags_arma,
                                  exog_cols=self.exog_cols,
                                  lag_exog=self.lags_exog)
            params = model.params

            # Instantiate parameter names
            param_names = ['mu']
            param_names += [f'gamma_lag{i + 1}' for i in range(self.lags_arma[0])]
            param_names += [f'delta_lag{i + 1}' for i in range(self.lags_arma[1])]

            # Create list with exogenous variables repeated the number of lags
            exog_variables = [[f'{item}_lag{i + 1}' for i in range(lag)] for item, lag in
                              zip(self.exog_cols, self.lags_exog)]
            param_names += [f'pi_{item}' for sublist in exog_variables for item in sublist]
            param_names += ['omega', 'omega_star', 'beta', 'beta_star', 'alpha', 'alpha_star', 'phi', 'phi_star']

        elif self.model_type == 'x-garch':
            model = tvArmaApArchXGarch(df=self.df,
                                       returns_col=self.returns_col,
                                       dep_cols=self.dep_cols,
                                       lag_arma=self.lags_arma,
                                       exog_cols=self.exog_cols,
                                       lag_exog_level=self.lags_exog,
                                       xgarch_cols=self.x_garch_cols,
                                       lag_exog_sigma=self.x_garch_lags)
            params = model.params

            # Instantiate parameter names
            param_names = ['mu']
            param_names += [f'gamma_lag{i + 1}' for i in range(self.lags_arma[0])]
            param_names += [f'delta_lag{i + 1}' for i in range(self.lags_arma[1])]
            param_names += ['omega', 'omega_star', 'beta', 'beta_star', 'alpha', 'alpha_star', 'phi', 'phi_star']

            # Create parameter names for all the lagged exogenous regressors (both sigma and normal asymmetric)
            x_all = self.exog_cols + self.x_garch_cols
            x_lags_all = self.lags_exog + self.x_garch_lags
            all_variables = [[f'{item}_lag{i + 1}' for i in range(lag)] for item, lag in zip(x_all, x_lags_all)]
            asym_variables = [[f'{item}_lag{i + 1}' for i in range(lag)] for item, lag in
                              zip(self.exog_cols, self.lags_exog)]

            # Add unique parameter names to the list with parameter names
            exog_param_names = []
            exog_param_names += [[f'pi_{item}', f'pi_star_{item}'] for sublist in all_variables for item in sublist]
            exog_param_names += [[f'phi_{item}', f'phi_star_{item}'] for sublist in asym_variables for item in sublist]

            if len(self.exog_cols) > 0:
                param_names += [item for sublist in exog_param_names for item in sublist]

        else:
            model = tvArmaApARCHX(df=self.df,
                                  returns_col=self.returns_col,
                                  dep_cols=self.dep_cols,
                                  lag_arma=self.lags_arma,
                                  exog_cols=self.exog_cols,
                                  lag_exog=self.lags_exog)

            params = model.params

            # Instantiate parameter names
            param_names = ['mu']
            param_names += [f'gamma_lag{i + 1}' for i in range(self.lags_arma[0])]
            param_names += [f'delta_lag{i + 1}' for i in range(self.lags_arma[1])]
            param_names += ['omega', 'omega_star', 'beta', 'beta_star', 'alpha', 'alpha_star', 'phi', 'phi_star']

            # Create parameter names for all the lagged exogenous regressors
            variables = [[f'{item}_lag{i + 1}' for i in range(lag)] for item, lag in
                         zip(self.exog_cols, self.lags_exog)]

            # Add unique parameter names to the list with parameter names
            exog_param_names = []
            exog_param_names += [[f'phi_{item}', f'phi_star_{item}'] for sublist in variables for item in sublist]
            exog_param_names += [[f'pi_{item}', f'pi_star_{item}'] for sublist in variables for item in sublist]

            if len(self.exog_cols) > 0:
                param_names += [item for sublist in exog_param_names for item in sublist]

        return params, param_names
