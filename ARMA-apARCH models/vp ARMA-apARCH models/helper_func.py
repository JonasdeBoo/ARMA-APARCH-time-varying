import numpy as np

def regimeshift(df, x_cols, h, levels):
    """
    This function calculates the regime a variable exists in. This is done in pursuit of determining a two regime
    ARMA-apARCH-X model, where the parameters shift based on the regime they are in. This function is based on the function
    by Terasvirta (2008), and will be exploited similarly to Thomassen (2018).

    _____________________
    Parameters:
        - df: DataFrame from which x_cols are entries.
        - x_cols, the value on which this function will determine the regime
        - h, the degree of converge to either zero or 1.
        - levels, the baseline, more deviation of x from the level will result in values of 0 or 1.
    """
    if isinstance(x_cols, str):
        deviation = df[x_cols].values - levels
        G = (1 + np.exp(-h * deviation)) ** -1
    else:
        mean_dev = []
        for level, col in zip(levels, x_cols):
            deviation = df[col].values - level
            mean_dev += [deviation]
        G = (1 + np.exp(-h * np.mean(mean_dev, axis=0))) ** -1

    return G