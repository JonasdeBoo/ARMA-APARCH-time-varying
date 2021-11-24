## ThesisCode

This code will perform all calculations for the MSc Thesis of Jonas de Boo (u1272278). The goal of this research is to analyse the effect of non-financial Twitter sentiment of 12 U.S. companies on their respective stock price volatility.
The files is structured as follows. 
- Code can be found in folder Code, all file names indicate the specific coding step of the research progress, for each step in the process another folder is created (e.g. data collection, sentiment analysis, checking input variables and plotting data and constructing the (ARMA-)GARCH(-X) models).
- The folder GARCH process contains a subfolder where the vp-GARCH models are modeled
- In notebooks there are number Jupyter Notebooks, these notebooks use the constructed classes to calculate all relevent results. Calculations are done to retrieve DataFrames, figures or regression results. In all the plots a specific color code for layout purposes is used. This consists of the following colors:
    ['seagreen', 'mediumaquamarine', 'steelblue', 'cornflowerblue', 'navy', 'black']

