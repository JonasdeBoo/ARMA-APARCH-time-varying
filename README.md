## ThesisCode

This code will perform all calculations for the MSc Thesis of Jonas de Boo (u1272278). The goal of this research is to analyse non-financial Twitter sentiment of 15 companies.
The file is structured as follows. 
- Code can be found in folder Code, all file names indicate the specific coding step of the research progress, for each step in the process another folder is created (e.g. data collection, 
sentiment analysis, checking input variables and plotting data and constructing the (ARMA-)GARCH(-X) models).
  - Each step makes use of .py files and JupyterNotebooks. The .py files are used to create functions or classes,
    which are used per unique company in the JupyterNotebooks to neatly display results and plots. 
    In each Jupyter Notebook calculations are done to retrieve DataFrames, figures or regression results. These are directly stored in seperate folders, which can in turn be     accessed for the next step in the total calculation process. In all the plots a specific color code for layout purposes is used. This consists of the following colors:
    ['seagreen', 'mediumaquamarine', 'steelblue', 'cornflowerblue', 'navy', 'black']
    
- Data from each step can be found in the folder Data, each file here containes data. As with the code, data folders are created for every single
step of the coding process.
