# Analysis
The main analysis routines are coded in three `.py` files in this directory.
1. `DiMuonAnalysis.py` reproduces the dimuon spectrum.
2. `HiggsAnalysis.py` applies the approximate selections used in the CMS 2012 Higgs Analysis. 
3. `HiggsAnalysisML.py` uses signal and background MC samples to train a Machine Learning network and perform an MVA-based analysis on the data.

Before running any of the analysis codes, copy the Skimmer outputs to the respective directories. The directories in the repository contain few outputs from the Skimmer already, but these are *extremely* small subsets of the full datasets, and are meant for checking purposes only; the analysis outputs may not have any significant results.
## Prerequisites
#### Python
The required Python version is 2.7.
#### ROOT
ROOT (CERN) is the most important requirement. Install ROOT 6.12.X using instructions from https://root.cern.ch/downloading-root. On MacOS with HomeBrew installed, one can run (note the seemingly conflicting flags, but only this seems to work, otherwise ROOT gets configured with Python 3):
```shell
brew install root --with-python@2 --without-python
```
#### Other Python packages
A few more Python packages are required. They can be installed using pip, with commands like 
```shell
sudo pip install packagename
```
**The packages are: numpy, matplotlib, scipy, sklearn, xgboost, pandas.**
(XGBoost requires sklearn to be installed, which in turn requires scipy.)
Please install other missing packages if importing fails at runtime.

## 1. Dimuon Analysis
Run the code using
```shell
python DiMuonAnalysis.py data/
```
The output will be named `DiMuon.root`.
## 2. Higgs Analysis
Run the code using
```shell
python HiggsAnalysis.py data/
```
The output will be named `Higgs.root`.
## 3. Higgs Analysis with ML
Run the code using
```shell
python HiggsAnalysisML.py
```
The output will be named `HiggsML.root`.

## Handling Outputs
The histograms contained in the .root output files can be accessed as follows:
```shell
root -l filename.root
```
Inside the ROOT interpreter (interactive shell), type:
```
TBrowser t
```
A new window will appear.
* Double-click filename.root inside the "ROOT Files" in the browser on the left. Double-click on any histogram to display.
* To set y-axis to log, right click on the canvas around the frame of the histogram and select SetLogy.
* To increase bin width when input data is scarce, right-click on the histogram bars and select Rebin and enter the factor.
* Plots can be exported, if needed, using the File -> Save/Save As option.

Close the TBrowser and type `.q` in the interactive shell to exit ROOT.