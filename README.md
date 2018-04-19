# Analyses using CMS 2012 Open Data

The aim of the project is to make use of CMS 2012 open data and perform the following analyses:
1. Reproduction of the Dimuon spectrum.
2. A very simplified attempt to approximately apply the selections used in the CMS Higgs analysis in 2012 in the 4-lepton channel, using only muon final states.
3. An MVA-based analysis of the Higgs to 4-lepton decay with muons in the final state.

All analyses are performed in two steps:
1. **Skimmer:** The skimming step accesses CMS open data and selects events with at least two oppositely charged muons and produces an event-based root tree in the output with only relevant information. It is written in C++ in the usual CMSSW EDAnalyzer format, and is common for all following analyses.
2. **Analysis:** There are three analysis files written in Python. Each of these does one of the tasks described above.

***Instructions for running each of the steps is appended in the respective directories.***

The general workflow is summarized in the following figure:

![alt text](https://raw.githubusercontent.com/mondalspandan/CMS-2012-Open-Data/master/workflow.png)