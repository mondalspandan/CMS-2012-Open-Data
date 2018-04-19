# Skimmer

The Skimmer accesses CMS 2012 open data and selects events with at least two oppositely charged muons and stores an event-based root tree in the output containing some limited muon, electron and jet information. It is written in the standard CMSSW EDAnalyzer format. It can be run either using the recommended [CMS VM](http://opendata.cern.ch/record/252) or through LXPLUS or other CERN computing resources with CMSSW support.

## Run using CMS VM
1. Install CMS Virtual Machine using Step 1 of the guide at [CMS 2011 Virtual Machines: How to install](http://opendata.cern.ch/docs/cms-virtual-machine-2011).
2. The next step is to set up a CMSSW environment and create a blank EDAnalyzer. Just replace `DemoAnalyzer` with `MuonEventsSkimmer` in the Step 2 of the guide at [CMS 2011 Virtual Machines: How to install](http://opendata.cern.ch/docs/cms-virtual-machine-2011) and follow the rest.
3. Replace the `BuildFile.xml` and `src/MuonEventsSkimmer.cc` files with the ones provided in this repository.
4. Delete the `muoneventsskimmer_cfg.py` file and copy the five `*_cfg_*.py` files from this repository.
5. Copy the `datasets` directory. Make sure the directory structure is intact after making all these changes.
6. Open each of the `.py` files and replace the line
    ```python
    datasetdir = '/afs/cern.ch/.../MuonEventsSkimmer/datasets/' #For LXPLUS
    ```
    with the path in the VM where the datasets are located.
7. Build using 
    ```shell
    scram b
    ```
8. Run on data using
    ```shell
    cmsRun muoneventsskimmer_cfg_2012data.py
    ```
9. Run on other datasets as needed. To run on a single dataset .root file which is known to contain a Higgs event, run on `muoneventsskimmer_cfg_data_with_Higgs_event.py`.

## Run on LXPLUS
(Using LXPLUS requires having a CERN account)
1. `ssh` to your LXPLUS account and navigate to a work directory.
2. Setup a CMSSW environment:
    ```shell
    cmsrel CMSSW_5_3_32
    cd CMSSW_5_3_32/src/
    cmsenv
    ```
3. Make a new empty EDAnalyzer:
    ```shell
    mkdir Dimuon
    cd Dimuon
    mkedanlzr MuonEventsSkimmer
    cd MuonEventsSkimmer
    ```
4. Replace the `BuildFile.xml` and `src/MuonEventsSkimmer.cc` files with the ones provided in this repository.
5. Delete the `muoneventsskimmer_cfg.py` file and copy the five `*_cfg_*.py` files from this repository.
6. Copy the `datasets` and `crab` directories. Make sure the directory structure is intact after making all these changes.
7. Open each of the `.py` files and replace the path in the line
    ```python
    datasetdir = '/afs/cern.ch/.../MuonEventsSkimmer/datasets/' #For LXPLUS
    ```
    with the absolute path of the LXPLUS directory where the datasets are located.
8. Build using 
    ```shell
    scram b
    ```
#### Run locally
9. Run on data using
    ```shell
    cmsRun muoneventsskimmer_cfg_2012data.py
    ```
10. Run on other datasets as needed. To run on a single dataset .root file which is known to contain a Higgs event, run on `muoneventsskimmer_cfg_data_with_Higgs_event.py`.
#### Run using CRAB3
(running jobs on CRAB requires a valid CMS grid proxy and a T2 storage account)
9. Open each of the `crabconfig_*.py` files in the `crab` directory and replace:
    * The path in the line `datasetdir = '...'` with the absolute path of the LXPLUS directory where the datasets are located.
    * The path in `config.JobType.psetName = '...'` with the correct path of the CMSSW configuration file.
    * `config.Site.storageSite = 'T2_IN_TIFR'` with the T2 storage site which you have write access to.
10. Initiate CRAB environment and grid proxy using:
    ```shell
    source /cvmfs/cms.cern.ch/crab3/crab.sh
    voms-proxy-init --voms cms --valid 168:00
    ```
11. Submit CRAB jobs for data using:
    ```shell
    crab submit -c crabconfig_2012data.py
    ```
12. Submit jobs for other MC samples using corresponding CRAB configuration file.
13. To monitor job status and retrieve CRAB job outputs, check [Running CMSSW code on the Grid using CRAB3](https://twiki.cern.ch/twiki/bin/view/CMSPublic/WorkBookCRAB3Tutorial).

## Outputs
Outputs are .root file(s) named `DoubleMuSkimmedTree.root` or `DoubleMuSkimmedTree_*.root`, depending on whether the programme was run locally or on CRAB. These outputs need to be copied to the respective directories inside the `Analysis` directory to be used as inputs by the analysis codes.