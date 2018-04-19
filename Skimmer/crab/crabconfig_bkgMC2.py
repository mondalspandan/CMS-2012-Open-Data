name = 'MuonSkim'
from CRABClient.UserUtilities import config, getUsernameFromSiteDB
import FWCore.Utilities.FileUtils as FileUtils

datasetdir = '/afs/cern.ch/work/s/spmondal/private/OpenData/CMSSW_5_3_32/src/MuonResonance/MuonEventsSkimmer/datasets/'
filelistMC="CMS_MonteCarlo2012_Summer12_DR53X_ZZTo4mu_8TeV-powheg-pythia6_AODSIM_PU_RD1_START53_V7N-v1_20000_file_index.txt"
filesMC = FileUtils.loadListFromFile(datasetdir+filelistMC)

config = config()
config.General.workArea = 'crab_'+name
config.General.transferOutputs = True
config.General.transferLogs = True
config.General.requestName = 'MuonSkim_2012_ZZTo4muMC'

config.JobType.pluginName = 'Analysis'
config.JobType.psetName = '/afs/cern.ch/work/s/spmondal/private/OpenData/CMSSW_5_3_32/src/MuonResonance/MuonEventsSkimmer/muoneventsskimmer_cfg_MC_ZZTo4mu.py'
config.JobType.maxMemoryMB = 2400
#config.JobType.numCores = 4
config.JobType.outputFiles = ['DoubleMuSkimmedTree.root']

config.Data.splitting = 'FileBased'
config.Data.publication = False

config.Data.userInputFiles = filesMC
config.Data.unitsPerJob = 10
config.Data.outputPrimaryDataset = "Run2012_ZZTo4mu"
config.Data.outputDatasetTag = 'MuonSkim_2012_ZZTo4muMC'
config.Data.outLFNDirBase = '/store/user/%s/t3store2/MuonSkim' % (getUsernameFromSiteDB())
config.Site.storageSite = 'T2_IN_TIFR'
