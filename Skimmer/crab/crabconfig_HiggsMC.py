name = 'MuonSkim'
from CRABClient.UserUtilities import config, getUsernameFromSiteDB
import FWCore.Utilities.FileUtils as FileUtils

datasetdir = '/afs/cern.ch/work/s/spmondal/private/OpenData/CMSSW_5_3_32/src/MuonResonance/MuonEventsSkimmer/datasets/'
filelistMC_Higgs="CMS_MonteCarlo2012_Summer12_DR53X_SMHiggsToZZTo4L_M-125_8TeV-powheg15-JHUgenV3-pythia6_AODSIM_PU_S10_START53_V19-v1_10000_file_index.txt"
filesMC_Higgs = FileUtils.loadListFromFile(datasetdir+filelistMC_Higgs)

config = config()
config.General.workArea = 'crab_'+name
config.General.transferOutputs = True
config.General.transferLogs = True
config.General.requestName = 'MuonSkim_2012HiggsMC'

config.JobType.pluginName = 'Analysis'
config.JobType.psetName = '/afs/cern.ch/work/s/spmondal/private/OpenData/CMSSW_5_3_32/src/MuonResonance/MuonEventsSkimmer/muoneventsskimmer_cfg_Higgs.py'
config.JobType.maxMemoryMB = 2400
#config.JobType.numCores = 4
config.JobType.outputFiles = ['DoubleMuSkimmedTree.root']

config.Data.splitting = 'FileBased'
config.Data.publication = False

config.Data.userInputFiles = filesMC_Higgs
config.Data.unitsPerJob = 10
config.Data.outputPrimaryDataset = "Run2012_DoubleMuParked"
config.Data.outputDatasetTag = 'MuonSkim_2012HiggsMC'
config.Data.outLFNDirBase = '/store/user/%s/t3store2/MuonSkim' % (getUsernameFromSiteDB())
config.Site.storageSite = 'T2_IN_TIFR'
