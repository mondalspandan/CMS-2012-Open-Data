name = 'MuonSkim'
from CRABClient.UserUtilities import config, getUsernameFromSiteDB
import FWCore.Utilities.FileUtils as FileUtils

datasetdir = '/afs/cern.ch/work/s/spmondal/private/OpenData/CMSSW_5_3_32/src/MuonResonance/MuonEventsSkimmer/datasets/'
filelists2012=[ 'CMS_Run2012B_DoubleMuParked_AOD_22Jan2013-v1_10000_file_index.txt',
                'CMS_Run2012B_DoubleMuParked_AOD_22Jan2013-v1_20000_file_index.txt',
                'CMS_Run2012B_DoubleMuParked_AOD_22Jan2013-v1_20001_file_index.txt',
                'CMS_Run2012B_DoubleMuParked_AOD_22Jan2013-v1_20002_file_index.txt',
                'CMS_Run2012B_DoubleMuParked_AOD_22Jan2013-v1_210000_file_index.txt',
                'CMS_Run2012B_DoubleMuParked_AOD_22Jan2013-v1_30000_file_index.txt',
                'CMS_Run2012B_DoubleMuParked_AOD_22Jan2013-v1_310000_file_index.txt',
                'CMS_Run2012C_DoubleMuParked_AOD_22Jan2013-v1_10000_file_index.txt',
                'CMS_Run2012C_DoubleMuParked_AOD_22Jan2013-v1_10001_file_index.txt',
                'CMS_Run2012C_DoubleMuParked_AOD_22Jan2013-v1_10002_file_index.txt',
                'CMS_Run2012C_DoubleMuParked_AOD_22Jan2013-v1_10003_file_index.txt',
                'CMS_Run2012C_DoubleMuParked_AOD_22Jan2013-v1_10010_file_index.txt',
                'CMS_Run2012C_DoubleMuParked_AOD_22Jan2013-v1_10011_file_index.txt',
                'CMS_Run2012C_DoubleMuParked_AOD_22Jan2013-v1_10013_file_index.txt',
                'CMS_Run2012C_DoubleMuParked_AOD_22Jan2013-v1_10016_file_index.txt',
                'CMS_Run2012C_DoubleMuParked_AOD_22Jan2013-v1_10018_file_index.txt',
                'CMS_Run2012C_DoubleMuParked_AOD_22Jan2013-v1_10021_file_index.txt',
                'CMS_Run2012C_DoubleMuParked_AOD_22Jan2013-v1_10022_file_index.txt',
                'CMS_Run2012C_DoubleMuParked_AOD_22Jan2013-v1_10024_file_index.txt',
                'CMS_Run2012C_DoubleMuParked_AOD_22Jan2013-v1_20000_file_index.txt'
            ]
files2012data=[]
for filelist in filelists2012:
    files2012data += FileUtils.loadListFromFile(datasetdir+filelist)

config = config()
config.General.workArea = 'crab_'+name
config.General.transferOutputs = True
config.General.transferLogs = True
config.General.requestName = 'MuonSkim_2012opendata'

config.JobType.pluginName = 'Analysis'
config.JobType.psetName = '/afs/cern.ch/work/s/spmondal/private/OpenData/CMSSW_5_3_32/src/MuonResonance/MuonEventsSkimmer/muoneventsskimmer_cfg_2012data.py'
config.JobType.maxMemoryMB = 2400
#config.JobType.numCores = 4
config.JobType.outputFiles = ['DoubleMuSkimmedTree.root']

config.Data.splitting = 'FileBased'
config.Data.publication = False

config.Data.userInputFiles = files2012data
config.Data.unitsPerJob = 50
config.Data.outputPrimaryDataset = "Run2012_DoubleMuParked"
config.Data.outputDatasetTag = 'MuonSkim_2012opendata'
config.Data.outLFNDirBase = '/store/user/%s/t3store2/MuonSkim' % (getUsernameFromSiteDB())
config.Site.storageSite = 'T2_IN_TIFR'
