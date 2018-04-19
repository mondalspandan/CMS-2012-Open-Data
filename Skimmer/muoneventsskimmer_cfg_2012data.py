import FWCore.ParameterSet.Config as cms
import FWCore.PythonUtilities.LumiList as LumiList
import FWCore.ParameterSet.Types as CfgTypes
import FWCore.Utilities.FileUtils as FileUtils

process = cms.Process("MuonAn")
process.load("FWCore.MessageService.MessageLogger_cfi")
process.options = cms.untracked.PSet( wantSummary = cms.untracked.bool(True) )
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )               # -1 means process all events. Change this to a number like 10000 to run a test job

datasetdir = '/afs/cern.ch/work/s/spmondal/private/OpenData/CMSSW_5_3_32/src/MuonResonance/MuonEventsSkimmer/datasets/' #For LXPLUS

##JSON file
JSON2012 = datasetdir+'Cert_190456-208686_8TeV_22Jan2013ReReco_Collisions12_JSON.txt'

myLumis = LumiList.LumiList(filename = JSON2012).getCMSSWString().split(',')        # List of all lumisections that are verified as good by CMS

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
    files2012data += FileUtils.loadListFromFile(datasetdir+filelist)        # Keep accumulating all files from each of the filelists


print "Will process %d AOD files." %len(files2012data)  
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(*files2012data    
    )
)

# apply JSON file
process.source.lumisToProcess = CfgTypes.untracked(CfgTypes.VLuminosityBlockRange())
process.source.lumisToProcess.extend(myLumis)

process.MuonAn = cms.EDAnalyzer('MuonEventsSkimmer'
)

process.TFileService = cms.Service("TFileService",
       fileName = cms.string('DoubleMuSkimmedTree.root')
)

process.p = cms.Path(process.MuonAn)
