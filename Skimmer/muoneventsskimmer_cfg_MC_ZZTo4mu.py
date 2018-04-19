import FWCore.ParameterSet.Config as cms
import FWCore.PythonUtilities.LumiList as LumiList
import FWCore.ParameterSet.Types as CfgTypes
import FWCore.Utilities.FileUtils as FileUtils

process = cms.Process("MuonAn")
process.load("FWCore.MessageService.MessageLogger_cfi")
process.options = cms.untracked.PSet( wantSummary = cms.untracked.bool(True) )
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )

datasetdir = '/afs/cern.ch/work/s/spmondal/private/OpenData/CMSSW_5_3_32/src/MuonResonance/MuonEventsSkimmer/datasets/' #For LXPLUS

##JSON file
#JSON2012 = datasetdir+'Cert_190456-208686_8TeV_22Jan2013ReReco_Collisions12_JSON.txt'

#myLumis = LumiList.LumiList(filename = JSON2012).getCMSSWString().split(',')

MClist=['CMS_MonteCarlo2012_Summer12_DR53X_ZZTo4mu_8TeV-powheg-pythia6_AODSIM_PU_RD1_START53_V7N-v1_20000_file_index.txt'
        ]
rootfilelist=[]
for filelist in MClist:
    rootfilelist += FileUtils.loadListFromFile(datasetdir+filelist)


print "Will process %d AOD files." %len(rootfilelist)
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(*rootfilelist    
    )
)

# apply JSON file
#process.source.lumisToProcess = CfgTypes.untracked(CfgTypes.VLuminosityBlockRange())
#process.source.lumisToProcess.extend(myLumis)

process.MuonAn = cms.EDAnalyzer('MuonEventsSkimmer'
)

process.TFileService = cms.Service("TFileService",
       fileName = cms.string('DoubleMuSkimmedTree.root')
)

process.p = cms.Path(process.MuonAn)
