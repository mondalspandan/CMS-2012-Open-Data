#!/usr/bin/env python
from ROOT import *
import ROOT
import os,sys
from math import log
import numpy as np, xgboost, pandas as pd, random, matplotlib.pyplot as plt

outfilename= "HiggsML.root"
ZMass=91.

trainFrac=0.8           # Set the fraction of the signal and background MC samples that will be used for training the ML network


#============================================================================
# Define output histograms
#============================================================================
higgsmin,higgsmax,nhiggsbins=40.,238.,66

h_4mumass = TH1F('h_4mumass','h_4mumass',nhiggsbins,higgsmin,higgsmax)
h_4mumass.GetXaxis().SetTitle("4-Muon invariant mass (in GeV/c)")
h_4mumass.GetYaxis().SetTitle("Number of events")

h_2mu2emass = TH1F('h_2mu2emass','h_2mu2emass',nhiggsbins,higgsmin,higgsmax)
h_2mu2emass.GetXaxis().SetTitle("2-Muon, 2-Electron invariant mass (in GeV/c)")
h_2mu2emass.GetYaxis().SetTitle("Number of events")

h_higgsmass = TH1F('h_higgsmass','h_higgsmass',nhiggsbins,higgsmin,higgsmax)
h_higgsmass.GetXaxis().SetTitle("Candidate Higgs invariant mass (in GeV/c)")
h_higgsmass.GetYaxis().SetTitle("Number of events")

h_Zmass = TH1F('h_Zmass','h_Zmass',40,70.,110.)
h_Zmass.GetXaxis().SetTitle("Candidate Z invariant mass (in GeV/c) from Z -> #mu#mu/ee")
h_Zmass.GetYaxis().SetTitle("Number of events")

h_skimmed = TH1F('h_skimmed','h_skimmed',2,0.,2)
h_skimmed.GetYaxis().SetTitle("Number of events after skimming")
#============================================================================

def getpT(P4):              # A function to use as "key" in the "sorted()" command
    return P4.pt()


#============================================================================
# The main event-wise analysis routine, defined inside a function
#============================================================================
def AnalyzeEvent(ttree,ievent,isSignal,dirname="MCHiggs"):
    '''
    This function is called on a per-event basis. 
    
    Inputs:
    ttree: An event-based ROOT tree of type ROOT.TChain; contains all events.
    ievent: The event number to be analyzed, int.
    isSignal: Boolean. True for signal, false for background. Irrelevant for validation/data samples.
    dirname: To distinguish between samples.
    
    Returns:
    0: numpy.array of quantities described later, or numpy.nan.
    1: "4mu" or "2mu2e" or "none" depending on what the event is classified as.
    '''
    
    ttree.GetEntry(ievent)
        
    for attrib in ['muP4','nMu','muIso','muCharge','muSIP3D','nEle','eleP4','eleIso','eleCharge','eleSIP3D','nJet','jetP4']:
        exec(attrib+" = ttree.__getattr__('"+attrib+"')")           # Load each of the necessary values stored in the tree in a variable of the same name
    
    #------------------------------------------------------------------------
    # Electron redefinition
    #------------------------------------------------------------------------  
    '''
    We defined electrons with pT > 5 to be good electrons in the skimmer code.
    However, I later found that the CMS Higgs analysis used e pT > 7 GeV to define good
    electrons. To avoid rerunning the skimmer again, we run over the selected electrons
    in an event and keep only the ones above 7 GeV. All parallel lists (charge) must
    also be modified accordingly.
    '''     
    nEleold=nEle
    eleP4old=eleP4[:]
    eleChargeold=eleCharge[:]
    eleIsoold=eleIso[:]
    nEle=0
    eleP4=[]
    eleCharge=[]
    eleIso=[]
    for ie in range(nEleold):
        if eleP4old[ie].pt()>7.:
            eleP4.append(eleP4old[ie])
            eleCharge.append(eleChargeold[ie])
            eleIso.append(eleIsoold[ie])
            nEle+=1
    #------------------------------------------------------------------------ 
        
    muP4=list(muP4)                 # Conversion from ROOT.vector to python list
    eleP4=list(eleP4)
    jetP4=list(jetP4)    
    
    #------------------------------------------------------------------------
    # H -> ZZ -> 4mu
    #------------------------------------------------------------------------    
    
    if nMu>=3 and nEle<=1 and (dirname=="MCbkgZto4mu" or dirname=="MCHiggs"):       # nMu >=3 as I wanted to look at a more inclusive selection to increase signal efficiency. But this worsens the training, so later I dropped the idea. nMu==3 case is effectively excluded in the next if condition, and we are effectively looking at nMu>=4.
    
    # We specify folder name as we distinctly want to train with Zto4mu sample as background in nMu>=3 category.
        
        muPlus=[]
        muMinus=[]
        for imu in range(nMu):
            if muCharge[imu]>0:
                muPlus.append(muP4[imu])                # Make two different lists of muon 4-vectors (LorentzVector class in ROOT), one for mu+ and one for mu-.
            else:
                muMinus.append(muP4[imu])
        
        #*********** >=2 muons and >=2 electrons ***************
        
        if len(muPlus)>=2 and len(muMinus)>=2:          # Require at least two mu+ and at least two mu-
            #print
            #print nMu    
            
            ZP4temp=[]
            Zmus=[]
            for muP in muPlus:
                for muN in muMinus:
                    ZP4temp.append(muP+muN)             # A list to store all possible combinations of mu+ and mu-
                    Zmus.append([muP,muN])              # A parallel list to identify the muons that went into each dimuon sum
            
            #print [Z.M() for Z in ZP4temp]
            
            massdiff=[abs(Z.M()-ZMass) for Z in ZP4temp]                        # A list containing |m_ll - m_Z| values, parallel to the dimuon list
            val, idx = min((val, idx) for (idx, val) in enumerate(massdiff))    # This steps stores the list index of the dimuon candidate that has the lowest value of |m_ll - m_Z| inside the variable idx. "val" contains min(|m_ll - m_Z|), but is not used.
            
            Z1=ZP4temp[idx]                             # Call the best dimuon combination Z1.
            
            # Note: Unlike in cut-and-count based analysis, we are not requiring a mass cut for Z1 here! We are only identifying the particles.
                
            mu1=Zmus[idx][0]                            # Store info of the muons that comprise Z1 as mu1 and mu2
            mu2=Zmus[idx][1]
            muPlus.remove(mu1)
            muMinus.remove(mu2)                         # Remove the used up mu+ and mu- from our muon collection
            
            ZP4temp=[]
            Zmus=[]
            for muP in muPlus:
                for muN in muMinus:
                    ZP4temp.append(muP+muN)             # Repeat the same process to find the next best dimuon combination
                    Zmus.append([muP,muN])
            
            massdiff=[abs(Z.M()-ZMass) for Z in ZP4temp]
            val, idx = min((val, idx) for (idx, val) in enumerate(massdiff))
            
            Z2=ZP4temp[idx]                             # Store the LorentzVector of the second dimuon candidate with mass closest to Z_mass in Z2
            
            mu3=Zmus[idx][0]                            # Store info of the muons that comprise Z2 as mu3 and mu4
            mu4=Zmus[idx][1]
                
            DiZMass=(Z1+Z2).M()                         # Find diZ (candidate Higgs) mass
            
            if nEle==0:
                leadept=np.nan
                leadeeta=np.nan
            else:
                leade=sorted(eleP4, key=getpT, reverse=True)[0]         # First element of a reverse-sorted electron 4-vector list, sorted by its pT
                leadept=leade.pt()                                      # Hence this gives the electron with highest pT (leading electron)
                leadeeta=leade.eta()                                    # We also store the lead electron eta
            
            if nJet==0:
                leadjetpt=np.nan
                leadjeteta=np.nan
            else:
                leadjet=sorted(jetP4, key=getpT, reverse=True)[0]       # First element of a reverse-sorted jet 4-vector list, sorted by its pT. We store pT and eta of lead jet
                leadjetpt=leadjet.pt()
                leadjeteta=leadjet.eta()
            
            dtrow=[nMu,mu1.pt(),mu2.pt(),mu3.pt(),mu4.pt(),mu1.eta(),mu2.eta(),mu3.eta(),mu4.eta(),                 # nMu, Mu pT and eta
                    muIso[muP4.index(mu1)],muIso[muP4.index(mu2)],muIso[muP4.index(mu3)],muIso[muP4.index(mu4)],    # Mu isolations. We use the list.index() method to find the corresponding element from the muIso list
                    muSIP3D[muP4.index(mu1)],muSIP3D[muP4.index(mu2)],muSIP3D[muP4.index(mu3)],muSIP3D[muP4.index(mu4)], # Mu 3D IP significance
                    nEle, leadept, leadeeta,                                                                        # Lead electron info
                    nJet, leadjetpt, leadjeteta,                                                                    # Lead jet info
                    Z1.M(),Z2.M(),Z1.pt(),Z2.pt(),Z1.eta(),Z2.eta(),                                                # Reconstructed Z info
                    DiZMass, isSignal, int(random.random()<trainFrac), 1.                                           # Other info, not used in training, but used to identify signal vs. background, training vs. testing datasets. All weights are set to 1. Whether training or testing, is decided using a random number generator.
                  ]
                    
            # We make up a row of all the above raw and reconstructed quantities and return this array from this function
                    
            return np.array(dtrow), "4mu"
            
            
        #*********** (==2 Muons and ==1 electron) or (==1 Muon1 and ==2 electrons) ***************
        # Introduced this section to look at a more inclusive selection to recover signal efficiency. But this worsens the training, so later I dropped the idea.
        
        #elif (len(muPlus)>=2 and len(muMinus)==1) or (len(muPlus)==1 and len(muMinus)>=2):
            #ZP4temp=[]
            #Zmus=[]
            #for muP in muPlus:
                #for muN in muMinus:
                    #ZP4temp.append(muP+muN)        
                    #Zmus.append([muP,muN])
            
            ##print [Z.M() for Z in ZP4temp]
            
            #massdiff=[abs(Z.M()-ZMass) for Z in ZP4temp]
            #val, idx = min((val, idx) for (idx, val) in enumerate(massdiff))
            #Z1=ZP4temp[idx]
            #mu1=Zmus[idx][0]
            #mu2=Zmus[idx][1]
            #muPlus.remove(mu1)
            #muMinus.remove(mu2)
            #mu3 = sorted(muPlus+muMinus, key=getpT, reverse=True)[0]
            
            #if nEle==0:
                #leadept=np.nan
                #leadeeta=np.nan
            #else:
                #leade=sorted(eleP4, key=getpT, reverse=True)[0]
                #leadept=leade.pt()
                #leadeeta=leade.eta()
            
            #if nJet==0:
                #leadjetpt=np.nan
                #leadjeteta=np.nan
            #else:
                #leadjet=sorted(jetP4, key=getpT, reverse=True)[0]
                #leadjetpt=leadjet.pt()
                #leadjeteta=leadjet.eta()
            
            #dtrow=[nMu,mu1.pt(),mu2.pt(),mu3.pt(),np.nan,mu1.eta(),mu2.eta(),mu3.eta(),np.nan,
                    #muIso[muP4.index(mu1)],muIso[muP4.index(mu2)],muIso[muP4.index(mu3)],np.nan,
                    #muSIP3D[muP4.index(mu1)],muSIP3D[muP4.index(mu2)],muSIP3D[muP4.index(mu3)],np.nan,
                    #nEle, leadept, leadeeta,
                    #nJet, leadjetpt, leadjeteta,
                    #Z1.M(),np.nan,Z1.pt(),np.nan,Z1.eta(),np.nan,
                    #np.nan, isSignal, int(random.random()<trainFrac), 1.
                    #]
            #return np.array(dtrow), "4mu"
    
    #------------------------------------------------------------------------ 
    
    
    #------------------------------------------------------------------------
    # H -> ZZ -> 2mu + 2e
    #------------------------------------------------------------------------   
            
    if nMu>=2 and nEle>=2 and (dirname=="MCbkgZto2mu2e" or dirname=="MCHiggs"):
        muPlus=[]
        muMinus=[]
        for imu in range(nMu):
            if muCharge[imu]>0:
                muPlus.append(muP4[imu])                # Make two different lists of muon 4-vectors, one for mu+ and one for mu-.
            else:
                muMinus.append(muP4[imu])
                
        elePlus=[]
        eleMinus=[]
        for iele in range(nEle):
            if eleCharge[iele]>0:
                elePlus.append(eleP4[iele])             # Make two different lists of electron 4-vectors, one for e+ and one for e-.
            else:
                eleMinus.append(eleP4[iele])
        
        if len(muPlus)>=1 and len(muMinus)>=1 and len(elePlus)>=1 and len(eleMinus)>=1:     # Require at least one each of mu+, mu-, e+, e- in the event.
        
            ZP4temp=[]
            whichLep=[]
            Zleps=[]
            for muP in muPlus:
                for muN in muMinus:
                    ZP4temp.append(muP+muN)             # Fill dimuon combinations in a list
                    Zleps.append([muP,muN])             # Fill corresponding leptons
                    whichLep.append("mu")               # A parallel list stating that this dilepton combination is a dimuon comb.
            for eleP in elePlus:
                for eleN in eleMinus:
                    ZP4temp.append(eleP+eleN)           # Fill dielectron combinations in the SAME list
                    Zleps.append([eleP,eleN])           # Fill corresponding leptons
                    whichLep.append("e")                # A parallel list stating that this dilepton combination is a dielectron comb.
                    
            massdiff=[abs(Z.M()-ZMass) for Z in ZP4temp]                        # Store |m_ll - m_Z| values in a parallel list
            val, idx = min((val, idx) for (idx, val) in enumerate(massdiff))    # This steps stores the list index of the dilepton candidate that has the lowest value of |m_ll - m_Z| inside the variable idx
            
            Z1=ZP4temp[idx]                             # Best combination is stored as Z1
            
            lead=whichLep[idx]
            if lead=="mu":
                lepPlus=elePlus
                lepMinus=eleMinus                       # If Z1 is a dimuon combination, store electron values in lepPlus and lepMinus
                mu1=Zleps[idx][0]
                mu2=Zleps[idx][1]
            else:
                lepPlus=muPlus
                lepMinus=muMinus                        # If Z1 is a dielectron combination, store muon values in lepPlus and lepMinus
                ele1=Zleps[idx][0]
                ele2=Zleps[idx][1]
            
            ZP4temp=[]
            Zleps=[]
            for lepP in lepPlus:
                for lepN in lepMinus:
                    ZP4temp.append(lepP+lepN)           
                    Zleps.append([lepP,lepN])           # Now we repeat the same process with lepPlus and lepMinus, which now contain info of the lepton opposite in flavour to the Z1 constituents
            
            massdiff=[abs(Z.M()-ZMass) for Z in ZP4temp]
            val, idx = min((val, idx) for (idx, val) in enumerate(massdiff))        # Same way to find the best dilepton candidate among the other leptons
            Z2=ZP4temp[idx]  
            if lead=="mu":
                ele1=Zleps[idx][0]                      # If Z1 was made of muons, these leptons are electrons
                ele2=Zleps[idx][1]
            else:
                mu1=Zleps[idx][0]
                mu2=Zleps[idx][1]
            
            DiZMass=(Z1+Z2).M()
            
            if nJet==0:
                leadjetpt=np.nan
                leadjeteta=np.nan
            else:
                leadjet=sorted(jetP4, key=getpT, reverse=True)[0]           # First element of a reverse-sorted jet 4-vector list, sorted by its pT. We store pT and eta of lead jet
                leadjetpt=leadjet.pt()
                leadjeteta=leadjet.eta()
                
#            print [imu.pt() for imu in muP4]
#            print [iele.pt() for iele in eleP4]
#            print mu1.pt()
#            print

            dtrow=[nMu,mu1.pt(),mu2.pt(),mu1.eta(),mu2.eta(),               # nMu, Mu pT and eta
                    muIso[muP4.index(mu1)],muIso[muP4.index(mu2)],          # Muon isolations
                    muSIP3D[muP4.index(mu1)],muSIP3D[muP4.index(mu2)],      # Muon 3D IP significances
                    nEle, ele1.pt(),ele2.pt(),ele1.eta(),ele2.eta(),        # nEle, Ele pT and eta
                    eleIso[eleP4.index(ele1)],eleIso[eleP4.index(ele2)],    # Ele isolations
                    eleSIP3D[eleP4.index(ele1)],eleSIP3D[eleP4.index(ele2)],# Ele 3D IP significances
                    nJet, leadjetpt, leadjeteta,                            # Lead jet info
                    Z1.M(),Z2.M(),Z1.pt(),Z2.pt(),Z1.eta(),Z2.eta(),        # Reconstructed Z candidate info
                    DiZMass, isSignal, int(random.random()<trainFrac), 1.   # Other info, not used in training, but used to identify signal vs. background, training vs. testing datasets. All weights are set to 1. Whether training or testing, is decided using a random number generator.
                  ]
            return np.array(dtrow), "2mu2e"
        #------------------------------------------------------------------------ 
        
    return np.nan, "none"   # When no category requirement is satisfied

#============================================================================


#============================================================================
# Making dataframes with each row corresponding to a event
#============================================================================    
'''
The idea is two make two pandas dataframes, one for 4mu and one for 2mu2e, to train and validate
an ML network. For this we generate one row per event containing certain information, and add this
row to one of the dataframes (depending on the category of the event). This is done for signal and 
background MC events. Information for each row is derived using the analysis function above.
'''

matrix4mu=[]
matrix2mu2e=[]  

for dirname in ["MCHiggs","MCbkgZto4mu","MCbkgZto2mu2e"]:       # Directory names are hardcoded. Corresponding skimmed root files need to be placed inside the directories.

    #------------------------------------------------------------------------
    # Load corresponding files for each directory
    #------------------------------------------------------------------------
    
    ttree = TChain("MuonAn/MuonEvents")

    rootfilelist=[fl for fl in os.listdir(dirname) if fl.endswith(".root")] 
    for rootfile in rootfilelist:
        ttree.Add(dirname+"/"+rootfile)                         # Adds all root files to the TChain
    print "Reading %d file(s) from %s."%(len(rootfilelist),dirname)
        
    NEntries = ttree.GetEntries()

    if dirname=="MCHiggs":      # MCHiggs directory contains Higgs MC events, which is our signal
        isSignal=1
    else:
        isSignal=0

    if len(sys.argv)>1:
        #if sys.argv[2]=="test":
        NEntries=min(NEntries,int(sys.argv[1]))     # Test mode with limited events
        print "WARNING: Running in TEST MODE"

    print 'NEntries = '+str(NEntries)
    #------------------------------------------------------------------------
           
           
    #------------------------------------------------------------------------
    # Start event loop
    #------------------------------------------------------------------------
    for ievent in range(NEntries):
        if ievent%10000==0: print "Processed %d of %d events: ~%d%%" %(ievent,NEntries,int(ievent*100/NEntries))
        
        row,reg=AnalyzeEvent(ttree,ievent,isSignal,dirname)     # Call the Analyze function to get the row information
        
        if reg=="4mu":                                          # Fill the row to the relevant matrix depending on how the Analyze function categorized the event
            matrix4mu.append(row)
        elif reg=="2mu2e":
            matrix2mu2e.append(row)
    #------------------------------------------------------------------------

#    print len(matrix4mu),len(matrix2mu2e)

# Now we convert each of the two matrices (python lists) to pandas dataframes for easier data handling.
# We use column names as follows.

col4mu=['nMu','mu1.pt','mu2.pt','mu3.pt','mu4.pt','mu1.eta','mu2.eta','mu3.eta','mu4.eta',
        'mu1.iso','mu2.iso','mu3.iso','mu4.iso',
        'mu1.3Dip','mu2.3Dip','mu3.3Dip','mu4.3Dip',
        'nEle', 'lead.e.pt', 'lead.e.eta',
        'nJet', 'lead.jet.pt', 'lead.jet.eta',
        'Z1.M','Z2.M','Z1.pt','Z2.pt','Z1.eta','Z2.eta',
        'DiZMass','isSignal','isTraining','weight'
        ]
dtframe4mu=pd.DataFrame(data=np.array(matrix4mu),columns=col4mu) 

col2mu2e=['nMu','mu1.pt','mu2.pt','mu1.eta','mu2.eta',
        'mu1.iso','mu2.iso',
        'mu1.3Dip','mu2.3Dip',
        'nEle','ele1.pt','ele2.pt','ele1.eta','ele2.eta',
        'ele1.iso','ele2.iso',
        'ele1.3Dip','ele2.3Dip',
        'nJet', 'lead.jet.pt', 'lead.jet.eta',
        'Z1.M','Z2.M','Z1.pt','Z2.pt','Z1.eta','Z2.eta',
        'DiZMass','isSignal','isTraining','weight'
        ]
dtframe2mu2e=pd.DataFrame(data=np.array(matrix2mu2e),columns=col2mu2e) 


# We shuffle the rows of the dataframes so that they are not arranged in a organized way.

def shuffledf(df):
    return df.sample(frac=1).reset_index(drop=True)

dtframe4mu=shuffledf(dtframe4mu)
dtframe2mu2e=shuffledf(dtframe2mu2e)

print dtframe4mu.shape
print dtframe2mu2e.shape

#print dtframe4mu.head()
#print
#print dtframe2mu2e.head()
#print dtframe4mu.tail()
#print
#print dtframe2mu2e.tail()
#============================================================================


#============================================================================
# Define what all variables we want to train on
#============================================================================
'''
This variable named XtraVars is a very important. In the rest of the code, the set of variables which are
used for training the network is decided by this. Basically, I am using this to define how many variables
to exclude from the right end of each row. By default, we are not supposed to use 'isSignal', 'isTraining'
'weight' variables to train, as they don't have any event information at all. In addition, we might wish
to not use DiZMass, or any of the reconstructed Z information. Accordingly, this variable could be altered
to include only the intended variables. This is a simple way to control the workflow.

- To train with only raw information (mu, ele and jet info), use XtraVars=10
- To train with raw information + reconstructed Z information, use XtraVars=4
- To train with raw information + reconstructed Z information + reconstructed diZ information, use XtraVars=3
'''

XtraVars=4

#============================================================================



#============================================================================
# Training with XGBoost
#============================================================================

def train(LR,dtframe,col,isVerbose=False):
    '''
    This is the function that runs the training with XGBoost. We use a binary classifier
    and train it to learn 1 whenever the isSignal of the row is set to 1, and learn 0
    whenever isSignal!=1.
    
    Inputs:    
    LR: Learning rate (hyperparameter of XGBoost classifier), float
    dtframe: pandas dataframe
    col: columns of the dataframe to train on
    isVerbose: boolean, set to true to see training progress and step-wise error
    
    Returns:
    The trained binary classifier
    '''
    
    print "Training on: ",col
    print "Learning rate =",LR
    cls=xgboost.XGBClassifier(n_estimators=1000, nthread=32, learning_rate=LR)

    evalset=[
           ( dtframe.ix[dtframe["isTraining"]==1, col],
             dtframe.ix[dtframe["isTraining"]==1, "isSignal"]==1,
             dtframe.ix[dtframe["isTraining"]==1, "weight"]
           ), #Training set
           
           
           ( dtframe.ix[dtframe["isTraining"]==0, col],
             dtframe.ix[dtframe["isTraining"]==0, "isSignal"]==1,
             dtframe.ix[dtframe["isTraining"]==0, "weight"]
           ) #Testing set
        ]

    cls.fit(evalset[0][0], evalset[0][1], evalset[0][2], eval_set=evalset, early_stopping_rounds=100, eval_metric=["error"], verbose=isVerbose)
    return cls
    
    '''
    The list evalset is constructed in a particular way.
    It has two tuples:
        * First tuple corresponds to training set and has three elements:
            - Set of relevant values of all events whose isTraining is True
            - A list parallel to the Training event list with boolean values of whether it is a signal event
            - A list parallel to the Training event list containing weights (all 1, in this case)
        * Second tuple corresponds to testing/validation set and has three elements:
            - Set of relevant values of all events whose isTraining is False
            - A list parallel to the Testing event list with boolean values of whether it is a signal event
            - A list parallel to the Testing event list containing weights (all 1, in this case)
    
    The testing set needs to be passed on as the eval_set argument of the classifier so that it can calculate
    the error in estimation in the testing set parallely while training.
    
    early_stopping_rounds=100 means the network will stop training if there is no improvement in error in validation
    set in the last 100 iterations.
    '''
#============================================================================


#============================================================================
# Hyperparameter tuning: Optimizing Learning Rate
#============================================================================
'''
One important task while using ML is to tune the hyperparameters to obtain lowest error rate while
simultaneously minimizing the difference in error rates between the training and testing datasets
to avoid overfitting. This is particularly time consuming as there are a plethora of 
hyperparameters that one needs to look at and one usually performs a random search in a
d-dimensional hyperparameter space, instead of a sequential search, since the performance for
various combinations can be quite erratic.

In this case, I perform a sequential search on only one hyperparameter: Learning Rate,
which is relatively important. However, a sequential on just one variable takes a few hours in my
machine. Hence, I do not look at other parameters at all.

The following section takes care of this and produces plots of errors of training and testing cases
as a function of LR. I find that LR = 0.15 gives a satisfactory result. I have commented out this
section and it can be uncommented to reproduce the tuning results.
'''

#LRset=np.arange(.01,.41,0.01)          # Sequential search in this range
#errset1=[]
#errset2=[]
#for LR in LRset:
    #cls4mu=train(LR,dtframe4mu,col4mu[:-XtraVars])
    #errset1.append(cls4mu.evals_result_["validation_0"]["error"][-1])
    #errset2.append(cls4mu.evals_result_["validation_1"]["error"][-1])
#mininfo="LR:"+"%.2f"%LRset[np.argmin(errset2)]+", Err_test:""%.3f"%min(errset2)+", Err_train:"+"%.3f"%errset1[np.argmin(errset2)]
#plt.plot(LRset,errset1,label="Train")
#plt.plot(LRset,errset2,label="Test")
#plt.title("min @ "+mininfo)
#plt.legend()
#plt.savefig("cls4mu_error.png")
#============================================================================


#============================================================================
# Run the training and store the trainings in two separate XGBoost classifiers
#============================================================================
cls4mu=train(0.15,dtframe4mu,col4mu[:-XtraVars],True)
cls2mu2e=train(0.15,dtframe2mu2e,col2mu2e[:-XtraVars],True)
#============================================================================


#============================================================================
# Validate the training
#============================================================================
discset=np.arange(0.,1.,0.01)

def validate(cls,dtframe,col,name):
    '''
    This function can be called to validate a training, i.e., find how well the classifier can distinguish
    signal events apart from background events.
    
    Input:
    cls: A trained XGBoost Classifier
    dtframe: The pandas dataframe
    col: names of the columns in a list/array format, must match with those used to train cls
    name: A string to append to the output validation plot filenames
    
    Returns:
    0: List with fraction of signal events passing the discriminator value located in the same index inside discset
    1: List with fraction of background events passing the discriminator value located in the same index inside discset
    2: Numpy array containing significance value corresponding to each discriminator value located in the same index inside discset
    
    We define significance as (fraction of signal events passing)/sqrt(fraction of background events passing) for each disc value
    '''
    
    bkgvals=cls.predict_proba(dtframe.loc[(dtframe["isTraining"]==0) & (dtframe["isSignal"]==0),col])[:,1]
    sigvals=cls.predict_proba(dtframe.loc[(dtframe["isTraining"]==0) & (dtframe["isSignal"]==1),col])[:,1]
    
    #Note, we strictly use events for which "isTraining"==0. Training and testing datasets MUST be exclusive.
    
    plt.clf()
    
    bkgfracset=[]
    sigfracset=[]
    for disc in discset:
        bkgcount=0
        for bkg in bkgvals:
            if bkg > disc: bkgcount += 1
        bkgfracset.append(float(bkgcount)/len(bkgvals))     # Fraction of bkg events above each discriminator value
        
        sigcount=0
        for sig in sigvals:
            if sig > disc: sigcount += 1
        sigfracset.append(float(sigcount)/len(sigvals))     # Fraction of signal events above each discriminator value
        
    signifset=np.divide(np.array(sigfracset),np.sqrt(np.array(bkgfracset)), out=np.zeros_like(np.array(sigfracset)), where=np.sqrt(np.array(bkgfracset))!=0)
    # Calculates significance using the formula defined above. If denominator is zero, set significance as zero instead.
    
    #------------------------------------------------------------------------
    # Plot sig and bkg performance in one plot and significance in another
    #------------------------------------------------------------------------
    plt.plot(discset,sigfracset,label='Signal')
    plt.plot(discset,bkgfracset,label='Background')
    
    plt.xlabel("Discriminator value")
    plt.ylabel("Fraction passing cut")
    plt.legend()
    plt.savefig("signaleff_"+name+".png")
    
    plt.clf()
    plt.plot(discset,signifset)
    plt.xlabel("Discriminator value")
    plt.ylabel("Significance")
    plt.savefig("significance_"+name+".png")
    #------------------------------------------------------------------------
    
    return sigfracset,bkgfracset,signifset
    
sigfracset4mu,bkgfracset4mu,signifset4mu=validate(cls4mu,dtframe4mu,col4mu[:-XtraVars],"4mu")
sigfracset2mu2e,bkgfracset2mu2e,signifset2mu2e=validate(cls2mu2e,dtframe2mu2e,col2mu2e[:-XtraVars],"2mu2e")
#============================================================================


#============================================================================
# Finding the operating point: Optimizing the discriminator value
#============================================================================
'''
We simply find the point with highest value of significance, and set the corresponding value
of the discriminator as the cut off operating point.

Since this value would be dependent on the training, which is itself dependent on random numbers
that decide the training dataset, we save these information in a log file.
'''
logfile=open("logfile.txt","w")
logfile.write("Extra variables = "+str(XtraVars)+"\n\n")

maxInd=np.argmax(signifset4mu[:-15])        # The [:-15] is to ensure that the very sharp fluctuations in the last few values of the significance (due to bkg fraction (denominator) values close to 0) are not used while picking the maximum.
disccut4mu=discset[maxInd]
line1= "Using disc cut of %.2f for 4 Mu region.\nBkg frac = %f\nSig frac = %f\n"%(disccut4mu,bkgfracset4mu[maxInd],sigfracset4mu[maxInd])
logfile.write(line1)
print line1

logfile.write("\n")

maxInd=np.argmax(signifset2mu2e[:-15])
disccut2mu2e=discset[maxInd]
line2 = "Using disc cut of %.2f for 2 Mu, 2 e region.\nBkg frac = %f\nSig frac = %f\n"%(disccut2mu2e,bkgfracset2mu2e[maxInd],sigfracset2mu2e[maxInd])
logfile.write(line2)
print line2

logfile.close()

## When using very small training/validation/data set, uncomment the following lines to manually set a reasonable value of the discriminator cut to get some events in the output.
#disccut4mu=0.6
#disccut2mu2e=0.6
#print "WARNING: Manually readjusted discriminator cuts to 0.6. Read in-comment code for details." 

#============================================================================



#============================================================================
# Apply the training on data (finally!!)
#============================================================================
dirname="data"
data4mu=[]
data2mu2e=[]

#------------------------------------------------------------------------
# Load data files and run the same analysis routine to get a similar dataframe for data
#------------------------------------------------------------------------
ttree = TChain("MuonAn/MuonEvents")

rootfilelist=[fl for fl in os.listdir(dirname) if fl.endswith(".root")]
for rootfile in rootfilelist:
    ttree.Add(dirname+"/"+rootfile)
print "Reading %d file(s)."%(len(rootfilelist))
    
NEntries = ttree.GetEntries()

isSignal=0

if len(sys.argv)>1:
    #if sys.argv[2]=="test":
    NEntries=min(NEntries,int(sys.argv[1]))
    print "WARNING: Running in TEST MODE"

print 'NEntries = '+str(NEntries)
        
        
for ievent in range(NEntries):
    if ievent%10000==0: print "Processed %d of %d events: ~%d%%" %(ievent,NEntries,int(ievent*100/NEntries))
    
    row,reg=AnalyzeEvent(ttree,ievent,isSignal)
    if reg=="4mu":     
        data4mu.append(row)
    elif reg=="2mu2e":
        data2mu2e.append(row)

dtframedata4mu=pd.DataFrame(data=np.array(data4mu),columns=col4mu)
dtframedata2mu2e=pd.DataFrame(data=np.array(data2mu2e),columns=col2mu2e)
#------------------------------------------------------------------------


#------------------------------------------------------------------------
# Predict disc value of each event in the data for each category
#------------------------------------------------------------------------
datadiscs4mu=cls4mu.predict_proba(dtframedata4mu[col4mu[:-XtraVars]])[:,1]
datadiscs2mu2e=cls2mu2e.predict_proba(dtframedata2mu2e[col2mu2e[:-XtraVars]])[:,1]
#------------------------------------------------------------------------


#------------------------------------------------------------------------
# Fill information from events with disc values above cut-off value into histograms.
# We call these events the selected events and these are potentially Higgs events!
#------------------------------------------------------------------------

for i,disc in enumerate(datadiscs4mu):
    if disc>disccut4mu:
        DiZMass=dtframedata4mu.loc[i,"DiZMass"]
        Z1Mass=dtframedata4mu.loc[i,"Z1.M"]
        Z2Mass=dtframedata4mu.loc[i,"Z2.M"]
        
        if not np.isnan(DiZMass):                       # This safety check would be relevant if I didn't comment out the 3 lepton case
            h_4mumass.Fill(DiZMass)
            h_higgsmass.Fill(DiZMass)
        if not np.isnan(Z1Mass): h_Zmass.Fill(Z1Mass)
        if not np.isnan(Z2Mass): h_Zmass.Fill(Z2Mass)
        
for i,disc in enumerate(datadiscs2mu2e):
    if disc>disccut2mu2e:
        DiZMass=dtframedata2mu2e.loc[i,"DiZMass"]
        Z1Mass=dtframedata2mu2e.loc[i,"Z1.M"]
        Z2Mass=dtframedata2mu2e.loc[i,"Z2.M"]
        
        if not np.isnan(DiZMass):                       # This safety check would be relevant if I didn't comment out the 3 lepton case
            h_2mu2emass.Fill(DiZMass)
            h_higgsmass.Fill(DiZMass)
        if not np.isnan(Z1Mass): h_Zmass.Fill(Z1Mass)
        if not np.isnan(Z2Mass): h_Zmass.Fill(Z2Mass)
        
print "Done."
#------------------------------------------------------------------------
#============================================================================   


#============================================================================
# Write the output histograms to an output file
#============================================================================            
f = TFile(outfilename,'RECREATE')
f.cd()
h_skimmed.SetBinContent(1,NEntries)
h_skimmed.Write()
h_4mumass.Write()
h_2mu2emass.Write()
h_Zmass.Write()
h_higgsmass.Write()

print "Histograms written to %s." %outfilename
#============================================================================ 
