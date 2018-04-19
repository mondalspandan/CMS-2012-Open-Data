#!/usr/bin/env python
from ROOT import *
import ROOT
import os,sys
from math import log

outfilename= "Higgs.root"
ZMass=91.


#============================================================================
# Load one/all input files in a single event-based tree structure: TChain
# (Step wise explanation presented in DiMuonAnalysis.py)
#============================================================================
if len(sys.argv)<2 or len(sys.argv)>3:
    print "Usage: python HiggsAnalysis.py RootFileORDir [maxEvents]"
    sys.exit()

skimmedtree = TChain("MuonAn/MuonEvents")
inp=sys.argv[1]
if inp.endswith(".root"):
    skimmedtree.Add(inp)
    print "Reading file:",inp
else:
    rootfilelist=[fl for fl in os.listdir(inp) if fl.endswith(".root")]
    for rootfile in rootfilelist:
        skimmedtree.Add(inp+"/"+rootfile)
    print "Reading %d files."%(len(rootfilelist))

NEntries = skimmedtree.GetEntries()

if len(sys.argv)>2:
    #if sys.argv[2]=="test":
    NEntries=min(NEntries,int(sys.argv[2]))
    print "WARNING: Running in TEST MODE"

print 'NEntries = '+str(NEntries)
#============================================================================


#============================================================================
# Define output histograms
#============================================================================
higgsmin,higgsmax,nhiggsbins=40.,440.,100

h_4mumass = TH1F('h_4mumass','h_4mumass',nhiggsbins,higgsmin,higgsmax)
h_4mumass.GetXaxis().SetTitle("4-Muon invariant mass (in GeV/c)")
h_4mumass.GetYaxis().SetTitle("Number of events")

h_2mu2emass = TH1F('h_2mu2emass','h_2mu2emass',nhiggsbins,higgsmin,higgsmax)
h_2mu2emass.GetXaxis().SetTitle("2-Muon, 2-Electron invariant mass (in GeV/c)")
h_2mu2emass.GetYaxis().SetTitle("Number of events")

h_higgsmass = TH1F('h_higgsmass','h_higgsmass',nhiggsbins,higgsmin,higgsmax)
h_higgsmass.GetXaxis().SetTitle("Candidate Higgs invariant mass (in GeV/c)")
h_higgsmass.GetYaxis().SetTitle("Number of events")

h_Zmumumass = TH1F('h_Zmumumass','h_Zmumumass',40,70.,110.)
h_Zmumumass.GetXaxis().SetTitle("Candidate Z invariant mass (in GeV/c) from Z -> #mu#mu")
h_Zmumumass.GetYaxis().SetTitle("Number of events")

h_Zeemass = TH1F('h_Zeemass','h_Zeemass',40,70.,110.)
h_Zeemass.GetXaxis().SetTitle("Candidate Z invariant mass (in GeV/c) from Z -> ee")
h_Zeemass.GetYaxis().SetTitle("Number of events")

h_Zmass = TH1F('h_Zmass','h_Zmass',40,70.,110.)
h_Zmass.GetXaxis().SetTitle("Candidate Z invariant mass (in GeV/c) from Z -> #mu#mu/ee")
h_Zmass.GetYaxis().SetTitle("Number of events")

h_skimmed = TH1F('h_skimmed','h_skimmed',2,0.,2)
h_skimmed.GetYaxis().SetTitle("Number of events after skimming")
#============================================================================

#def getpT(P4):              # A function to use as "key" in the "sorted()" command
#    return P4.pt()

#def isZ(P4):
#    if abs(P4.M()-ZMass)<6.: return True
#    return False

#============================================================================
# Start the event loop
#============================================================================
for ievent in range(NEntries):
    if ievent%10000==0: print "Processed %d of %d events: ~%d%%" %(ievent,NEntries,int(ievent*100/NEntries))
    
    skimmedtree.GetEntry(ievent)
    
    for attrib in ['muP4','nMu','muCharge','nEle','eleP4','eleCharge']: #['muP4','nMu','muIso','muCharge','nEle','eleP4','eleIso','eleCharge','nJet','jetP4']
        exec(attrib+" = skimmedtree.__getattr__('"+attrib+"')")      # Load each of the necessary values stored in the trees in a variable of the same name
    
    muP4=list(muP4)                 # Conversion from ROOT.vector to python list
    eleP4=list(eleP4)
    
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
    eleP4old=eleP4[:]               # [:] to create a copy of the list instead of simply storing the pointer.
    eleChargeold=eleCharge[:]
    nEle=0
    eleP4=[]
    eleCharge=[]
    for ie in range(nEleold):
        if eleP4old[ie].pt()>7.:
            eleP4.append(eleP4old[ie])
            eleCharge.append(eleChargeold[ie])
            nEle+=1
    #------------------------------------------------------------------------ 
    
    
    #------------------------------------------------------------------------
    # H -> ZZ -> 4mu
    #------------------------------------------------------------------------
    if nMu>=4:                  
        muPlus=[]
        muMinus=[]
        for imu in range(nMu):                  # Make two different lists of muon 4-vectors (LorentzVector class in ROOT), one for mu+ and one for mu-.
            if muCharge[imu]>0:
                muPlus.append(muP4[imu])
            else:
                muMinus.append(muP4[imu])
        
        if len(muPlus)>=2 and len(muMinus)>=2:          # Require at least two mu+ and two mu-
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
            
            Z1=ZP4temp[idx]                                     # Store the LorentzVector of the dimuon candidate with mass closest to Z_mass in Z1
            if Z1.M()>40. and Z1.M()<120. :                     # Assert that Z1 must have a mass between 40 and 120 GeV, in line with CMS Higgs analysis
                
                muP1=Zmus[idx][0]
                muN1=Zmus[idx][1]
                muPlus.remove(muP1)
                muMinus.remove(muN1)                            # Remove the used up mu+ and mu- from our muon collection
                
                h_Zmumumass.Fill(Z1.M())                        # Fill the invariant mass of the dilepton (Z) candidate in relevant category.
                h_Zmass.Fill(Z1.M())
                
                #leadmuP = sorted(muPlus, key=getpT, reverse=True)[0]
                #leadmuN = sorted(muMinus, key=getpT, reverse=True)[0]                
                
                ZP4temp=[]
                #Zmus=[]
                for muP in muPlus:
                    for muN in muMinus:
                        ZP4temp.append(muP+muN)                 # Repeat the same process to find the next best dimuon combination (we have already removed the used up muons from muP and muN)
                        #Zmus.append([muP,muN])
                
                massdiff=[abs(Z.M()-ZMass) for Z in ZP4temp]
                val, idx = min((val, idx) for (idx, val) in enumerate(massdiff))        # Same way to find the best dilepton candidate among the remaining muons
                
                #Z2=(leadmuP+leadmuN)                       # I experimented with some other methods to select the second muon pair, removed later.
                
                Z2=ZP4temp[idx]                         # Store the LorentzVector of the second dimuon candidate with mass closest to Z_mass in Z2
                if Z2.M()>12. and Z2.M()<120.:          # Assert that Z2 must have a mass between 12 and 120 GeV, in line with CMS Higgs analysis
                    #muP2=leadmuP
                    #muN2=leadmuN
                    #print Z1.M(),Z2.M()
                    
                    h_Zmumumass.Fill(Z2.M())            # Fill histograms
                    h_Zmass.Fill(Z2.M())
                    
                    #if isZ(Z1) or isZ(Z2):             # I experimented with adding an additional requirement that at least one Z is on-shell, later removed.
                    
                    DiZMass=(Z1+Z2).M()                 # Add 4-momenta of Z1 and Z2, and call this the candidate Higgs
                    
                    #nl2combmass=0              
                    #for muP in [muP1,muP2]:
                        #for muN in [muN1,muN2]:
                            #if (muP+muN).M()>12.:      # Other method, removed.
                                #nl2combmass+=1
                    #if nl2combmass >= 3:
                    
                    #print "4mu:", DiZMass
                    
                    h_4mumass.Fill(DiZMass)             # Fill Higgs histograms
                    h_higgsmass.Fill(DiZMass)    
     
    #------------------------------------------------------------------------ 
    
    
    #------------------------------------------------------------------------
    # H -> ZZ -> 2mu + 2e
    #------------------------------------------------------------------------
    if nMu>=2 and nEle>=2:
        muPlus=[]
        muMinus=[]
        for imu in range(nMu):
            if muCharge[imu]>0:
                muPlus.append(muP4[imu])            # Make two different lists of muon 4-vectors, one for mu+ and one for mu-.
            else:
                muMinus.append(muP4[imu])
                
        elePlus=[]
        eleMinus=[]
        for iele in range(nEle):
            if eleCharge[iele]>0:
                elePlus.append(eleP4[iele])         # Make two different lists of electron 4-vectors, one for e+ and one for e-.
            else:
                eleMinus.append(eleP4[iele])
        
        if len(muPlus)>=1 and len(muMinus)>=1 and len(elePlus)>=1 and len(eleMinus)>=1:         # Require at least one each of mu+, mu-, e+, e- in the event.
            ZP4temp=[]
            whichLep=[]
            for muP in muPlus:
                for muN in muMinus:
                    ZP4temp.append(muP+muN)         # Fill dimuon combinations in a list
                    whichLep.append("mu")           # A parallel list stating that this dilepton combination is a dimuon comb.
            for eleP in elePlus:
                for eleN in eleMinus:
                    ZP4temp.append(eleP+eleN)       # Fill dielectron combinations in the SAME list
                    whichLep.append("e")            # A parallel list stating that this dilepton combination is a dielectron comb.
                    
            massdiff=[abs(Z.M()-ZMass) for Z in ZP4temp]                            # Store |m_ll - m_Z| values in a parallel list
            val, idx = min((val, idx) for (idx, val) in enumerate(massdiff))        # This steps stores the list index of the dilepton candidate that has the lowest value of |m_ll - m_Z| inside the variable idx
            
            Z1=ZP4temp[idx]                         # Best combination is stored as Z1
            if Z1.M()>40. and Z1.M()<120. :         # Require Z1 to be between 40 and 120 GeV
                
                h_Zmass.Fill(Z1.M())                # Fill histogram
                
                Z1lep=whichLep[idx]
                
                if whichLep[idx]=="mu":             
                    lepPlus=elePlus                 # If Z1 is a dimuon combination, store electron values in lepPlus and lepMinus
                    lepMinus=eleMinus
                    h_Zmumumass.Fill(Z1.M())
                else:
                    lepPlus=muPlus                  # If Z1 is a dielectron combination, store muon values in lepPlus and lepMinus
                    lepMinus=muMinus
                    h_Zeemass.Fill(Z1.M())
                
                #leadlepP = sorted(lepPlus, key=getpT, reverse=True)[0]
                #leadlepN = sorted(lepMinus, key=getpT, reverse=True)[0]
                
                #Z2=(leadlepP+leadlepN)
                #if Z2.M()>12.:
                
                ZP4temp=[]
                for lepP in lepPlus:
                    for lepN in lepMinus:
                        ZP4temp.append(lepP+lepN)  # Now we repeat the same process with lepPlus and lepMinus, which now contain info of the lepton opposite in flavour to the Z1 constituents
                
                massdiff=[abs(Z.M()-ZMass) for Z in ZP4temp]
                val, idx = min((val, idx) for (idx, val) in enumerate(massdiff))    # Same way to find the best dilepton candidate among the other leptons
                Z2=ZP4temp[idx]
                if Z2.M()>12. and Z2.M()<120. :                 # Assert that Z2 must have a mass between 12 and 120 GeV
                    Z2=ZP4temp[idx]
                    
                    if Z1lep=="mu":
                        h_Zeemass.Fill(Z2.M())                  # If Z1 leptons were muons, Z2 leps are electrons, so fill Z->ee histo    
                    else:
                        h_Zmumumass.Fill(Z2.M())
                        
                    h_Zmass.Fill(Z2.M())
                    #print Z1.M(),Z2.M()
                    
                    #if isZ(Z1) or isZ(Z2):
                    
                    DiZMass=(Z1+Z2).M()                         # Add 4-momenta of Z1 and Z2, and call this the candidate Higgs
                    
                    #print "2mu2e:", DiZMass
                    #print Z1.M(),Z2.M()                    
                    
                    h_2mu2emass.Fill(DiZMass)                   # Fill histos
                    h_higgsmass.Fill(DiZMass)
    #------------------------------------------------------------------------
    
#============================================================================   


#============================================================================
# Write the histograms to an output file
#============================================================================   
f = TFile(outfilename,'RECREATE')
f.cd()
h_skimmed.SetBinContent(1,NEntries)
h_skimmed.Write()
h_4mumass.Write()
h_2mu2emass.Write()
h_Zmumumass.Write()
h_Zeemass.Write()
h_Zmass.Write()
h_higgsmass.Write()

print "Histograms written to %s." %outfilename
#============================================================================
