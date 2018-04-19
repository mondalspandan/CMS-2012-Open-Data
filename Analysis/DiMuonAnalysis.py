#!/usr/bin/env python
from ROOT import *
import ROOT
import os,sys
from math import log

outfilename= "DiMuon.root"      # Name of output file containing the final histograms

#============================================================================
# Load one/all input files in a single event-based tree structure: TChain
#============================================================================

if len(sys.argv)<2 or len(sys.argv)>3:
    print "Usage: python DiMuonAnalysis.py RootFileORDir [maxEvents]"       # To ensure correct number of inputs
    sys.exit()

skimmedtree = TChain("MuonAn/MuonEvents")       # This is a TChain to read and store event-based trees
inp=sys.argv[1]         
if inp.endswith(".root"):                       # If input is a single .root file, read the input file as a root file
    skimmedtree.Add(inp)
    print "Reading file:",inp
else:                                           # Else assume it's a directory and read all .root files inside this directory
    rootfilelist=[fl for fl in os.listdir(inp) if fl.endswith(".root")]
    for rootfile in rootfilelist:
        skimmedtree.Add(inp+"/"+rootfile)       # Add trees from each of these input root files to our TChain
    print "Reading %d files."%(len(rootfilelist))

NEntries = skimmedtree.GetEntries()             # Number of events in the TChain

if len(sys.argv)>2:                             # Option to test the programme with a subset of all events, specified by user
    #if sys.argv[2]=="test":
    NEntries=min(NEntries,int(sys.argv[2]))     # Sanity check to choose the smaller of the two
    print "WARNING: Running in TEST MODE"

print 'NEntries = '+str(NEntries)
#============================================================================


#============================================================================
# Define output histograms
#============================================================================

h_dimumass = TH1F('h_dimumass','h_dimumass',1000,1.,120.)
h_dimumass.GetXaxis().SetTitle("DiMuon invariant mass (in GeV/c)")
h_dimumass.GetYaxis().SetTitle("Number of events")

h_dimumassxlog = TH1F('h_dimumassxlog','h_dimumassxlog',1000,log(.4,10.),log(200,10.))      # log along x-axis
h_dimumassxlog.GetXaxis().SetTitle("log_{10} DiMuon invariant mass (in log(GeV/c))")
h_dimumassxlog.GetYaxis().SetTitle("Number of events")

h_skimmed = TH1F('h_skimmed','h_skimmed',2,0.,2)
h_skimmed.GetYaxis().SetTitle("Number of events after skimming")
#============================================================================


#============================================================================
# Start the event loop
#============================================================================

for ievent in range(NEntries):
    if ievent%100000==0: print "Processed %d of %d events: ~%d%%" %(ievent,NEntries,int(ievent*100/NEntries))
    
    skimmedtree.GetEntry(ievent)
    
    for attrib in ['muP4','nMu','muCharge']: #['muP4','nMu','muIso','muCharge','nEle','eleP4','eleIso','eleCharge','nJet','jetP4']
        exec(attrib+" = skimmedtree.__getattr__('"+attrib+"')")             # Load each of the necessary values stored in the trees in a variable of the same name
    
    if nMu<2: continue                              # Sanity check to make sure there are at least 2 muons
    
    muPlus=[]
    muMinus=[]
    for imu in range(nMu):                          # Make two different lists of muon 4-vectors (LorentzVector class in ROOT), one for mu+ and one for mu-.
        if muCharge[imu]>0:
            muPlus.append(muP4[imu])
        else:
            muMinus.append(muP4[imu])
    
    '''
     In each event, for every combination of mu+ mu- pair, we calculate the invariant mass of the "sum" of the two muons.
     If the pair actually originated from a single parent particle, the event distribution of this invariant mass across all events will show peaks at the mass of the parent particle, while all incorrect combinations will yield random values and appear as stray backgrounds.
     Addition (+) is an operation defined in the LorentzVector class in ROOT. It returns a new LorentzVector whose E, px, py and pz each are the sum of the corresponding quantities of the inputs. The sum is called the reconstructed particle.
     M() is a function defined in the LorentzVector class that calculates the invariant mass as sqrt(E^2-p^2). We use it to skip performing a manual calculation using the said formula.
    '''
    
    for muP in muPlus:
        for muN in muMinus:
            dimumass=(muP+muN).M()                  
            h_dimumass.Fill(dimumass)               # Fill the invariant mass value in a histogram
            h_dimumassxlog.Fill(log(dimumass,10.))  # Fill log of the invariant mass in another histogram

#============================================================================


#============================================================================
# The TSpectrum class is used find major peaks, but did not work efficiently for this histogram, so I commented it out
#============================================================================
#spec=TSpectrum(20)
#spec.Search(h_dimumassxlog,.001,"peaks",.1)
#buf=spec.GetPositionX()
#peaks=[]
#for i in range(spec.GetNPeaks()):
#    peaks.append(buf[i])
#peaks.sort()
#print "Major peaks:", [10**i for i in peaks], "GeV."
#============================================================================


#============================================================================
# Write the histograms to an output file
#============================================================================
            
f = TFile(outfilename,'RECREATE')
f.cd()
h_skimmed.SetBinContent(1,NEntries)             # To get an idea of how many events passed the skimming step
h_skimmed.Write()
h_dimumass.Write()
h_dimumassxlog.Write()

print "Histograms written to %s." %outfilename
#============================================================================
