// -*- C++ -*-
//
// Package:    MuonEventsSkimmer
// Class:      MuonEventsSkimmer
// 
/**\class MuonEventsSkimmer MuonEventsSkimmer.cc MuonResonance/MuonEventsSkimmer/src/MuonEventsSkimmer.cc

 Description: Skims 2012 opendata to keep events with two oppositely charged muons. Limited event information is saved. Potentially compatible with 2011 opendata.

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Spandan Mondal,,,
//         Created:  Sun Feb  4 08:54:44 CEST 2018
// $Id$
//
//

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

//=====================================================
// My includes (not default)
//=====================================================
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "DataFormats/Common/interface/Ref.h"
#include "TTree.h"
#include "TLorentzVector.h"
#include "math.h"
// Muon
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/MuonReco/interface/MuonSelectors.h"
#include "DataFormats/MuonReco/interface/MuonIsolation.h"
#include "DataFormats/MuonReco/interface/MuonPFIsolation.h"
// Vertex
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
// Electrons
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrackFwd.h"
// Jets
#include "DataFormats/JetReco/interface/PFJet.h"

#include "TMath.h"
typedef math::XYZTLorentzVector LorentzVector;
#ifdef __CINT__
#pragma link C++ class std::vector<LorentzVector>+;        // Need to make a dictionary for ROOT to recognize TLorentzVector
//#pragma link C++ class TLorentzVector+;
#endif

//=====================================================

//
// class declaration
//

class MuonEventsSkimmer : public edm::EDAnalyzer {
   public:
      explicit MuonEventsSkimmer(const edm::ParameterSet&);
      ~MuonEventsSkimmer();

      static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);


   private:
      virtual void beginJob() ;
      virtual void analyze(const edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;

      virtual void beginRun(edm::Run const&, edm::EventSetup const&);
      virtual void endRun(edm::Run const&, edm::EventSetup const&);
      virtual void beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&);
      virtual void endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&);
      
      //=====================================================
      // Here I declare the global variables
      //=====================================================      
      TTree *tree_;
      TH1F *h_total;
      int nMu, nEle, nJet;
      std::vector<LorentzVector> muP4, eleP4, jetP4;
      std::vector<float> muIso, eleIso, muSIP3D, eleSIP3D;
      std::vector<int> muCharge, eleCharge;
//       bool writeEvent = false;
      //=====================================================

      // ----------member data ---------------------------
};

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
MuonEventsSkimmer::MuonEventsSkimmer(const edm::ParameterSet& iConfig)

{
    //now do what ever initialization is needed
    
    //=====================================================
    // Initialize a tree in the output and add branch for 
    // quantities that I wish to store in the output
    //===================================================== 
    /*
        Note how each branch has a pointer to a global variable.
        This way if we change the global variable values inside any
        function and then call the "tree->Fill()" function, an event
        with branches equal to those values will be saved in the 
        output. Thus, we can simply keep updating the values of
        the global variables from inside the event loop function
        and call Fill() function at the end of each event, if the
        event needs to be saved.
    */
    edm::Service<TFileService> fs;
    
    h_total = fs->make<TH1F>("h_total", "h_total", 2, 0, 2); 
    
    tree_ = fs->make<TTree>("MuonEvents","tree");
    
    tree_->Branch("nMu",&nMu,"nMu/I");    
    tree_->Branch("muP4",&muP4);
    tree_->Branch("muIso",&muIso);
    tree_->Branch("muSIP3D",&muSIP3D);    
    tree_->Branch("muCharge",&muCharge);
    
    tree_->Branch("nEle",&nEle,"nEle/I");    
    tree_->Branch("eleP4",&eleP4);
    tree_->Branch("eleIso",&eleIso);
    tree_->Branch("eleSIP3D",&eleSIP3D);
    tree_->Branch("eleCharge",&eleCharge);
    
    tree_->Branch("nJet",&nJet,"nJet/I");    
    tree_->Branch("jetP4",&jetP4);    
    //=====================================================
}



MuonEventsSkimmer::~MuonEventsSkimmer()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called for each event  ------------
void
MuonEventsSkimmer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;

 
    /*
        This is the where one should perform the analysis.
        CMSSW calls this function on a per-event basis. So the code in this part
        is basically inside a loop over events. We do whatever we want to do for
        EACH event inside this function.
    */

#ifdef THIS_IS_AN_EVENT_EXAMPLE
   Handle<ExampleData> pIn;
   iEvent.getByLabel("example",pIn);
#endif
   
#ifdef THIS_IS_AN_EVENTSETUP_EXAMPLE
   ESHandle<SetupData> pSetup;
   iSetup.get<SetupRecord>().get(pSetup);
#endif
    
    //=====================================================
    // Access AOD collections
    //=====================================================
    /*
        Objects are stored inside different collections in the input AOD file.
        One needs to access these collections and "Get" the physics objects
        using CMSSW/ROOT notations. I access and store them in similarly named
        variables.
        
        Ref: https://twiki.cern.ch/twiki/bin/view/CMSPublic/SWGuideAodDataTable
        (This reference is newer and could be incompatible with 2012 data, but it seems
        the objects that I am accessing had the same notations in 2012. So I could
        use this without issues.)
    */
    
    edm::Handle<reco::MuonCollection> muons;
    iEvent.getByLabel("muons", muons);
    
    edm::Handle<reco::GsfElectronCollection> electrons;
    iEvent.getByLabel("gsfElectrons",electrons);
    
    edm::Handle<reco::PFJetCollection> jets;
    iEvent.getByLabel("ak5PFJets",jets);                    // Choosing anti-kT PF jets with R = 0.5 (thin jets)
    
    edm::Handle<reco::VertexCollection> primvtxHandle;
    iEvent.getByLabel("offlinePrimaryVertices", primvtxHandle);
    //=====================================================
    
    
    //=====================================================
    // Get location of the primary vertex
    //=====================================================
    reco::VertexCollection primvtx;
    if ( primvtxHandle.isValid() )
    {
      primvtx = *primvtxHandle;
    }
    
    math::XYZPoint point(primvtx[0].position());
    //=====================================================
    
    
    //=====================================================
    // Reset global variables (to reuse from previous iteration)
    //=====================================================
    nMu=0;
    muP4.clear();
    muIso.clear();
    muSIP3D.clear();
    muCharge.clear();
    
    nEle=0;
    eleP4.clear();
    eleIso.clear();
    eleSIP3D.clear();
    eleCharge.clear();
    
    nJet=0;
    jetP4.clear();
    
//     writeEvent=false;
    //=====================================================
    
    
    //=====================================================
    // Start choosing events suiting our needs
    //=====================================================
    
    if (muons->size() >= 2) {
        
        // Define and initiate local variables
        int nMuPlus = 0;
        int nMuMinus = 0;
        int nMutemp = 0;
        float mu_iso = 1.;
        float ele_iso = 1.;
        float muIP3d=10., muSigIP3d=10., muSIP3d=10.;
        float eleIP3d=10., eleSigIP3d=10., eleSIP3d=10.;
        bool hasSIP3D=false;
        
        h_total->Fill(1);           // Fill a counter to keep track of total dimuon events processed, just in case
        
        //------------------------------------------------
        // Loop over all muons
        //------------------------------------------------
        /*
            The idea is to first go through all the "good" muons and count how many
            are mu+ and mu-. If there is at least one of each, we keep the event.
        */
        for (reco::MuonCollection::const_iterator mu = muons->begin(); mu != muons->end(); mu++) {      //muon loop
        
            /*
                First we calculate 3D IP significance using the formula prescribed.
                However, some muons may not have global track information.
                
                If a muon has global 3D IP sig: we check if it passes required 3DIPsig criteria
                If it doesn't, we keep it without checking whether it passes the criteria                
            */
            
            if ((mu->globalTrack()).isNonnull()) {                
                muIP3d = sqrt((mu->globalTrack()->dxy(point) * mu->globalTrack()->dxy(point)) + (mu->globalTrack()->dz(point) * mu->globalTrack()->dz(point)));
	            muSigIP3d = sqrt((mu->globalTrack()->d0Error() * mu->globalTrack()->d0Error()) + (mu->globalTrack()->dzError() * mu->globalTrack()->dzError()));
	            muSIP3d = muIP3d/muSigIP3d;
                hasSIP3D = true;
            }
            else {
                hasSIP3D = false;
                muSIP3d = 10.;
            }
	        
	        // Calculate muon's total isolation using the recommended formula
            mu_iso = ((mu->pfIsolationR04()).sumChargedHadronPt + (mu->pfIsolationR04()).sumNeutralHadronEt + (mu->pfIsolationR04()).sumPhotonEt) / mu->pt();
            
            /* Now we see if the muon matches the "good" muon criteria
                - pT > 5 GeV
                - |eta| < 2.4
                - Isolation < .4
                - 3D IP significance < 4 or unavailable
            */
            if (mu->pt() > 5. && std::abs(mu->eta()) < 2.4 && mu_iso < .4 && ( !hasSIP3D || std::abs(muSIP3d) < 4.)) {
                ++nMutemp;
                if (mu->charge() > 0) ++nMuPlus;            // Count number of good mu+ and good mu-
                if (mu->charge() < 0) ++nMuMinus;                
                
                /*
                    We fill the global variables for this good muon in this loop itself.
                    Note: Just assigning the values to the variables does not mean they are
                    being saved in the output! We shall save this information only if further
                    conditions are satisfied. I assign the muon values in this loop itself
                    only to save creating a new muon loop in the final filling section.
                */
                muP4.push_back(mu->p4());
                muIso.push_back(mu_iso);
                muSIP3D.push_back(muSIP3d);
                muCharge.push_back(mu->charge());                
            }            
        }
        //------------------------------------------------
        
        
        //------------------------------------------------
        // Require two oppositely charged good muons and save event
        //------------------------------------------------
        if (nMutemp >= 2 && nMuPlus >= 1 && nMuMinus >= 1) {
            nMu=nMutemp;
//             writeEvent=true;
            
            // Loop over electron collection and choose good electrons and save them in the event
            for (reco::GsfElectronCollection::const_iterator ele = electrons->begin(); ele != electrons->end(); ele++) {
                
                // Calculate electron isolation
                ele_iso = ((ele->pfIsolationVariables()).chargedHadronIso + (ele->pfIsolationVariables()).neutralHadronIso + (ele->pfIsolationVariables()).photonIso) /ele->pt();
                
                // Calculate electron 3D IP significance (track info is always available)
                eleIP3d = sqrt((ele->gsfTrack()->dxy(point) * ele->gsfTrack()->dxy(point)) + (ele->gsfTrack()->dz(point) * ele->gsfTrack()->dz(point)));
                eleSigIP3d = sqrt ((ele->gsfTrack()->d0Error() * ele->gsfTrack()->d0Error()) + (ele->gsfTrack()->dzError() * ele->gsfTrack()->dzError()) );
                eleSIP3d = eleIP3d / eleSigIP3d;
                
                /* See if the electron matches the "good" electron criteria
                    - pT > 5 GeV
                    - |eta| < 2.5
                    - Isolation < .4
                    - 3D IP significance < 4
                */
                if (ele->pt() > 5. && std::abs(ele->eta()) < 2.5 && ele_iso < .4 && std::abs(eleSIP3d) < 4.) {
                    ++nEle;
                    eleP4.push_back(ele->p4());
                    eleIso.push_back(ele_iso);
                    eleSIP3D.push_back(eleSIP3d);
                    eleCharge.push_back(ele->charge());
                }
            }
            
            // Loop over jet collection and choose good jets and save them in the event
            for (reco::PFJetCollection::const_iterator jet = jets->begin(); jet != jets->end(); jet++) {
            
                /*
                    Good jet criteria:
                    - pT > 25 GeV
                    - |eta| < 2.5
                */
                if (jet->pt() > 25. && std::abs(jet->eta()) < 2.5)  {
                    ++nJet;
                    jetP4.push_back(jet->p4());
                }
            }
            
            // Now fill all muon, electron and jet variables in the output histogram            
            tree_->Fill();
        } 
        //------------------------------------------------
           
    }
    //=====================================================
    
    
    // The rest of this code are default functions generated automatically
    // when creating the EDAnalyzer within CMSSW. I have not made any changes.


}


// ------------ method called once each job just before starting event loop  ------------
void 
MuonEventsSkimmer::beginJob()
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
MuonEventsSkimmer::endJob() 
{
}

// ------------ method called when starting to processes a run  ------------
void 
MuonEventsSkimmer::beginRun(edm::Run const&, edm::EventSetup const&)
{
}

// ------------ method called when ending the processing of a run  ------------
void 
MuonEventsSkimmer::endRun(edm::Run const&, edm::EventSetup const&)
{
}

// ------------ method called when starting to processes a luminosity block  ------------
void 
MuonEventsSkimmer::beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&)
{
}

// ------------ method called when ending the processing of a luminosity block  ------------
void 
MuonEventsSkimmer::endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&)
{
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void
MuonEventsSkimmer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(MuonEventsSkimmer);
