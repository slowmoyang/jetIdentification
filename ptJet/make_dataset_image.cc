#include "utils.cc"


void make_dataset(const char* input_path){

    auto input_file = new TFile(input_path, "READ");
    auto input_jet = (TTree*) input_file->Get("jetAnalyser/jetAnalyser");
    const int input_entries = input_jet->GetEntries();

    int partonId, nMatchedJets;
    float pt,eta;
    std::vector<float> *dau_pt = 0;
    std::vector<float> *dau_deta = 0;
    std::vector<float> *dau_dphi = 0;
    std::vector<int> *dau_charge = 0;

    input_jet->SetBranchAddress("partonId",        &partonId);
    input_jet->SetBranchAddress("pt",              &pt);
    input_jet->SetBranchAddress("eta",             &eta);
    input_jet->SetBranchAddress("nMatchedJets",    &nMatchedJets);
    input_jet->SetBranchAddress("dau_pt",          &dau_pt);
    input_jet->SetBranchAddress("dau_deta",        &dau_deta);
    input_jet->SetBranchAddress("dau_dphi",        &dau_dphi);
    input_jet->SetBranchAddress("dau_charge",      &dau_charge);


    std::string temp_dir = "$JET/ptJet/data/temp";
    exec_mkdir(temp_dir);

    std::stringstream ss;
    ss << string(temp_dir) << "/" << "dataset.root";
    const char* temp_path = ss.str().c_str();
    ss.str("");

    auto output_file = new TFile(temp_path, "RECREATE");
    auto output_tree = new TTree("jet", "jet");
    output_tree->SetDirectory(output_file);

    // constant
    const int channel = 3;
    const int height = 33;
    const int width = 33;

    const float deta_max = 0.4;
    const float dphi_max = 0.4;

    float image[3267];
    int label[2];

    output_tree->Branch("image",           &image,           "image[3267]/F");
    output_tree->Branch("label",           &label,           "label[2]/I");
    output_tree->Branch("partonId",        &partonId,        "partonId/I");
    output_tree->Branch("nMatchedJets",    &nMatchedJets,    "nMatchedJets/I");
    output_tree->Branch("pt",              &pt,              "pt/F");
    output_tree->Branch("eta",             &eta,             "eta/F");

    for(auto j=0; j < input_entries; ++j){
        input_jet->GetEntry(j);
        if(j%1000==0)
            cout << j << "th jet" << endl;
        // Gluon
        if(partonId == 21){
            label[0] = 0;
            label[1] = 1;
        }
        // Light Quark
        else if(( partonId==1) or (partonId==2) or (partonId==3) ){
            label[0] = 1;
            label[1] = 0;
        }
        // heavy quark and anti-matter
        else{
            continue;
        }

        // Image
        for(int i=0; i < 3267; ++i){
            image[i] = 0;
        }

        for(int d = 0; d < dau_pt->size(); ++d){
            int w = int( (dau_deta->at(d) + deta_max) / (2*deta_max/33));
            int h = int( (dau_dphi->at(d) + dphi_max) / (2*dphi_max/33));

            if((w < 0) or (w > 32) or (h < 0) or (h>32)){
                continue;
            }

            // charged particle
            if(dau_charge->at(d)){
                // pT
                image[0 + 33*h + w] += float(dau_pt->at(d));
                // multiplicty
                image[2178 + 33*h + w] += 1.0;
            }
            // neutral particle
            else{
                image[1089 + 33*h + w] += float(dau_pt->at(d));
            }
        }

        output_tree->SetDirectory(output_file);
        output_tree->Fill();
    }

    int output_entries = output_tree->GetEntries();

    output_tree->SetDirectory(output_file);
    output_tree->Write();
    output_file->Close();

    // output_dir
    ss << "$JET/ptJet/data/dataset_image_" << std::to_string(output_entries);
    std::string output_dir = ss.str();
    ss.str("");

    // rename directory
    exec_mv(temp_dir, output_dir);
}
