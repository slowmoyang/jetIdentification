void macro(const char*  input_path,
           const char*  output_path){

    auto input_file = new TFile(input_path, "READ");
    auto input_tree = (TTree*) input_file->Get("jet");

    float image[3267];
    int label[2];
    int partonId, nMatchedJets;
    float pt,eta;

    input_tree->SetBranchAddress("image",           &image);
    input_tree->SetBranchAddress("label",           &label);
    input_tree->SetBranchAddress("partonId",        &partonId);
    input_tree->SetBranchAddress("pt",              &pt);
    input_tree->SetBranchAddress("eta",             &eta);
    input_tree->SetBranchAddress("nMatchedJets",    &nMatchedJets);

    // the total number of quarks
    int tnq = input_tree->Project("quark", "partonId", "partonId!=21");
    cout << "the total number of quark: " << tnq << endl;
    // the total number of gluons
    int tng = input_tree->Project("gluon", "partonId", "partonId==21");
    cout << "the total number of gluon: " << tng << endl;

    auto output_file = new TFile(output_path, "RECREATE");

    float label_weak;

    // weakly supervision classification
    auto output_tree = new TTree("jet", "jet");
    output_tree->SetDirectory(output_file);
    output_tree->Branch("image",           &image,           "image[3267]/F");
    output_tree->Branch("label",           &label,           "label[2]/I");
    output_tree->Branch("label_weak",      &label_weak,      "label_weak/F");
    output_tree->Branch("partonId",        &partonId,        "partonId/I");
    output_tree->Branch("nMatchedJets",    &nMatchedJets,    "nMatchedJets/I");
    output_tree->Branch("pt",              &pt,              "pt/F");
    output_tree->Branch("eta",             &eta,             "eta/F");

    int nGluon = 0;
    int nQuark = 0;
    int input_entries = input_tree->GetEntries();
    for(auto j=0; j < input_entries; ++j){
        input_tree->GetEntry(j);

        if( ( j%1000 == 0 ) or ( j == (input_tree->GetEntries()-1) ) ) {
            cout << j << "th jet: " << partonId << endl;
        }

        // Gluon
        if(partonId==21){
            ++nGluon;
            // gluon output_tree dataset (label_weak=0.7)
            if(nGluon < 0.7*tng){
                label_weak = 0.7;
                output_tree->SetDirectory(output_file);
                output_tree->Fill();
            }
            // gluon output_tree dataset (label_weak=0.3)
            else{
                label_weak = 0.3;
                output_tree->SetDirectory(output_file);
                output_tree->Fill();
            }
        }
        // quark
        else{
            ++nQuark;
            // gluon output_tree dataset (label_weak=0.7)
            if(nQuark < 0.3 * tnq){
                label_weak = 0.7;
                output_tree->SetDirectory(output_file);
                output_tree->Fill();
            }
            // gluon output_tree dataset (label_weak=0.3)
            else{
                label_weak = 0.3;
                output_tree->SetDirectory(output_file);
                output_tree->Fill();
            }


        }
    }
    output_tree->SetDirectory(output_file);
    output_tree->Write();
    output_file->Close();
    cout << "the total number of gluon: " << nGluon << endl;
    cout << "the total number of quark: " << nQuark << endl;
}

