#include "utils.cc"

void make_wsc_dataset(const char*  input_path){

    auto input_file = new TFile(input_path, "READ");
    auto input_tree = (TTree*) input_file->Get("jet");
    int input_entries = input_tree->GetEntries();

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


    ///////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////


    std::string temp_dir ="$JET/ptJet/data/temp";
    exec_mkdir(temp_dir);

    std::stringstream ss;
    ss << temp_dir << "/" << "dataset_wsc.root";
    const char* temp_path = ss.str().c_str();
    ss.str("");

    auto output_file = new TFile(temp_path, "RECREATE");

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

    int output_entries = output_tree->GetEntries();

    output_tree->SetDirectory(output_file);
    output_tree->Write();

    output_file->Close();

    cout << "the total number of gluon: " << nGluon << endl;
    cout << "the total number of quark: " << nQuark << endl;

    // output_dir
    ss << "$JET/ptJet/data/dataset_wsc_image_" << std::to_string(output_entries);
    std::string output_dir = ss.str();
    ss.str("");

    // rename directory
    exec_mv(temp_dir, output_dir);
}



void split_wsc_dataset(const char* input_path,
                       string const& output_dir){


    auto input_file = new TFile(input_path, "READ");
    auto input_tree = (TTree*) input_file->Get("jet");

    float image[3267];
    int label[2];
    int partonId, nMatchedJets;
    float label_weak, pt, eta;

    input_tree->SetBranchAddress("image",           &image);
    input_tree->SetBranchAddress("label",           &label);
    input_tree->SetBranchAddress("label_weak",      &label_weak);
    input_tree->SetBranchAddress("partonId",        &partonId);
    input_tree->SetBranchAddress("pt",              &pt);
    input_tree->SetBranchAddress("eta",             &eta);
    input_tree->SetBranchAddress("nMatchedJets",    &nMatchedJets);

    stringstream ss;

    string temp_training_path = comb_dnf(output_dir, "training.root");
    string temp_validation_path = comb_dnf(output_dir, "validation.root");
    string temp_test_path = comb_dnf(output_dir, "test.root");

    auto training_file = new TFile(temp_training_path.c_str(), "RECREATE");
    auto training_tree = new TTree("jet", "jet");
    training_tree->SetDirectory(training_file);
    training_tree->Branch("image",           &image,           "image[3267]/F");
    training_tree->Branch("label",           &label,           "label[2]/I");
    training_tree->Branch("label_weak",      &label_weak,      "label_weak/F");
    training_tree->Branch("partonId",        &partonId,        "partonId/I");
    training_tree->Branch("nMatchedJets",    &nMatchedJets,    "nMatchedJets/I");
    training_tree->Branch("pt",              &pt,              "pt/F");
    training_tree->Branch("eta",             &eta,             "eta/F");

    auto validation_file = new TFile(temp_validation_path.c_str(), "RECREATE");
    auto validation_tree = new TTree("jet", "jet");
    validation_tree->SetDirectory(validation_file);
    validation_tree->Branch("image",           &image,           "image[3267]/F");
    validation_tree->Branch("label",           &label,           "label[2]/I");
    validation_tree->Branch("label_weak",      &label_weak,      "label_weak/F");
    validation_tree->Branch("partonId",        &partonId,        "partonId/I");
    validation_tree->Branch("nMatchedJets",    &nMatchedJets,    "nMatchedJets/I");
    validation_tree->Branch("pt",              &pt,              "pt/F");
    validation_tree->Branch("eta",             &eta,             "eta/F");

    auto test_file = new TFile(temp_test_path.c_str(), "RECREATE");
    auto test_tree = new TTree("jet", "jet");
    test_tree->SetDirectory(test_file);
    test_tree->Branch("image",           &image,           "image[3267]/F");
    test_tree->Branch("label",           &label,           "label[2]/I");
    test_tree->Branch("label_weak",      &label_weak,      "label_weak/F");
    test_tree->Branch("partonId",        &partonId,        "partonId/I");
    test_tree->Branch("nMatchedJets",    &nMatchedJets,    "nMatchedJets/I");
    test_tree->Branch("pt",              &pt,              "pt/F");
    test_tree->Branch("eta",             &eta,             "eta/F");


    int input_entries = input_tree->GetEntries();
    int tr_end = int(0.6*input_entries);
    int val_end = int(0.8*input_entries);

    for(auto j=0; j < input_entries; ++j){

        if( j <= tr_end){
            training_tree->SetDirectory(training_file);
            training_tree->Fill();
        }
        else if( (j>tr_end)and(j<=val_end)){
            validation_tree->SetDirectory(validation_file);
            validation_tree->Fill();
        }
        else if( j > val_end){
            test_tree->SetDirectory(test_file);
            test_tree->Fill();
        }
        else{
            cout << "STH WRONG" << endl;
        }
    }

    int tr_entries = training_tree->GetEntries();
    int val_entries = validation_tree->GetEntries();
    int test_entries = test_tree->GetEntries();

    training_tree->SetDirectory(training_file);
    training_tree->Write();
    training_file->Close();

    validation_tree->SetDirectory(validation_file);
    validation_tree->Write();
    validation_file->Close();

    test_tree->SetDirectory(test_file);
    test_tree->Write();
    test_file->Close();
 
    /*
    // Rename output files
    ss << output_dir << "/" << "training_set_" << std::to_string(tr_entries) << ".root";
    string training_path = ss.str();
    ss.str("");

    ss << output_dir << "/" << "validation_set_" << std::to_string(val_entries) << ".root";
    auto validation_path = ss.str();
    ss.str("");

    ss << output_dir << "/" << "test_set_" << std::to_string(test_entries) << ".root";
    auto test_path = ss.str();
    ss.str("");

    exec_mv(temp_training_path, training_path);
    exec_mv(temp_validation_path, validation_path);
    exec_mv(temp_test_path, test_path);
    */
}
