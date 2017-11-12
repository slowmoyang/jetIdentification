#include "utils_sys.cc"
#include "utils_math.cc"


string make_dataset_step1(const char* input_path, string data_dir="$JET/ptJet/data"){
    auto input_file = new TFile(input_path, "READ");
    auto input_tree = (TTree*) input_file->Get("jetAnalyser/jetAnalyser");
    Int_t input_entries = input_tree->GetEntries();

    int partonId, nMatchedJets, nmult, cmult;
    float pt,eta, ptD, axis1, axis2;
    vector<float> *dau_pt = 0;
    vector<float> *dau_deta = 0;
    vector<float> *dau_dphi = 0;
    vector<int> *dau_charge = 0;    

    // For images
    input_tree->SetBranchAddress("dau_pt",          &dau_pt);
    input_tree->SetBranchAddress("dau_deta",        &dau_deta);
    input_tree->SetBranchAddress("dau_dphi",        &dau_dphi);
    input_tree->SetBranchAddress("dau_charge",      &dau_charge);
    // For variabless
    input_tree->SetBranchAddress("nmult",           &nmult);
    input_tree->SetBranchAddress("cmult",           &cmult);
    input_tree->SetBranchAddress("ptD",             &ptD);
    input_tree->SetBranchAddress("axis1",           &axis1);
    input_tree->SetBranchAddress("axis2",           &axis2);
    // For labels
    input_tree->SetBranchAddress("partonId",        &partonId);
    // For cuts
    input_tree->SetBranchAddress("pt",              &pt);
    input_tree->SetBranchAddress("eta",             &eta);
    input_tree->SetBranchAddress("nMatchedJets",    &nMatchedJets);

    ////////////////////////////////////////////////////////////////////////
    // Output files
    ///////////////////////////////////////////////////////////////////////

    string output_dir = combine_path(data_dir, "temp");
    exec_mkdir(output_dir);

    string output_fname = "step1.root";
    string output_path = combine_path(output_dir, output_fname);

    auto output_file = new TFile(output_path.c_str(), "RECREATE");
    auto output_tree = new TTree("jet", "jet");
    output_tree->SetDirectory(output_file);

    // constant
    const int channel = 3;
    const int height = 33;
    const int width = 33;

    const float deta_max = 0.4;
    const float dphi_max = 0.4;

    float image[3267], variables[5];
    int label[2];

    output_tree->Branch("image",           &image,           "image[3267]/F");
    output_tree->Branch("variables",       &variables,       "variables[5]/F");
    output_tree->Branch("label",           &label,           "label[2]/I");
    output_tree->Branch("partonId",        &partonId,        "partonId/I");
    output_tree->Branch("nMatchedJets",    &nMatchedJets,    "nMatchedJets/I");
    output_tree->Branch("pt",              &pt,              "pt/F");
    output_tree->Branch("eta",             &eta,             "eta/F");

    // For image scaling
    float nmultMax = 0.;
    float cmultMax = 0.;
    float axis1Max = 0.;
    float axis2Max= 0;

    for(auto i=0; i<input_entries; ++i){
	input_tree->GetEntry(i);
	if(pt<100) continue;
	if(abs(eta)>2.4) continue;
	if(not((partonId==1)or(partonId==2)or(partonId==3)or(partonId==21))) continue;
	if(axis2==std::numeric_limits<float>::infinity()) continue;
	
	if(nmultMax < nmult) nmultMax = nmult;
	if(cmultMax < cmult) cmultMax = cmult;
	if(axis1Max < axis1) axis1Max = axis1;
	if(axis2Max < axis2) axis2Max = axis2;
    }

    cout << "nmultMax: " << nmultMax << endl
         << "cmultMax: " << cmultMax << endl
         << "axis1Max: " << axis1Max << endl
         << "axis2Max: " << axis2Max << endl;

    const char* scale_para_path = combine_path(output_dir, "vars_scale_para.json").c_str();
    ofstream scale_para_file(scale_para_path);
    scale_para_file << "{" << endl;
    scale_para_file << "     \"nmult\" : "  << nmult << ","  << endl;
    scale_para_file << "     \"cmult\" : "  << cmult << ","  << endl;
    scale_para_file << "     \"axis1\" : "  << axis1 << ","  << endl;
    scale_para_file << "     \"axis2\" : "  << axis2         << endl;
    scale_para_file << "}" << endl;

    for(auto i=0; i<input_entries; ++i){
	input_tree->GetEntry(i);
	if(pt<100) continue;
	if(abs(eta)>2.4) continue;
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
	else{
	    continue;
	}

	// std::numeric_limits<float>::infinity()
	if(axis2==-std::log(0.)) continue;

	//Variables
	variables[0] = cmult / cmultMax;
	variables[1] = nmult / nmultMax;
	variables[2] = ptD;
	variables[3] = axis1 / axis1Max;
	variables[4] = axis2 / axis2Max;
	
	
	// Init iamge array
	for(int i=0; i < 3*33*33; ++i) image[i] = 0;

	for(int d = 0; d < dau_pt->size(); ++d){
	    int w = int( (dau_deta->at(d) + deta_max) / (2*deta_max/33));
	    int h = int( (dau_dphi->at(d) + dphi_max) / (2*dphi_max/33));

	    if((w < 0) or (w >= 33) or (h < 0) or (h>=33)) continue;

	    // charged particle
	    if(dau_charge->at(d)){
		// pT
		image[0 + 33*h + w] += float(dau_pt->at(d));
		// multiplicty
		image[2*33*33 + 33*h + w] += 1.0;
	    }
	    // neutral particle
	    else{
		image[33*33 + 33*h + w] += float(dau_pt->at(d));
	    }
	}

	output_tree->SetDirectory(output_file);
	output_tree->Fill();

    }

    int output_entries = output_tree->GetEntries();

    output_tree->SetDirectory(output_file);
    //output_tree->Print();
    output_file->Write();
    output_file->Close();
    input_file->Close();

    stringstream ss;
    ss << "dataset_" << std::to_string(output_entries);
    string new_dir = combine_path(data_dir, ss.str());
    ss.str("");

    exec_mv(output_dir, new_dir);

    string new_output_path = combine_path(new_dir, output_fname);

    return new_output_path; 
}


string make_dataset_step2(const char* input_path){

    // 
    auto input_file = new TFile(input_path, "READ");
    auto input_key = input_file->GetListOfKeys()->At(0)->GetName();
    auto input_tree = (TTree*) input_file->Get(input_key);
    // input_tree->Print();
    unsigned int input_entries = input_tree->GetEntries();

    float image[3*33*33], variables[5], pt, eta;
    int label[2], partonId, nMatchedJets;

    input_tree->SetBranchAddress("image",           &image);
    input_tree->SetBranchAddress("variables",       &variables);
    input_tree->SetBranchAddress("label",           &label);
    input_tree->SetBranchAddress("partonId",        &partonId);
    input_tree->SetBranchAddress("pt",              &pt);
    input_tree->SetBranchAddress("eta",             &eta);
    input_tree->SetBranchAddress("nMatchedJets",    &nMatchedJets);

    // Find maximum
    float cptMax=0, nptMax=0, cmulMax=0;
    for(auto i=0; i<input_entries; ++i){
        input_tree->GetEntry(i);
    
        float cptLocalMax = *std::max_element(image,            image + 33*33);
        float nptLocalMax = *std::max_element(image + 33*33,    image + 2*33*33);
        float cmulLocalMax = *std::max_element(image + 2*33*33, image + 3*33*33);
    
        if(cptMax < cptLocalMax)   cptMax = cptLocalMax;
        if(nptMax < nptLocalMax)   nptMax = nptLocalMax;
        if(cmulMax < cmulLocalMax) cmulMax = cmulLocalMax;
    }

    cout << "cptMax: "   << cptMax  << endl
         << "nptMax: "   << nptMax  << endl
         << "cmultMAx: " << cmulMax << endl;



    string output_dir = get_dpath(string(input_path));

    string scale_para_path = combine_path(output_dir, "image_scale_parameter.json");
    ofstream scale_para_file(scale_para_path.c_str());
    scale_para_file << "{" << endl;
    scale_para_file << "    \"cpt\" : " << cptMax   << endl;
    scale_para_file << "    \"npt\" : " << nptMax   << endl;
    scale_para_file << "    \"cmul\" : " << cmulMax << endl;
    scale_para_file << "}"                          << endl;
    scale_para_file.close();



    //
    string output_fname = "step2.root";
    string output_path = combine_path(output_dir, output_fname);

    auto output_file = new TFile(output_path.c_str(), "RECREATE");
    auto output_tree = input_tree->CloneTree(0);
    output_tree->SetDirectory(output_file);

    for(auto i=0; i<input_entries; ++i){
	input_tree->GetEntry(i);
	for(auto j=0; j<33*33; ++j){
	    image[j] /= cptMax;
	    image[j+33*33] /= nptMax;
	    image[j+2*33*33] /= cmulMax;
	}
	output_tree->SetDirectory(output_file);
	output_tree->Fill();
    }

    // output_tree->Print();
    output_file->Write();
    output_file->Close();
    input_file->Close();


    return output_path;

}


string split_dataset(const char* input_path){

    auto input_file = new TFile(input_path, "READ");
    auto input_key = input_file->GetListOfKeys()->At(0)->GetName();
    auto input_tree = (TTree*) input_file->Get(input_key);
    // input_tree->Print();
    Int_t input_entries = input_tree->GetEntries();

    float image[3*33*33], variables[5], pt, eta;
    int label[2], partonId, nMatchedJets;

    input_tree->SetBranchAddress("image",           &image);
    input_tree->SetBranchAddress("variables",       &variables);
    input_tree->SetBranchAddress("label",           &label);
    input_tree->SetBranchAddress("partonId",        &partonId);
    input_tree->SetBranchAddress("pt",              &pt);
    input_tree->SetBranchAddress("eta",             &eta);
    input_tree->SetBranchAddress("nMatchedJets",    &nMatchedJets);

    // Output
    auto output_dir = get_dpath(string(input_path));

    int num_training = int(input_entries*0.6);
    int num_validation = int(input_entries*0.2);
    int num_test = input_entries - num_training - num_validation;

    // training
    string training_fname = "training_" + std::to_string(num_training) + ".root";
    string training_path = combine_path(output_dir, training_fname);
    cout << "Training file path: " << training_path << endl;

    auto training_file = new TFile(training_path.c_str(), "RECREATE");
    auto training_tree = input_tree->CloneTree(0);
    training_tree->SetDirectory(training_file);

    // validation file and tree
    string validation_fname = "validation_" + std::to_string(num_validation) + ".root";
    const char* validation_path = combine_path(output_dir, validation_fname).c_str();
    cout << "Validation file path: " << validation_path << endl;

    auto validation_file = new TFile(validation_path, "RECREATE");
    auto validation_tree = input_tree->CloneTree(0);
    validation_tree->SetDirectory(validation_file);

    // test file and tree
    string test_fname = "test_" + std::to_string(num_test) + ".root";
    const char* test_path = combine_path(output_dir, test_fname).c_str();
    cout << "Test file path: " << test_path << endl;

    auto test_file = new TFile(test_path, "RECREATE");
    auto test_tree = input_tree->CloneTree(0);
    test_tree->SetDirectory(test_file);


    // std::vector<int> range;
    //for(int i=0; i<input_entries; ++i)
    //     range.push_back(i);


    // auto rng = std::default_random_engine{};
    // std::shuffle(std::begin(range), std::end(range), rng);


    int training_end = num_training;
    int validation_end = num_training + num_validation;


    int counter = 0;
    //for(auto&& idx : range) {
    for(int i=0; i<input_entries; ++i){
        input_tree->GetEntry(i);

        if(i%10000==0) cout << i << endl;

        if(counter < training_end)
            training_tree->Fill();
        else if(counter < validation_end)
            validation_tree->Fill();
        else
            test_tree->Fill();

        counter++;

    }

    cout << "Here5" << endl;

    cout << "Training set: " << training_tree->GetEntries() << endl;
    cout << "Validation set: " << validation_tree->GetEntries() << endl;
    cout << "Test set: " << test_tree->GetEntries() << endl;

    // training_tree->Print();
    training_file->Write();
    training_file->Close();

    // validation_tree->Print();
    validation_file->Write();
    validation_file->Close();

    // test_tree->Print();
    test_file->Write();
    test_file->Close();

    input_file->Close();

    return training_path;
}


void make_weak_dataset(const char*  input_path, Double_t fRich=0.8, Double_t fPoor=0.2){

    auto input_file = new TFile(input_path, "READ");
    auto input_key = input_file->GetListOfKeys()->At(0)->GetName();
    cout << "Input key: " << input_key << endl;
    auto input_tree = (TTree*) input_file->Get(input_key);
    int input_entries = input_tree->GetEntries();

    float image[3267], variables[5];
    int label[2];
    int partonId, nMatchedJets;
    float pt,eta;

    input_tree->SetBranchAddress("image",           &image);
    input_tree->SetBranchAddress("variables",       &variables);
    input_tree->SetBranchAddress("label",           &label);
    input_tree->SetBranchAddress("partonId",        &partonId);
    input_tree->SetBranchAddress("pt",              &pt);
    input_tree->SetBranchAddress("eta",             &eta);
    input_tree->SetBranchAddress("nMatchedJets",    &nMatchedJets);

    /*************************************
    * Q : numQuark / G: numGluon
    * R : fRich / P : fPoor
    *
    * 1) Q = rQ + pQ
    *
    * 2) G = rG + pG
    * 
    * 3-1) R = rG / (rQ + rG)
    * 3-2) (1/R - 1)rG - rQ = 0
    *
    * 4-1) P = pG / (pQ + pG)
    * 4-2) (
    *
    * x = [rQ, pQ, rG, pG]
    * 
    *
    * [  1,  1,     0,     0 ] [ rQ ]   [ Q ]
    * [  0,  0,     1,     1 ] [ pQ ] = [ G ]
    * [ -1,  0, 1/R-1,     0 ] [ rG ]   [ 0 ]
    * [  0, -1,     0, 1/P-1 ] [ pG ]   [ 0 ]
    ******************************************/

    // the total number of quarks
    int numQuarks = input_tree->Project("quark", "partonId", "partonId!=21");
    cout << "the total number of quark: " << numQuarks << endl;
    // the total number of gluons
    int numGluons = input_tree->Project("gluon", "partonId", "partonId==21");
    cout << "the total number of gluon: " << numGluons << endl;

    Int_t numVars = 4;
    Int_t numEqs = 4;

    Double_t AData[] = {1.,  1.,    0.,   0.,
                        0.,  0.,     1,    1,
                       -1.,  0., 1/fRich-1,   0.,
                        0., -1.,    0., 1/fPoor-1};

    Double_t bData[] = {Double_t(numQuarks), Double_t(numGluons), 0., 0.};

    auto x = solvSysLinEqs(numVars, numEqs, AData, bData);

    Double_t numRichQuarks = x[0];
    Double_t numRichGluons = x[2];

    ///////////////////////////////////////////////////////////////////

    
    auto output_dir = get_dpath(string(input_path));
    string output_fname = "training_weak_" + std::to_string(input_entries) + ".root";
    const char* output_path = combine_path(output_dir, output_fname).c_str();

    auto output_file = new TFile(output_path, "RECREATE");
    auto output_tree = input_tree->CloneTree(0);
    output_tree->SetDirectory(output_file);

    float label_weak;
    output_tree->Branch("label_weak",      &label_weak,      "label_weak/F");

    int gluonCount = 0;
    int quarkCount = 0;
    for(auto j=0; j < input_entries; ++j){
        input_tree->GetEntry(j);
        // Gluon
        if(partonId==21){
            label_weak = gluonCount < numRichGluons ? fRich : fPoor;
            output_tree->SetDirectory(output_file);
            output_tree->Fill();
            gluonCount++;
        }
        // quark
        else{
            label_weak = quarkCount < numRichQuarks ? fRich : fPoor;
            output_tree->SetDirectory(output_file);
            output_tree->Fill();
            quarkCount++;
        }
    }

    output_tree->SetDirectory(output_file);
    output_tree->Write();

    output_file->Close();

}



void macro(const char* input_path){

    // scale the discriminating variables and make jet images.
    cout << "******************** Phase 1 ********************" << endl;
    string path1 = make_dataset_step1(input_path);
    cout << "*************************************************" << endl;


    cout << "******************** Phase 2 ********************" << endl;
    string path2 = make_dataset_step2(path1.c_str());
    cout << "*************************************************" << endl;

    cout << "******************** Phase 3 ********************" << endl;
    string path3 = split_dataset(path2.c_str());
    cout << "*************************************************" << endl;

    cout << "******************** Phase 4 ********************" << endl;
    cout << "Input file: " << path3 << endl;
    make_weak_dataset(path3.c_str());
    cout << "*************************************************" << endl;

}
