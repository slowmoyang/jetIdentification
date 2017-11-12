TVectorD solvSysLinEqs(Int_t numVars, Int_t numEqs, Double_t AData[], Double_t bData[]){
    // A
    TMatrixD A(numVars, numEqs);
    A.SetMatrixArray(AData);
        
    // b
    TVectorD b;
    b.Use(numVars, bData);
    
    //
    TDecompLU lu(A);
    Bool_t ok;
    TVectorD x = lu.Solve(b, ok);

    Int_t nr = 0;

    while (!ok){
        lu.SetMatrix(A);
        lu.SetTol(0.1*lu.GetTol());
        if(nr++ > 10) break;
        x = lu.Solve(b, ok);
    }

    if(x.IsValid()){
        cout << "Solved with tol =" << lu.GetTol() << endl;
    }
    else{
        cout << "SOLVING FAILED :(" << endl;
    }
    
    return x;
}
