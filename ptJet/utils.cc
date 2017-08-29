void exec_mkdir(string const& dpath){
    std::stringstream ss;
    ss << "mkdir" << " " << dpath;
    const char* mkdir_command = ss.str().c_str();
    if(system(mkdir_command)==0){
        cout << dpath << endl;
    }
}


void exec_mv(string const& old_path, string const& new_path){

    std::stringstream ss;
    ss << "mv" << " " << old_path << " " << new_path;
    const char* com = ss.str().c_str();
    if(system(com)==0){
        cout << ss.str() << endl;
    }
    else{
        cout << "FAILED! :(" << endl;
    }
}

string join_path(int n, ...){
    
    stringstream ss;
    
    va_list arglist;
    char* arg;
    va_start(arglist, n);
    for(int i=0; i<n; i++){
        arg=va_arg(arglist, char*);
        if(i<(n-1)){
            ss << string(arg) << "/";
        }
        else if(i==(n-1)){
            ss << string(arg);
        }
    }
    string output = ss.str();
    return output;
}


string comb_dnf(string const& d, string const& f){
    stringstream ss;
    ss << d << "/" << f;
    string s = ss.str();
    return s;
}
