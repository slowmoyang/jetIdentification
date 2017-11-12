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

void exec_clear(){
    system("clear");
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

string combine_path(string const& p1, string const& p2){
    
    stringstream ss;
    
    string separator = "/";
    string last1 = p1.substr(p1.size()-1);

    string output;
    if(last1 == separator){
        ss << p1 << p2;
        output = ss.str();
    }
    else{
        ss << p1 << "/" << p2;
        output = ss.str();
    }
    
    return output;
    
}


string get_dpath(string const& path){
    string separator = "/";
    
    string dpath;
    for(auto it=(path.end()-1); it != (path.begin()-1); --it){
        if(*it==separator){
            dpath = string(path.begin(), it);
            break;
        }
    }
    return dpath;
    
}
