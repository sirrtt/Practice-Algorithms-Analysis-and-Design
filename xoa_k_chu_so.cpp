#include <iostream>
#include <cmath>

using namespace std;

bool check_ascending(string a){
    int count=0;
    for(int i=0; i<a.size(); i++){
        if(a[i]<=a[i+1]) count++;
    }
    if(count==a.size()) return true;
    return false;
}

void xoa(string a, int k){
    if(a.size()<k){
        cout<<"error";
        return;
    }
    for(int i=0; i<k; i++){
        if(check_ascending(a)){
            a.pop_back();
        }
        else{
            for(int i=0; i<a.size(); i++){
                if(a[i]>a[i+1]){
                    a.erase(a.begin()+i);
                    break;
                }
            }
        }
    }
    if(a.size()==0) a="0";
    int result=stoi(a);
    cout<<result;
}

int main(){
    string a;
    int k;
    cin>>a>>k;
    xoa(a, k);
    return 0;
}