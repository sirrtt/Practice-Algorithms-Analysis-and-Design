#include <iostream>
#include <bits/stdc++.h>
#include <vector>

using namespace std;

void xuat(vector<vector<string>> kq){
    for(int i=0; i<kq.size(); i++){
        for(int j=0; j<kq[i].size(); j++)
           cout<<kq[i][j]<<" ";
        cout<<endl;
    }
}

int ktra_doixung(string &a, int trai, int phai){
    while(trai<phai)
        if (a[trai++]!=a[phai--]) 
            return 0;
    return 1;
}
    
void doixung(vector<vector<string>> &kq, string &a, vector<string> &ht, int bd){
    if(bd>=a.length()) kq.push_back(ht);
    for(int i=bd; i<a.length(); i++){
        if(ktra_doixung(a, bd, i)==1){
            ht.push_back(a.substr(bd, i-bd+1));
            doixung(kq, a, ht, i+1);
            ht.pop_back();
        }
    }
}

int main(){
    string a;
    cin>>a;
    vector<string> ht;
    vector<vector<string>> kq;
    doixung(kq, a, ht, 0);
    xuat(kq);
}