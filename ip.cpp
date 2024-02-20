#include <iostream>
#include <vector>
#include <bits/stdc++.h>

using namespace std;

void LayIPdung(string ip, vector<string> &kq, string diachiip, int vitri, int dem){
	if(ip.size()==vitri&&dem==4){
		diachiip.pop_back();
		kq.push_back(diachiip);
		return;
	}

	if(ip.size()<vitri+1) return;
	diachiip=diachiip+ip.substr(vitri, 1)+'.';
	LayIPdung(ip, kq, diachiip, vitri+1, dem+1);
	diachiip.erase(diachiip.end()-2, diachiip.end());

	if(ip.size()<vitri+2||ip[vitri]=='0') return;
	diachiip=diachiip+ip.substr(vitri, 2)+'.';
	LayIPdung(ip, kq, diachiip, vitri+2, dem+1);
	diachiip.erase(diachiip.end()-3, diachiip.end());

	if(ip.size()<vitri+3||int(ip.substr(vitri, 3))>255) return;
	diachiip=diachiip+ip.substr(vitri, 3)+'.';
	LayIPdung(ip, kq, diachiip, vitri+3, dem+1);
	diachiip.erase(diachiip.end()-4, diachiip.end());
}

void xuat(vector<string> kq){
    for(int i=0; i<kq.size(); i++) cout<<kq[i]<<endl;
}

int main(){
    vector<string> kq;
	string ip;
    cin>>ip;
	LayIPdung(ip, kq, "", 0, 0);
    xuat(kq);
    return 0;
}
