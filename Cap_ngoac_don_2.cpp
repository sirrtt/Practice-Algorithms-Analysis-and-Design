#include <iostream>
#include <bits/stdc++.h>

using namespace std;

void ngoac_don(int n, vector<string> &a, string s, int dong, int mo){
	if(dong<mo){
		ngoac_don(n, a, s+")", dong+1, mo);
	}
    if(mo==n&&dong==n){
		a.push_back(s);
		return;
	}
	if(mo<n){
		ngoac_don(n, a, s+"(", dong, mo+1);
	}
}

int main(){
	int n;
    cin>>n;
	vector<string> a;
	ngoac_don(n, a, "", 0, 0);
	for(auto s:a) cout<<s<<endl;
	return 0;
}
