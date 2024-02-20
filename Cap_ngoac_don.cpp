#include <iostream>
#include <bits/stdc++.h>

using namespace std;

vector<string> ngoac_don(int n){
	vector<string> k, q;
    if(n==2){
		k.push_back("(())");
		k.push_back("()()");
		return k;
	}
	if(n==1){
		k.push_back("()");
		return k;
	}
	k=ngoac_don(n-1); 
	for(int j=0; j<k.size(); j++){
		string a="("; 
        string b="("; 
        string c="(";
		a=a+k[j]+")";
		b=b+")"+k[j];
		c=k[j]+c+")";
		q.push_back(a);
		q.push_back(b);
		q.push_back(c);
	}
	q.pop_back();
	return q;
}

int main(){
    int n;
    cin>>n;
	vector<string> a=ngoac_don(n); 
	for(int i=0; i<a.size(); i++){
		cout<<a[i]<<endl; 
	}
}


