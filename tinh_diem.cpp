#include <iostream>
#include <vector>
#include <bits/stdc++.h>

using namespace std;

float trb(float *hs, float *kq, int n_){
    float tong=0.0;
    for(int j=0; j<n_; j++) tong=tong+hs[j]*kq[j];
    tong=round(tong*0.1)/10;
    return tong;
}

void how(int i, float *hs, float *kq, float point, int n){
    for(float j=0.25; j<=10.0; j=j+0.25){
        kq[i]=j;
        if(i<n) how(i+1, hs, kq, point, n);
        else{
            if(trb(hs, kq, n)==point){
                for(int b=0; b<n; b++)  cout<<kq[b]<<" ";
                cout<<endl;
            }
            return;
        }
        if(trb(hs, kq, i)>point) break;
    }
    return;
}

void nhap(float *hs, int &n, float &point){
    cin>>n;
    for(int i=0; i<n; i++) cin>>hs[i];
    cin>>point;
}

int main(){
    int n;
    float point;
    float *hs=new float[100];
    float *kq=new float[100];
    nhap(hs, n, point);
    how(0, hs, kq, point, n);
    return 0;
}