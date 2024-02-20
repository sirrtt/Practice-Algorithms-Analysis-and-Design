#include <iostream>
#include <vector>

using namespace std;

int vach(vector<int> &mang, int &trai, int &phai){
    int p=trai-1;
    int truc=mang[phai];
    for(int i=trai; i<phai; i++)
        if(mang[i]<truc){
            p=p+1;
            swap(mang[p], mang[i]);
        }
    swap(mang[p+1], mang[phai]);
    return p+1;
}

int max_k(vector<int> &mang, int k, int left, int right){
    while(right>=left){
        int truc=vach(mang, left, right);
        if(k<=truc) right=truc-1;
        else left=truc+1;
        if(truc==k) return mang[truc];
    }
    return -1;
}

int main(){
    int n, k, tam;
    cin>>n>>k;
    vector<int> mang;
    for(int i=0; i<n; i++){
        cin>>tam;
        mang.push_back(tam);
    }
    cout<<max_k(mang, n-k, 0, n-1);
    return 0;
}