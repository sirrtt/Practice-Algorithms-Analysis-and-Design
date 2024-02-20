#include <iostream>
#include <vector>
#include <algorithm>
#include <set>
#include <map>
#include <utility>

using namespace std;

void output(vector<int> sol){
    for(int i=0; i<sol.size(); i++){
        cout<<sol[i]<<" ";
    }
}

int main(){
    int v, e;
    string a, b, c;
    cin>>v>>e;

    vector<string> heads;
    map<string, set<string>> point;
    map<string, set<string>>::iterator itr;
    set<string>::iterator itr_2;

    for(int i=0; i<v; i++){
        cin>>c;
        heads.push_back(c);
    }
    for(int i=0; i<e; i++){
        cin>>a>>b;
        point[a].insert(b);
        point[b].insert(a);
    }

    vector<int> sol;
    int count=0;

    for(int i=0; i<v; i++){
        count=0;
        for(itr=point.begin(); itr!=point.end(); itr++){
            if(itr->first==heads[i]){
                sol.push_back(itr->second.size());
                count++;
            }
        }
        if(count==0) sol.push_back(0);
    }

    // for(int i=0; i<v; i++)
    //     cout<<(point[heads[i]]).size()<<" ";
    output(sol);
    return 0;
}