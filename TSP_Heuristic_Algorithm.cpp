#include <iostream>
#include <vector>
#include <algorithm>
#include <set>
#include <map>
#include <utility>

using namespace std;

int check(vector<string> sol, string s){
    for(int i=0; i<sol.size(); i++)  if(sol[i]==s)   return 1;
    return 0;
} 

void output(vector<string> sol){
    for(int i=0; i<sol.size(); i++){
        cout<<sol[i]<<" ";
    }
}

int main(){
    int e, x;
    string s, start_point, end_point;
    cin>>e>>s;

    map<string, map<string, int>> graph;
    for(int i=0; i<e; i++){
        cin>>start_point>>end_point>>x;
        graph[start_point][end_point]=x;
    }

    map<string, map<string, int>>::iterator itr;
    map<string, int>::iterator itr_2;

    vector<string> sol;
    sol.push_back(s);
    vector<string> point;

    for(itr=graph.begin(); itr!=graph.end(); itr++){
        point.push_back(itr->first);
    }
    
    int min=10000;
    string z;
    for(int j=0; j<point.size()-1; j++){
        string y;
        min=10000;
        for(int i=0; i<point.size(); i++){
            if(check(sol, point[i])==0){
                y=point[i];
                if(min>graph[s][y]){
                    min=graph[s][y];
                    z=y;
                }
            }
        }
        sol.push_back(z);
        s=z;
    }
    sol.push_back(sol[0]);
    output(sol);
    return 0;
}