#include <iostream>
#include <vector>
#include <algorithm>
#include <set>
#include <map>

using namespace std;

int main(){
    int e;
    string start_point, end_point;
    cin>>e;

    map<string, set<string>> graph;
    for(int i=0; i<e; i++){
        cin>>start_point>>end_point;
        graph[start_point].insert(end_point);
    }

    map<string, set<string>>::iterator itr;
    set<string>::iterator itr_2;
    int d=0;
    for(itr=graph.begin(); itr!=graph.end(); itr++) d++;
    for(itr=graph.begin(); itr!=graph.end(); itr++){
        if(d!=itr->second.size()+1){
            cout<<"FALSE";
            return 0;
        }
    }
    cout<<"TRUE";
    return 0;
}