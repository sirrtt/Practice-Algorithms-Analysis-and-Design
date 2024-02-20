#include <iostream>
#include <vector>
#include <algorithm>
#include <set>
#include <map>
#include <utility>

using namespace std;

int main(){
    int e, n, x;
    cin>>e>>n;
    string start_point, end_point;
    
    map<string, map<string, int>> graph;
    map<string, map<string, int>>::iterator itr;
    map<string, int>::iterator itr_2;
    for(int i=0; i<e; i++){
        cin>>start_point>>end_point>>x;
        graph[start_point][end_point]=x;
    }

    vector<vector<string>> path;
    for(int i=0; i<n; i++){
        string a;
        vector<string> b;
        while(cin>>a&&a!="."){
            b.push_back(a);
        }
        path.push_back(b);
    }

    for(int i=0; i<n; i++){
        int sum=0;
        for(int j=0; j<path[i].size()-1; j++){
            string x, y;
            x=path[i][j];
            y=path[i][j+1];
            if(graph[x][y]==0){
                cout<<"FALSE"<<endl;
                sum=0;
                break;
            }
            else sum+=graph[x][y];
        }
        if(sum!=0) cout<<sum<<endl;
    }
}