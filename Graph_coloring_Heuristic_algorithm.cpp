#include <iostream>
#include <vector>
#include <algorithm>
#include <set>
#include <map>
#include <utility>

using namespace std;

int check(int x, set<string> point, map<string, int> change_to_color){
    for(string m:point){
        if(change_to_color[m]==x) return 1;
    }
    return 0;
}

int main(){
    int v, e;
    string input, start_point, end_point;
    vector<string> list_vertices;
    map<string, set<string>> adj_list;
    set<int> colors_used;
    vector<int> colors_used_2;
    map<string, int> change_to_color;

    cin>>v>>e;    
    for(int i=0; i<v; i++){
        cin>>input;
        list_vertices.push_back(input);
    }
    for(int i=0; i<e; i++){
        cin>>start_point>>end_point;
        adj_list[start_point].insert(end_point);
        adj_list[end_point].insert(start_point);
    }
    for(int i=0; i<v; i++){
        change_to_color[list_vertices[i]]=-1;
    }
    
    for(int i=0; i<v; i++){
        int y=0;
        for(int j=0; j<colors_used.size(); j++){
            set<string> point=adj_list[list_vertices[i]];
            if(check(colors_used_2[j], point, change_to_color)==0){
                cout<<colors_used_2[j]<<" ";
                change_to_color[list_vertices[i]]=colors_used_2[j];
                y++;
                break;
            }
        }
        if(y!=0) continue;
        else{
            int p=0;
            for(int k=0; k<colors_used.size(); k++){
                if(p==colors_used_2[k]) p++;
                else break;
            }
            cout<<p<<" ";
            colors_used.insert(p);
            colors_used_2.clear();
            for(int r:colors_used){ 
                colors_used_2.push_back(r); 
            }
            change_to_color[list_vertices[i]]=p;
        }
    }
}