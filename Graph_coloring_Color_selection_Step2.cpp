#include <iostream>
#include <vector>
#include <algorithm>
#include <set>
#include <map>
#include <utility>

using namespace std;

int check_adj(string c, map<string, set<string>> adj_list, map<string, int> change_to_color){
    set<string> point=adj_list[c]; //tạo set mới truy cập nội dung từ map chứa nội dung c
    // vector<string> test;
    // cout<<c;
    for(string j:point) //danh sách các đỉnh kề của c
        if(change_to_color[c]==change_to_color[j])
            return 0; //sai
    return 1; //đúng
}

int check(int x, set<string> point, map<string, int> change_to_color){
    for(string m:point){
        if(change_to_color[m]==x) return 1;
    }
    return 0;
}

int choose_color(string c, vector<int> colors_used_2, map<string, set<string>> adj_list, map<string, int> change_to_color){
    int flag, rs=100000;
    for(int i=0; i<colors_used_2.size(); i++){
        set<string> point=adj_list[c];
        if(check(colors_used_2[i], point, change_to_color)==0) return colors_used_2[i];
    }
    int p=0;
    // cout<<"I'm here, chon mau chua co"<<endl;
    for(int i=0; i<colors_used_2.size(); i++){
        if(p==colors_used_2[i]) p++;
        else break;
    }
    return p;
}

int main(){
    int v, e, n, c;
    string input, start_point, end_point;
    vector<string> list_vertices;
    map<string, set<string>> adj_list;
    vector<int> colors;
    vector<string> vertices_choose_colors;
    set<int> colors_used;
    vector<int> colors_used_2; //vector chứa phần tử của color_used để dễ truy cập
    map<string, int> change_to_color; //là dictionary màu của các điểm từ colors

    cin>>v>>e>>n;
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
        cin>>c;
        colors.push_back(c);
        change_to_color[list_vertices[i]]=c;
    }
    for(int i=0; i<n; i++){
        cin>>input;
        vertices_choose_colors.push_back(input);
    }
    for(int i=0; i<v; i++){
        if(colors[i]!=-1){
            colors_used.insert(colors[i]);
        }
    }
    for(int i:colors_used){ 
        colors_used_2.push_back(i); //i là nội dung đại diện cho phần tử trong set color_used, vòng for là cho chạy toàn bộ trong color_used
    }

    for(int i=0; i<n; i++){
        int a=change_to_color[vertices_choose_colors[i]];
        if(a==-1) cout<<choose_color(vertices_choose_colors[i], colors_used_2, adj_list, change_to_color)<<endl;
        else{
            int b;
            b=check_adj(vertices_choose_colors[i], adj_list, change_to_color);
            if(b==0) cout<<choose_color(vertices_choose_colors[i], colors_used_2, adj_list, change_to_color)<<endl;
            else cout<<"TRUE"<<endl;
        }
    }
    return 0;
}