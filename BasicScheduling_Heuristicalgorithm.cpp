#include <iostream>
#include <vector>
#include <algorithm>

using namespace std;

int check_min(vector<int> totaltime){
    int a=totaltime[0], pos=0;
    for(int i=1; i<totaltime.size(); i++){
        if(a>totaltime[i]){
            a=totaltime[i];
            pos=i;
        }
    }
    return pos;
}

void greedy(vector<int> totaltime, vector<pair<int, int>> jobs, vector<int> sol, int n, int m){
    int pos;
    for(int i=0; i<n; i++){
        pos=check_min(totaltime);
        sol[jobs[i].second]=pos;
        totaltime[pos]+=jobs[i].first;
    }
    for(int i=0; i<n; i++){
        cout<<sol[i]<<" ";
    }
}

int main(){
    int n, m, t;
    cin>>n>>m;
    vector<int> totaltime(m, 0);
    vector<pair<int, int>> jobs;
    vector<int> sol(n, 0);
    for(int i=0; i<n; i++){
        cin>>t;
        jobs.push_back(make_pair(t, i));
    }
    sort(jobs.begin(), jobs.end());
    reverse(jobs.begin(), jobs.end());
    greedy(totaltime, jobs, sol, n, m);
    return 0;
}