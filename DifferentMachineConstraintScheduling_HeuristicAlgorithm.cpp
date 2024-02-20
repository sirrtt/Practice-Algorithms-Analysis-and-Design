#include <iostream>
#include <vector>
#include <algorithm>

using namespace std;

int check_min(vector<int> totaltime){
    int a=totaltime[0], pos=0;
    for(int i=0; i<totaltime.size(); i++){
        if(a>totaltime[i]){
            a=totaltime[i];
            pos=i;
        }
    }
    return pos;
}

int check_max(vector<vector<int>> jobs, int i, int m){
    int a=jobs[0][i], pos=0;
    for(int k=0; k<m; k++){
        if(a<jobs[k][i]){
            a=jobs[k][i];
            pos=i;
        }
    }
    return a;
}

void greedy(vector<int> totaltime, vector<vector<int>> jobs, vector<int> sol, int n, int m){
    int pos=0;
    for(int i=0; i<n; i++){
        vector<int> test;
        int sum1=0;
        for(int j=0; j<m; j++){
            if(jobs[j][i]==-1)  sum1=check_max(jobs, i, m)+1;
            else sum1=totaltime[j]+jobs[j][i];
            test.push_back(sum1);
        }
        pos=check_min(test);
        sol[i]=pos;
        totaltime[pos]+=jobs[pos][i];
    }
    for(int i=0; i<n; i++){
        cout<<sol[i]<<" ";
    }
}

int main(){
    int n, m, t;
    cin>>n>>m;
    vector<int> totaltime(m, 0);
    vector<vector<int>> jobs;
    vector<int> sol(n, 0);
    for(int i=0; i<m; i++){
        vector<int> test;
        for(int j=0; j<n; j++){
            cin>>t;
            test.push_back(t);
        }
        jobs.push_back(test);
    }
    greedy(totaltime, jobs, sol, n, m);
    return 0;
}