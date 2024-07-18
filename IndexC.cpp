#include <queue>
#include <fstream>
#include <iostream>
#include <stdlib.h>
#include <unordered_map>
#include <string.h>
#include <map>
#include <numeric>
#include <utility>
#include <time.h>
#include <deque>
#include <set>
#include <algorithm>
#include <bitset>
#include <vector>
#include <omp.h>

#define MAXINT ((unsigned) 4294967295)

using namespace std;

#define MAXD 120
unsigned MAXDIS, MAXMOV, MASK;
vector<vector<unsigned> > con, conD;
vector<unsigned> *label, *delete_labels, *clab, *Pla;
vector<int> *pos, *cpos;
vector<pair<int, int> > v2degree;
vector<int> v2p, p2v, flg, vaff; // 删了label的顶点

vector<pair<int, int> > DyEdges;
// unordered_map<int, vector<>>

int totalV = 0, threads = 1, Dcnt = 3, dv = MAXD, minTgt;

double tt1 = 0, tt2 = 0, tt3 = 0, tt4 = 0;

void Parameter(){
    MAXDIS = 2; MAXMOV = 1;
    while( MAXINT / (totalV * 2) >= MAXDIS ) {
        MAXDIS *= 2;
        ++MAXMOV;
    }
    MASK = MAXDIS - 1;
}


void GraphInitial(string filename){
    string s;
    const char *filepath = filename.c_str();
    ifstream infile;

    infile.open(filepath);
    if(!infile.is_open()){
        cout<<"No such file!"<<endl;
        exit(-1);
    }

    long xx = 0;

    while(getline(infile, s)){
        char* strc = new char[strlen(s.c_str())+1];
        strcpy(strc, s.c_str());
        char* s1 = strtok(strc," ");
        
        if (xx == 0){
            totalV = atoi(s1);
            con.resize(totalV);
            conD.resize(totalV);
            Parameter();
        }else{
            while(s1){
                int va = xx - 1, vb = atoi(s1); // 
                conD[va].push_back(vb);
                s1=strtok(NULL," ");
            }
        }

        xx += 1;
        
        delete s1, strc;
    }

    infile.close();

    for( int i = 0; i < totalV; ++i )
		v2degree.push_back(make_pair(conD[i].size(), -i)); //  

	sort(v2degree.rbegin(), v2degree.rend());

    v2p.resize(totalV, -1);
    p2v.resize(totalV, -1);

    for (int i=0; i<v2degree.size(); ++i){
        v2p[-v2degree[i].second] = i;
        p2v[i] = -v2degree[i].second; // new 2 old
    }

    for (int i=0; i<v2degree.size(); ++i){
        int ovid = p2v[i];

        vector<unsigned>& local = conD[ovid];

        for( int p = 0; p < local.size(); ++p ) {
            int jj = local[p]; 
            int j = v2p[jj];

            con[j].push_back(i);
        }

        vector<unsigned>().swap(local);
    }

    cout<<con[0].size()<<endl;
}


bool can_update(int v, int dis, char *nowdis) {
    for( int i = 0; i < (int) label[v].size(); ++i ) {
        int w = label[v][i]>>MAXMOV, d = label[v][i]&MASK;
        if( nowdis[w] >= 0 && nowdis[w] + d <= dis ) return false;
    }
    return true;
}

bool can_update_add(int v, int dis, char *nowdis) {
    for( int i = 0; i < pos[v][dis]; ++i ) {
        int w = label[v][i]>>MAXMOV, d = label[v][i]&MASK;
        if( nowdis[w] >= 0 && nowdis[w] + d <= dis ) return false;
    }

    for( int i = 0; i < clab[v].size(); ++i ) {
        int w = clab[v][i]>>MAXMOV, d = clab[v][i]&MASK;
        if( nowdis[w] >= 0 && nowdis[w] + d <= dis ) return false;
    }
    
    return true;
}

bool can_update_delete(int v, int dis, char *nowdis) {

    int pla1 = dis<pos[v].size()? pos[v][dis]:label[v].size();
    for( int i = 0; i < pla1; ++i ) {
        int w = label[v][i]>>MAXMOV, d = label[v][i]&MASK;
        if( nowdis[w] >= 0 && nowdis[w] + d <= dis ) return false;
    }

    return true;
}


void IndexBuild(){

    omp_set_num_threads(threads);

    pos = new vector<int>[totalV];

    label = new vector<unsigned>[totalV]; // unsigned 整合了 id+dis

    for(int i = 0; i < totalV; ++i){
        label[i].push_back((((unsigned)i)<<MAXMOV) | 0);
    
        for( int j = 0; j < con[i].size() && con[i][j] < i; ++j ){
            label[i].push_back((((unsigned) con[i][j]) << MAXMOV) | 1);
        }

        pos[i].push_back(1), pos[i].push_back(label[i].size());
    }

    int dis = 2;
    for( long long cnt = 1; cnt && dis <= MAXDIS; ++dis ){
        cnt = 0;
        vector<unsigned> *label_new = new vector<unsigned>[totalV];
        #pragma omp parallel
        {
            int pid = omp_get_thread_num(), np = omp_get_num_threads();
            long long local_cnt = 0;
            unsigned char *used = new unsigned char[totalV/8+1];
            memset( used, 0, sizeof(unsigned char) * (totalV/8+1) );
            vector<int> cand;
            
            char *nowdis = new char[totalV];
            memset( nowdis, -1, sizeof(char) * totalV);

            for( int u = pid; u < totalV; u += np ){
                cand.clear();
                
                for( int i = 0; i < con[u].size(); ++i ){
                    int w = con[u][i];

                    for( int j = pos[w][dis-2]; j < pos[w][dis-1]; ++j ) {
                        int v = label[w][j] >> MAXMOV;
                        if( v >= u ) break;

                        if( !(used[v/8]&(1<<(v%8))) ) {
                            used[v/8] |= (1<<(v%8)), cand.push_back(v);
                        }
                    }
                }

                int n_cand = 0;
                for( int i = 0; i < (int) label[u].size(); ++i ) 
                    nowdis[label[u][i] >> MAXMOV] = label[u][i] & MASK;
                
                for( int i = 0; i < (int) cand.size(); ++i ) {
                    used[cand[i]/8] = 0;
                    if( can_update(cand[i], dis, nowdis) ) 
                        cand[n_cand++] = cand[i]; 
                }

                cand.resize(n_cand);
                sort(cand.begin(), cand.end());

                for( int i = 0; i < (int) cand.size(); ++i ) {
                    label_new[u].push_back((((unsigned)cand[i])<<MAXMOV) | (unsigned) dis), ++local_cnt;
                }
                for( int i = 0; i < (int) label[u].size(); ++i ) 
                    nowdis[label[u][i]>>MAXMOV] = -1;
            }

            #pragma omp critical
            {
                cnt += local_cnt;
            }

            delete[] used; delete[] nowdis;
        }

        #	pragma omp parallel
        {
            int pid = omp_get_thread_num(), np = omp_get_num_threads();
            for( int u = pid; u < totalV; u += np ){
                
                label[u].insert(label[u].end(), label_new[u].begin(), label_new[u].end());

                vector<unsigned>(label[u]).swap(label[u]);

                vector<unsigned>().swap(label_new[u]);

                if (cnt > 0) pos[u].push_back(label[u].size());
            }
        }
        
        cout<<"Distance: "<<dis<<"   Cnt: "<<cnt<<endl;
        delete[] label_new;
    }

}


void IndexSave(string path){
    FILE *fout = fopen( path.c_str(), "wb" );
    
    fwrite(&totalV, sizeof(int), 1, fout); // 顶点数

	for(int i = 0; i < totalV; ++i) {
		int len = (int) label[i].size();
		fwrite(&len, sizeof(int), 1, fout); // 每个顶点的 labels 的 数量
	}

    unsigned *s = new unsigned[totalV]; // 每个顶点的label总数是有限的

	for(int i = 0; i < totalV; ++i) {
		int len = (int) label[i].size();
		for(int j = 0; j < len; ++j) 
			s[j] = label[i][j];
		fwrite(s, sizeof(unsigned), len, fout); // 写入每个顶点的labels
	}

    // ===========================
    for(int i = 0; i < totalV; ++i) {
		int len = (int) pos[i].size();
		fwrite(&len, sizeof(int), 1, fout); // 每个顶点的 pos 的 数量
	}

    for(int i = 0; i < totalV; ++i) {
        int len = (int) pos[i].size();
        for(int j = 0; j < len; ++j) 
			s[j] = pos[i][j];
        fwrite(s, sizeof(unsigned), len, fout); // 写入每个顶点的labels
    }

    delete[] s;

    fwrite(&MAXMOV, sizeof(int), 1, fout);
    fclose(fout);

}


void IndexLoad(string path){
    FILE* fin = fopen( path.c_str(), "rb" );
    size_t elements_read = fread( &totalV, sizeof(int), 1, fin );
    
    int *len = new int[totalV];
	elements_read = fread(len, sizeof(int), totalV, fin); // 读取每个顶点的标签的数量

    label = new vector<unsigned>[totalV];
	unsigned *s = new unsigned[totalV];

	for( int i = 0; i < totalV; ++i ) {
		elements_read = fread(s, sizeof(unsigned), len[i], fin);
		label[i].reserve(len[i]);
		label[i].assign(s, s + len[i]);
	}

	elements_read = fread(len, sizeof(int), totalV, fin); // 读取每个顶点的标签的数量
    pos = new vector<int>[totalV];
	for( int i = 0; i < totalV; ++i ) {
		elements_read = fread(s, sizeof(unsigned), len[i], fin);
		pos[i].reserve(len[i]);
		pos[i].assign(s, s + len[i]);
	}

	delete[] s;
	delete[] len;

    elements_read = fread(&MAXMOV, sizeof(unsigned), 1, fin);
	MAXDIS = 1<<MAXMOV; MASK = MAXDIS-1;
}


void IndexPrint(){
    // string new_filename = "Indo_PSL.txt";
	// const char *file = new_filename.c_str();
	// fstream outfileX;
	// outfileX.open(file, ios::out);
	

    for (int i=0; i<totalV; ++i){
        vector<unsigned>& lab = label[i];
        // cout<<"id: "<<i<<"      ";
        for (int j=0; j<lab.size(); ++j){
            unsigned vid = lab[j] >> MAXMOV, dis = lab[j] & MASK;
            cout<<"$(v_"<<vid<<","<<dis<<")$, ";
        }

        // vector<unsigned>& cl = clab[i];
        // for (int j=0; j<cl.size(); ++j){
        //     unsigned vid = cl[j] >> MAXMOV, dis = cl[j] & MASK;
        //     cout<<"$("<<vid<<","<<dis<<")$, ";
        // }

        cout<<endl;

// outfileX<<lab.size()<<endl;
        // for (int j=0; j<pos[i].size(); ++j){
        //     cout<<pos[i][j]<<"  ";
        // }
        // cout<<endl;

    }
    // outfileX.close();
}


void IndexSize(string name){
    long long cntt = 0;
	const char *file = name.c_str();
	fstream outfileX;
	outfileX.open(file, ios::out);

    for (int i=0; i<totalV; ++i){
        cntt += label[i].size();
        // if (i < 100)
        // outfileX<<label[i].size()+clab[i].size()<<endl;
    }
    
    cout<<"label cnt: "<<cntt<<endl;


    outfileX<<cntt<<endl;
    outfileX.close();
}


void labelCheck(){
    // 主要检查label和pos的格式
    int dis = pos[0].size();

    for (int i=0; i<totalV; ++i){
        
        vector<unsigned>& lab = label[i];
        
        for (int dd = 1; dd<dis; ++dd){

            for (int pp = pos[i][dd-1]; pp<pos[i][dd]-1; ++pp){
                unsigned v1 = lab[pp]>>MAXMOV, d1 = lab[pp]&MASK;
                unsigned v2 = lab[pp+1]>>MAXMOV, d2 = lab[pp+1]&MASK;

                if (v1 >= v2) cout<<"error!"<<endl;
            }
        }
    }

    for (int i=0; i<totalV; ++i){
        vector<unsigned>& lab = label[i];
        for (int dd = 1; dd<dis; ++dd)
            for (int pp = pos[i][dd-1]; pp<pos[i][dd]; ++pp){
                unsigned v1 = lab[pp]>>MAXMOV, d1 = lab[pp]&MASK;
                if (d1 != dd) cout<<"error!"<<endl;
            }
    }
}


int CurDis(unsigned src, unsigned tgt, unsigned dcur){

    unsigned elem = src << MAXMOV | (dcur-1);

    for (int i=0; i<con[tgt].size(); ++i){ 
        // con[tgt][i]<src 的情况可以不考虑, check (src, dcur-1) 是否属于 L(con[tgt][i])
        unsigned vid = con[tgt][i];

        if ( vid < src) continue;
        
        int pla1 = dcur==1? 0:pos[vid][dcur-2],
            pla2 = pos[vid][dcur-1];

        for (int j=pla1; j<pla2; ++j)
            if (label[vid][j] == elem)
                return 1;
    }

    return 0;
}


// ============ for edge deletion =============

void DeleteGraph(float cc){    

    for (int i=0; i<Dcnt; ++i){
        int vid = (int)(totalV*cc) + i;
        int c = con[vid].size(), pla = (int)(c*0.9),
            adj = con[vid][pla]; // <i,adj>
        
        // vid = 0, adj = 4;
        // cout<<"delete: "<<vid<<"   "<<adj<<endl;
        
        conD[vid].push_back(adj); // for the index update
        conD[adj].push_back(vid);

        con[vid].erase(remove(con[vid].begin(), con[vid].end(), adj), con[vid].end());
        con[adj].erase(remove(con[adj].begin(), con[adj].end(), vid), con[adj].end());
    }
}


void IndexDel_Parallel(){ // 并行删除error label
    
    omp_set_num_threads(threads);

    vaff.resize(totalV, -1);
    
    delete_labels = new vector<unsigned>[totalV];

    int dis = 1;
    for( long long cnt = 1; cnt && dis <= MAXDIS; ++dis ){
        cnt = 0;
        vector<unsigned> *label_new = new vector<unsigned>[totalV];
        
        #pragma omp parallel
        {
            int pid = omp_get_thread_num(), np = omp_get_num_threads();
            long long local_cnt = 0;
            unsigned char *used = new unsigned char[totalV];
            memset( used, 0, sizeof(unsigned char) * (totalV) );
            vector<int> cand;
			
            int *nowdis = new int[totalV];
			memset( nowdis, -1, sizeof(int) * totalV);

            for( int u = pid; u < totalV; u += np ){
                cand.clear();
                
                int n_cand = 0, pla1 = pos[u][dis-1];
                int pla2 = dis < pos[u].size()? pos[u][dis]:label[u].size();

                for( int i = pla1; i < pla2; ++i ) nowdis[label[u][i] >> MAXMOV] = dis;
                // =========================================================================

                for (int i=0; i<con[u].size(); ++i){ // check the delete_label of neighbors
                    
                    int w = con[u][i];
                   
                    for( int j = 0; j < delete_labels[w].size(); ++j ) {
                        
                        int v = delete_labels[w][j] >> MAXMOV;
                        
                        if( v >= u ) break; // 已经排过序的
                        
                        if (nowdis[v] != dis) continue; // = (cand[i], dis) 不包含在 label[u] 中 =

                        if( used[v] == 0 ) {
                            used[v] = 1, cand.push_back(v);
                        }
                    }
                }

                // ====================================================

                for (int i=0; i<conD[u].size(); ++i){ // check the label of deleted neighbors
                    
                    int w = conD[u][i];

                    int pp1 = dis == 1? 0:pos[w][dis-2], pp2 = pos[w][dis-1];
                    
                    for( int j = pp1; j < pp2; ++j ) {
                        
                        int v = label[w][j] >> MAXMOV;

                        if( v >= u ) break;
                        
                        if (nowdis[v] != dis) continue; // = (cand[i], dis) 不包含在 label[u] 中 =

                        if( used[v] == 0 ) {
                            used[v] = 1, cand.push_back(v);
                        }
                    }                    
                
                    for( int j = 0; j < delete_labels[w].size(); ++j ) {
                        
                        int v = delete_labels[w][j] >> MAXMOV;
                        
                        if( v >= u ) break; // 已经排过序的
                        
                        if (nowdis[v] != dis) continue; // = (cand[i], dis) 不包含在 label[u] 中 =

                        if( used[v] == 0 ) {
                            used[v] = 1, cand.push_back(v);
                        }
                    }
                }

                // =========================================================================

                for( int i = 0; i < (int) cand.size(); ++i ){
                    used[cand[i]] = 0;

                    int D_now = CurDis(cand[i], u, dis);

                    if (D_now == 1) continue; // 不需要删除
                    
                    cand[n_cand++] = cand[i]; 
                }

                cand.resize(n_cand);

                sort(cand.begin(), cand.end());

                if (cand.size() > 0){
                    if (vaff[u] == -1) vaff[u] = (int) cand[0];
                    else               vaff[u] = min(vaff[u], (int) cand[0]); // 后续剪枝

                    cout<<"id: "<<u<<"    ";
                    for (int ii=0; ii<cand.size(); ++ii){
                        cout<<"v_"<<cand[ii]<<"  ";
                    }
                    cout<<endl;
                }

                for( int i = 0; i < (int) cand.size(); ++i ) {
                    label_new[u].push_back((((unsigned)cand[i])<<MAXMOV) | (unsigned) dis), ++local_cnt;
                }
                
                for( int i = pla1; i < pla2; ++i ) 
                    nowdis[label[u][i]>>MAXMOV] = -1;
            }


            #pragma omp critical
            {
                cnt += local_cnt;
            }

            delete[] used; delete[] nowdis;
        }

        #	pragma omp parallel
        {
            int pid = omp_get_thread_num(), np = omp_get_num_threads();
            
            for( int u = pid; u < totalV; u += np ){
                
                delete_labels[u].clear();
                delete_labels[u].insert(delete_labels[u].end(), label_new[u].begin(), label_new[u].end());
                vector<unsigned>(delete_labels[u]).swap(delete_labels[u]);
                vector<unsigned>().swap(label_new[u]);
                
                for (int i=0; i<delete_labels[u].size(); ++i){
                    label[u].erase( remove(label[u].begin(), label[u].end(), delete_labels[u][i]), label[u].end());
                    unsigned dd = delete_labels[u][i] & MASK;
                    for (int ij=dd; ij<pos[u].size(); ++ij) 
                        pos[u][ij] -= 1;
                }
            }
        }
       
        cout<<"Distance: "<<dis<<"  Delete Cnt: "<<cnt<<endl;
        delete[] label_new;
    }

    for (int i=0; i<totalV; ++i){
        cout<<vaff[i]<<endl;
    }
}


void IndexDel_Add(){ 

    omp_set_num_threads(threads);
    
    clab = new vector<unsigned>[totalV]; // 每回合更新的labels

    #pragma omp parallel
    {
        int pid = omp_get_thread_num(), np = omp_get_num_threads();
        for (int u=pid; u<totalV; u+=np){
            for (int i=pos[u][0]; i<pos[u][1]; ++i){
                unsigned vid = label[u][i] >> MAXMOV;

                if (vaff[vid] == -1) continue;
                
                clab[u].push_back(vid); // Vaff中的点都要被放入到激活名单
            }
        }
    }


    int dis = 2;
    for (long long cnt = 1; cnt > 0; ++dis){
        cnt = 0;
        vector<unsigned> *label_new = new vector<unsigned>[totalV];

        #pragma omp parallel
        {
            int pid = omp_get_thread_num(), np = omp_get_num_threads();
            long long local_cnt = 0, local_cnt1 = 0;
            unsigned char *used = new unsigned char[totalV/8+1];
            memset( used, 0, sizeof(unsigned char) * (totalV/8+1) );
            vector<int> cand;
            
            char *nowdis = new char[totalV];
            memset( nowdis, -1, sizeof(char) * totalV);
                        
            for (int u=pid; u<totalV; u+=np){ // 可以考虑减少线程调度的开销，从代码的层次
                
                cand.clear();
                
                for( int i = 0; i < con[u].size(); ++i ){

                    int w = con[u][i];

                    for (int j=0; j<clab[w].size(); ++j){

                        int v = clab[w][j]; // clab中包含所有需要考虑的vaff点

                        if ( v >= u ) break; // rank 剪枝
                        
                        if (vaff[u] == -1 and vaff[v] == -1) continue;

                        if( !(used[v/8]&(1<<(v%8))) ) {
                            used[v/8] |= (1<<(v%8)), cand.push_back(v);
                        }
                    }

                    if (vaff[u] != -1){ // vaff点需要遍历更多的候选label
                        
                        for( int j = pos[w][dis-2]; j < pos[w][dis-1]; ++j ){
                            
                            int v = label[w][j] >> MAXMOV;
                            
                            if ( v >= u ) break; // rank 剪枝

                            if (v < vaff[u]) continue;

                            if( !(used[v/8]&(1<<(v%8))) ) {
                                used[v/8] |= (1<<(v%8)), cand.push_back(v);
                            }
                        }
                    }
                }

                int n_cand = 0;
                if (cand.size() == 0) continue;
                
                int pla1 = dis<pos[u].size()? pos[u][dis]:label[u].size();

                for( int i = 0; i < pla1; ++i ) // 便于判断
                    nowdis[label[u][i] >> MAXMOV] = label[u][i] & MASK;

                for (int i=0; i<(int) cand.size(); ++i){
                    used[cand[i]/8] = 0;

                    if (nowdis[cand[i]] == dis){
                        ++local_cnt; // 直连标签在下一个回合中，需要进行判断
                        if (vaff[cand[i]] != -1) 
                            cand[n_cand++] = cand[i];
                    }else if( can_update_delete(cand[i], dis, nowdis)){
                        // cout<<"id: "<<u<<"  v_"<<cand[i]<<endl;
                        cand[n_cand++] = cand[i], ++local_cnt; 
                    }
                }

                cand.resize(n_cand);
                sort(cand.begin(), cand.end());

                for( int i = 0; i < (int) cand.size(); ++i ) {
                    label_new[u].push_back(cand[i]);
                }
                
                for( int i = 0; i < pla1; ++i ) // 便于判断
                    nowdis[label[u][i]>>MAXMOV] = -1;
            }

            #pragma omp critical
            {
                cnt += local_cnt;
            }

            delete[] used; delete[] nowdis;
        }

        #	pragma omp parallel
        {
            // Update labels and pos, 这里的insert方式要修改
            int pid = omp_get_thread_num(), np = omp_get_num_threads();

            for (int u=pid; u<totalV; u+=np){
                vector<unsigned>& lab = label[u];
                vector<int>& pp = pos[u];
                clab[u].clear();

                if (dis < pp.size()){ // 还是在现有的label之间
                
                    int p1 = pp[dis-1];

                    for (int ii=0; ii<label_new[u].size(); ++ii){
                        
                        unsigned vid = label_new[u][ii], elem = (vid<<MAXMOV)|dis;
                        int pla = -1, k;
                        clab[u].push_back(vid);

                        for(k=p1; k<pp[dis]; ++k){
                            if (lab[k] == elem){
                                pla = -2; break; // 不需要更新到labels中去  and vaff[vid] != -1
                            }else if (lab[k] > elem){
                                pla = k; break;
                            }
                        }

                        if (pla == -2) continue;

                        if (pla == -1) pla = k;
                        
                        p1 = pla;

                        lab.insert(lab.begin()+pla, elem);

                        for (int ii=dis; ii<pp.size(); ++ii) 
                            pp[ii] += 1;
                    }
                }else{
                    for (int i=0; i<label_new[u].size(); ++i){
                        unsigned elem = label_new[u][i]<<MAXMOV | dis;
                        lab.push_back(elem);
                        clab[u].push_back(label_new[u][i]);
                    }
                    pp.push_back(lab.size());
                }
                
                vector<unsigned>(clab[u]).swap(clab[u]);
                vector<unsigned>().swap(label_new[u]);
            }
        }
        delete[] label_new;
        cout<<"dis: "<<dis<<"  cnt: "<<cnt<<endl;
    }
}


// ============ for edge insertion =============

void Einsert(unsigned vid, vector<unsigned>& edges){
    int pla = -1;
    for (int i=0; i<edges.size(); ++i){
        if (edges[i] > vid){
            pla = i; break;
        }
    }

    if (pla == -1) edges.emplace_back(vid);
    else edges.emplace(edges.begin()+pla, vid);
}


void InsertGraph(float cc){
    
    int cnt = 0;
    float c = 0.8;
    
    while(cnt < Dcnt){
        
        int vid = (int)(totalV*cc) + cnt, adj = rand()%(int)(totalV*(1-c)) + (int)(totalV*c);
        // int vid = 0, adj = 5;

        if (adj >= totalV) continue;

        auto it = find(con[vid].begin(), con[vid].end(), adj);

        if ( it == con[vid].end() ){
            
            // cout<<vid<<"  "<<adj<<endl;

            Einsert(vid, conD[adj]);
            Einsert(adj, conD[vid]);

            Einsert(vid, con[adj]);
            Einsert(adj, con[vid]);
        }

        cnt += 1;
    }
}


void DynamicFile(int flg){
    string name = "dynamic.edge";
    string s;
    const char *filepath = name.c_str();
    ifstream infile;

    infile.open(filepath);
    if(!infile.is_open()){
        cout<<"No such file!"<<endl;
        exit(-1);
    }

    long xx = 0;

    while(getline(infile, s)){
        char* strc = new char[strlen(s.c_str())+1];
        strcpy(strc, s.c_str());
        char* s1 = strtok(strc," ");
        
        int ii = 0, va, vb;
        while(s1){
            if (ii % 2 == 0)
                va = atoi(s1);
            else{
                vb = atoi(s1);
                DyEdges.push_back(make_pair(va, vb));
                // cout<<va<<"  "<<vb<<endl;
            }

            ii += 1;
            s1=strtok(NULL," ");
        }

        xx += 1;
        
        delete s1, strc;
    }


    for (int i=0; i<DyEdges.size(); ++i){
        int vid = DyEdges[i].first, adj = DyEdges[i].second;
        if (flg == 0){
            Einsert(vid, conD[adj]);
            Einsert(adj, conD[vid]);

            Einsert(vid, con[adj]);
            Einsert(adj, con[vid]);
        }else{
            conD[vid].push_back(adj); // for the index update
            conD[adj].push_back(vid);

            con[vid].erase(remove(con[vid].begin(), con[vid].end(), adj), con[vid].end());
            con[adj].erase(remove(con[adj].begin(), con[adj].end(), vid), con[adj].end());

        }

    }
}


int DisVerdict(unsigned w, unsigned dmax, vector<int>& nowDis){
    // find a single path between tgt and w, where d<=dmax, return 1

    for (int i=0; i<pos[w][dmax]; ++i){
        unsigned vid = label[w][i] >> MAXMOV,
                 dis = label[w][i] & MASK;

        if (nowDis[vid] == -1) continue;

        if (nowDis[vid] + dis <= dmax or nowDis[vid] == 0) return 1;
    }

    return -1;
}


int ifdelete(unsigned tgt, unsigned tdis, unsigned v, unsigned vdis){ // remove tgt from label[v] 
    int flg = -1;
    vector<int> vec(totalV, -1);

    for (int i=0; i<pos[tgt][tdis-vdis]; ++i)
        vec[(label[tgt][i] >> MAXMOV)] = (label[tgt][i] & MASK);
    
    for (int i=0; i<clab[tgt].size(); ++i)  
        vec[ clab[tgt][i] ] = vdis; // 可以起到覆盖作用


    for (int i=0; i<clab[v].size(); ++i){
        
        unsigned w = clab[v][i];

        flg = DisVerdict(w, tdis-vdis, vec);

        if (flg != -1){
            flg = 1;
            break;
        }
    }

    return flg;
}


void Insert_Parallel(){

    omp_set_num_threads(threads);
    
    clab = new vector<unsigned>[totalV]; // 每回合更新的labels
    cpos = new vector<int>[totalV];

    vaff.resize(totalV, -1);

    for (int u=0; u<totalV; ++u){
        cpos[u].push_back(0);

        for (int i=0; i<conD[u].size(); ++i){
            if (conD[u][i] < u) {
                unsigned elem = conD[u][i] << MAXMOV | 1;
                clab[u].push_back(elem);
            }
        }

        if (clab[u].size() > 0){
            vector<unsigned>(clab[u]).swap(clab[u]);
            sort(clab[u].begin(), clab[u].end());
            vaff[u] = (clab[u][0] >> MAXMOV);   

            // cout<<" id: "<<u<<" :  ";
            // for (int ii=0; ii<clab[u].size(); ++ii){
            //     cout<<"$v_"<<(clab[u][ii] >> MAXMOV)<<"$ ";
            // }       
            // cout<<endl;
        } 
        
        cpos[u].push_back(clab[u].size());
    }


    int dis = 2;
    for (long long cnt = 1, cnttt = 0; cnt > 0; ++dis){
        cnt = 0, cnttt = 0;
        vector<unsigned> *label_new = new vector<unsigned>[totalV];

        #pragma omp parallel
        {
            int pid = omp_get_thread_num(), np = omp_get_num_threads();
            long long local_cnt = 0;
            unsigned char *used = new unsigned char[totalV/8+1];
            memset( used, 0, sizeof(unsigned char) * (totalV/8+1) );
            vector<int> cand;

            char *nowdis = new char[totalV];
            memset( nowdis, -1, sizeof(char) * totalV);

            for (int u=pid; u<totalV; u+=np){
                cand.clear();

                // === Obtain new labels ===
                for (int i=0; i<con[u].size(); ++i){
                    
                    int w = con[u][i];

                    for (int j=cpos[w][dis-2]; j<cpos[w][dis-1]; ++j){
                        
                        int v = clab[w][j] >> MAXMOV;

                        if (v >= u) break;

                        if( !(used[v/8]&(1<<(v%8))) ) {
                            used[v/8] |= (1<<(v%8)), cand.push_back(v);
                        }
                    }
                }

                if ( dis <= pos[0].size() )
                for (int i=0; i<conD[u].size(); ++i){
                    
                    int w = conD[u][i];

                    for (int j=pos[w][dis-2]; j<pos[w][dis-1]; ++j){

                        int v = label[w][j] >> MAXMOV;

                        if ( v >= u ) break;

                        if( !(used[v/8]&(1<<(v%8))) ) {
                            used[v/8] |= (1<<(v%8)), cand.push_back(v);
                        }
                    }
                }
            
                if (cand.size() == 0) continue;

                int n_cand = 0;
                int pla1 = dis<pos[0].size()? pos[u][dis]:label[u].size();
                int cla1 = dis<cpos[0].size()? cpos[u][dis]:clab[u].size();

                for(int i = 0; i < pla1; ++i) nowdis[label[u][i] >> MAXMOV] = label[u][i] & MASK;

                for(int i = 0; i < cla1; ++i) nowdis[clab[u][i] >> MAXMOV] = clab[u][i] & MASK;


                for (int i=0; i<(int) cand.size(); ++i){
                    used[cand[i]/8] = 0;

                    if (nowdis[cand[i]] != -1) 
                        continue; // 已经是直连，不需要再去判断

                    if( can_update_add(cand[i], dis, nowdis)){
                        cand[n_cand++] = cand[i];
                    }
                }


                cand.resize(n_cand);
                sort(cand.begin(), cand.end());

    // if (cand.size() > 0){
    // cout<<" id: "<<u<<" :  ";
    // for (int ii=0; ii<cand.size(); ++ii){
    //     cout<<"$v_"<<cand[ii]<<"$ ";
    // }       
    // cout<<endl;
    // }


                if (cand.size() > 0){
                    if (vaff[u] == -1) vaff[u] = (int) cand[0];
                    else               vaff[u] = min(vaff[u], (int) cand[0]); // 后续剪枝
                }

                for( int i = 0; i < (int) cand.size(); ++i ) {
                    label_new[u].push_back((((unsigned)cand[i])<<MAXMOV) | (unsigned) dis), ++local_cnt;
                }

                for(int i = 0; i < pla1; ++i)  nowdis[label[u][i]>>MAXMOV] = -1;
                for(int i = 0; i < cla1; ++i)  nowdis[clab[u][i]>>MAXMOV] = -1;
            }

            #pragma omp critical
            {
                cnt += local_cnt;
            }

            delete[] used; delete[] nowdis;
        }

        #	pragma omp parallel
        {   
            // === insert clab to label ===
            int pid = omp_get_thread_num(), np = omp_get_num_threads();

            for( int u = pid; u < totalV; u += np ){
                clab[u].insert(clab[u].end(), label_new[u].begin(), label_new[u].end());
                vector<unsigned>(clab[u]).swap(clab[u]);
                vector<unsigned>().swap(label_new[u]);
                cpos[u].push_back(clab[u].size());
            }
        }

        delete[] label_new;
        cout<<"dis: "<<dis<<"  cnt: "<<cnt<<endl;
    }

    // for (int i=0; i<totalV; ++i){
    //     cout<<vaff[i]<<endl;
    // }
}


int cand_remove(vector<int>& nowd, vector<unsigned>& lab, int dis){
    for (int i=0; i<lab.size(); ++i){
        unsigned w = lab[i] >> MAXMOV, d = lab[i] & MASK;
        if (d >= dis) break;

        if( nowd[w] > -1 && nowd[w] + d <= dis ) return 1;
    }

    return -1;
}

int cand_remove_no_prune(vector<int>& nowd, vector<unsigned>& lab, int dis){
    for (int i=0; i<lab.size(); ++i){
        unsigned w = lab[i] >> MAXMOV, d = lab[i] & MASK;
        if (d >= dis) break;

        if( nowd[w] > -1 && nowd[w] + d < dis ) return 1;
    }

    return -1;
}

// 针对的是 vaff[u] != -1 的顶点的判定
int cand_remove_2(vector<int>& nowD, int u, int v, int dis){
    
    vector<unsigned>& lab = label[v];

    for (int d=1; d<dis; ++d){
        int dd = dis-d; // clb 中的上限值
        unsigned vid = totalV, cnt = 0;
        dd = cpos[u].size()-1 <= dd? cpos[u].size()-1: dd;
        for (int j=1; j<=dd; ++j){
            if (cpos[u][j] == 0) continue;
            unsigned elem = clab[u][cpos[u][j]-1] >> MAXMOV;
            vid = max(vid, elem);
            cnt += 1;
        }

        if (vid == totalV and cnt == 0) vid = 0;
        
        for (int i=pos[v][d-1]; i<pos[v][d]; ++i){
            unsigned w = lab[i] >> MAXMOV, dw = lab[i] & MASK;
            if (w > vid) break; // 大于这个值表明，该距离值段的后续肯定没有交集
            if (nowD[w] > -1 && nowD[w] + dw <= dis) return 1;
        }
    }

    return -1;
}


void Insert_Remove_Parall(){

    vector<int> Rlab(totalV);
    long long cnt1 = 0, cnt2 = 0;
    Pla = new vector<unsigned>[totalV];

    #pragma omp parallel
    {
        int pid = omp_get_thread_num(), np = omp_get_num_threads();
        vector<int> nowdis(totalV, -1), nowdis_extra(totalV, -1);
        long long c1 = 0, c2 = 0;

        for( int u = pid; u < totalV; u += np ){
            
            for( int i = 0; i < clab[u].size(); ++i ) 
                nowdis[clab[u][i] >> MAXMOV] = clab[u][i] & MASK;

            for( int i = 0; i < label[u].size(); ++i ){
                unsigned v = label[u][i] >> MAXMOV, d = label[u][i] & MASK;
                if (nowdis[v] != -1){
                    Rlab[u] = 1, Pla[u].push_back(i);
                }else{
                    nowdis_extra[v] = d;
                }
            }

            // ==========================================

            for (int i=pos[u][0]; i<label[u].size(); ++i){ // dis=0,1的label不需要check
                
                unsigned vid = label[u][i] >> MAXMOV, dis = label[u][i] & MASK;

                if (nowdis_extra[vid] == -1) continue;

                int flg = -1;

                if (vaff[u] == -1 and vaff[vid] != -1){
                    flg = cand_remove(nowdis_extra, clab[vid], dis);
                }

                if (vaff[u] != -1 and vaff[vid] == -1 and vid > vaff[u])
                    flg = cand_remove_2(nowdis, u, vid, dis);
                
                if (vaff[u] != -1 and vaff[vid] != -1){
                    flg = cand_remove(nowdis, clab[vid], dis);
                    if (flg == -1) flg = cand_remove(nowdis_extra, clab[vid], dis);
                    if (flg == -1) flg = cand_remove_2(nowdis, u, vid, dis);
                }

// // === 对比测试 ===
// flg = cand_remove_no_prune(nowdis_extra, label[vid], dis);
// if (flg == -1) flg = cand_remove(nowdis, clab[vid], dis);
// if (flg == -1) flg = cand_remove(nowdis_extra, clab[vid], dis);
// if (flg == -1) flg = cand_remove_2(nowdis, u, vid, dis);

                if (flg == 1)  Rlab[u] = 1, Pla[u].push_back(i); // 对其他值的判定不会有影响
            } 

            for( int i = 0; i < clab[u].size(); ++i ) 
                nowdis[clab[u][i] >> MAXMOV] = -1;

            for( int i = 0; i < label[u].size(); ++i ){
                nowdis_extra[label[u][i] >> MAXMOV] = -1;
            }  
        }

        for( int u = pid; u < totalV; u += np ){
            
            vector<unsigned>& lab = label[u];
            
            if (Rlab[u] == 0) continue;

            int n_cand = 0, c = 0;
            sort(Pla[u].begin(), Pla[u].end());

            for (int i=0; i<label[u].size(); ++i){
                if (i != Pla[u][c]){
                    label[u][n_cand++] = label[u][i];
                }else{
                    if (c < Pla[u].size()) 
                        c += 1;
                }
            }

            label[u].resize(n_cand);
        }
    }
}


unsigned Query(int u, int v){

    if( u == v ) return 0;
	unsigned lu = (unsigned)label[u].size(), lv = (unsigned)label[v].size(), dis = MAXD;
	for( int i = 0, j = 0; i < lu && j < lv; ++i ) {
		
		for( ; j < lv && label[v][j]>>MAXMOV < label[u][i]>>MAXMOV; ++j ) ;
		if( j < lv && label[v][j]>>MAXMOV == label[u][i]>>MAXMOV) 
			dis = min(dis, (label[u][i]&MASK) + (label[v][j]&MASK));
	}
	return dis;
}



int main(){
    string na = "test";
    string name = "/mnt/data/zyy/Full/" + na + ".graph",
           Outpath = "/mnt/data/zyy/Index/" + na + ".bin",
           Outpath1 = "/mnt/data/zyy/Index/" + na + "_Update.bin";
    
    int choice = 0, Qcnt = 10000;
    float cp = 0.5;

    threads = 40;
    Dcnt = 100;

    GraphInitial(name);

    cout<<"cp: "<<cp<<"  Dcnt: "<<Dcnt<<"  Choice: "<<choice<<endl;
    cout<<na<<endl;
    if (choice == 0){
        float t = omp_get_wtime();
        
        IndexBuild();
        
        cout<<omp_get_wtime()-t<<" s"<<endl;    
        // string na1 = "socLiveJ_Execution_time.txt";
        // const char *file = na1.c_str();
        // fstream outfileX;
        // outfileX.open(file, ios::out);
        // outfileX<<omp_get_wtime()-t<<" s"<<endl;
        IndexPrint();
        IndexSave(Outpath);
    }else if (choice == 1){

        IndexLoad(Outpath);

        DeleteGraph(cp); // 必须先清理所有的删边，否则反向激活会有误差
        // DynamicFile(1);

        double t = omp_get_wtime();

        IndexDel_Parallel();
        
        IndexDel_Add();

        cout<<omp_get_wtime()-t<<" s"<<endl;

        // IndexPrint();
        // IndexSize("Update.txt");

    }else if (choice == 2){
        DeleteGraph(cp);

        double t = omp_get_wtime();
        IndexBuild();
        cout<<omp_get_wtime()-t<<" s"<<endl;

        IndexSize("PSL.txt");

    }else if (choice == 3){ // Edge insert
        
        IndexLoad(Outpath);

        InsertGraph(cp);
        // DynamicFile(0);

        double t = omp_get_wtime();

        Insert_Parallel();

        // cout<<omp_get_wtime()-t<<" s"<<endl;

        Insert_Remove_Parall();

        cout<<"Total time: "<<omp_get_wtime()-t<<" s"<<endl;
    
        // long long Lab_cnt = 0, Lab_n = 0;
        // for (int i=0; i<totalV; ++i){
        //     Lab_n += (label[i].size()+clab[i].size());
        //     Lab_cnt += (label[i].size()+clab[i].size()-Pla[i].size());
        // }
        // cout<<"No delete la:  "<<Lab_n<<endl;
        // cout<<"Total labels:  "<<Lab_cnt<<endl;

        
        // IndexPrint();
        // IndexSize("Add.txt");

    }else if (choice == 4){
        InsertGraph(cp);
        double t = omp_get_wtime();
        IndexBuild();
        cout<<"Reconstruct time: "<<omp_get_wtime()-t<<" s"<<endl;
        IndexSize("PSL.txt");
        // IndexPrint();
    }else{
        IndexLoad(Outpath);
        omp_set_num_threads(threads);
        printf( "sorting...\n");
        #pragma omp parallel
        {
            int pid = omp_get_thread_num(), np = omp_get_num_threads();
            for( int u = pid; u < totalV; u += np ) sort( label[u].begin(), label[u].end() );
        }
        double t = omp_get_wtime(), t1 = 0;
        for (int i=0; i<Qcnt; ++i){
            int u = rand() % totalV, v = rand() % totalV;
            int dis = Query(u,v);
            // cout<<u<<"  "<<v<<"  "<<dis<<endl;
        }
        t1 = omp_get_wtime() - t;
        printf("Average query time = %0.3lf ns\n", t1*1000000000.0/Qcnt);

    }

    

    return 0;
}


