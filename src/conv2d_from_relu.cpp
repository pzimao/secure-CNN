/*
NN-Layers-Benchmarking: This code benchmarks FC and Conv layers for a neural network

List of Authors:
Chiraag Juvekar, chiraag@mit.edu

License Information:
MIT License
Copyright (c) 2017, Massachusetts Institute of Technology (MIT)

*/
#include <utils/backend.h>
#include <iostream>
#include <random>
#include "pke/gazelle.h"
#include<string>
#include <fstream>
#include <regex>
#include <string>
#include <vector>

using namespace std;
using namespace lbcrypto;

void update_ct_matU(CTMat& ct_mat, const ui32 ct_idx,
        const SecretKey& sk, const uv64& in,
        const ui32 window_size, const ui32 num_windows, const FVParams& params){
    uv64 pt = packed_encode(in, params.p, params.logn);
    for(ui32 w=0; w<num_windows; w++){
        ct_mat[w][ct_idx] = Encrypt(sk, pt, params);
        for (ui32 i=0; i<params.phim; i++){
            pt[i] = ((pt[i] << window_size) % params.p);
        }
    }
} 


CTMat preprocess_ifmapU(const SecretKey& sk, const ConvLayer& in,
        const ui32 window_size, const ui32 num_windows, const FVParams& params){
    ui32 chn_pow2 = nxt_pow2(in.shape.h*in.shape.w);
    ui32 row_pow2 = nxt_pow2(in.shape.w);
    if (row_pow2*2 > params.phim){
        throw std::logic_error("Rows larger than half a ciphertext not supported");
    } else if(chn_pow2*2 > params.phim) {
        ui32 chn_pixels = row_pow2*in.shape.h;
        ui32 num_ct_chn = div_ceil(chn_pixels, params.phim);
        ui32 num_ct = num_ct_chn*2*div_ceil(in.shape.chn, 2);
        ui32 rows_per_ct = params.phim/2/row_pow2;
        CTMat ct_mat(num_windows, std::vector<Ciphertext>(num_ct, Ciphertext(params.phim)));
        for(ui32 curr_set=0; curr_set<div_ceil(in.shape.chn, 2); curr_set++){
            for(ui32 ct_offset=0; ct_offset<num_ct_chn*2; ct_offset++){
                uv64 packed_rows(params.phim, 0);
                for(ui32 row_offset=0; row_offset<rows_per_ct; row_offset++){
                    ui32 curr_h = ct_offset+2*num_ct_chn*row_offset;
                    if(curr_h < in.shape.h){
                        for(ui32 curr_seg=0; curr_seg<2; curr_seg++){
                            ui32 dest = row_offset*row_pow2+curr_seg*(params.phim >> 1);
                            ui32 curr_chn = 2*curr_set + curr_seg;                          
                            for(ui32 curr_w=0; curr_w<in.shape.w; curr_w++){
                                packed_rows[dest] = in.act[curr_chn][curr_h][curr_w];
                                dest++;
                            }
                            //添加
                            curr_chn++;
                            if(curr_chn == in.shape.chn){
                                break;
                            }
                         
                        } 
                        
                    } 
                }
                
                ui32 ct_idx = ct_offset + 2*num_ct_chn*curr_set;
                // Encode the packed channel and then encrypt its scaled copies
                update_ct_matU(ct_mat, ct_idx, sk, packed_rows, window_size, num_windows, params);
            }
        }

        return ct_mat;

    } else {
        // Pack multiple channels into a ciphertext
        ui32 tot_pixels = chn_pow2*in.shape.chn;
        ui32 num_ct = div_ceil(tot_pixels, params.phim);
        ui32 chn_per_ct = params.phim/chn_pow2;

        CTMat ct_mat(num_windows, std::vector<Ciphertext>(num_ct, Ciphertext(params.phim)));
        for(ui32 ct_idx=0; ct_idx<num_ct; ct_idx++){
            // Pack the appropriate number of channels into a single ciphertext
            uv64 packed_chn(params.phim, 0);
            for(ui32 chn_offset=0; chn_offset<chn_per_ct; chn_offset++){
                ui32 curr_chn = ct_idx*chn_per_ct+chn_offset;
                if(curr_chn == in.shape.chn) {
                    break;
                }

                ui32 dest = chn_offset*chn_pow2;
                for(ui32 curr_h=0; curr_h<in.shape.h; curr_h++){
                    for(ui32 curr_w=0; curr_w<in.shape.w; curr_w++){
                        packed_chn[dest] = in.act[curr_chn][curr_h][curr_w];
                        dest++;
                    }
                }
            }

            // Encode the packed channel and then encrypt its scaled copies
            update_ct_matU(ct_mat, ct_idx, sk, packed_chn, window_size, num_windows, params);
        }

        return ct_mat;
    }
}

double Round(double dSrc,  int iBit)
{
	double retVal 	= 0.0;
	int  intTmp		= 0;
	if(0 > iBit){
    	return 0;
	}
	if(0 > dSrc){
		dSrc *= -1;
		intTmp = (int)((dSrc +0.5/ pow(10.0, iBit)) * pow(10.0, iBit));
		retVal = (double)intTmp / pow(10.0, iBit);
		retVal *= -1;
	}else
	{
		intTmp = (int)((dSrc +0.5/ pow(10.0, iBit)) * pow(10.0, iBit));
		retVal = (double)intTmp / pow(10.0, iBit);
	}
	return retVal;
}


ConvLayer conv_2d_ptU(const ConvLayer& in, const Filter2D& filter, bool same, const ui32 p){
    ui32 out_h = in.shape.h - ((same) ? 0 : (filter.shape.f_h - 1));
    ui32 out_w = in.shape.w - ((same) ? 0 : (filter.shape.f_w - 1));
    ui32 offset_h = (same) ? (filter.shape.f_h-1)/2 : 0;
    ui32 offset_w = (same) ? (filter.shape.f_w-1)/2 : 0;

    ConvLayer out(filter.shape.out_chn, in.shape.h, in.shape.w);
    for(ui32 n=0; n<filter.shape.out_chn; n++){
        for (ui32 h = 0; h < out_h; h++){
            for (ui32 w = 0; w < out_w; w++){
                out.act[n][h][w] = filter.b[n];
                for(ui32 m=0; m<filter.shape.in_chn; m++){
                    for (ui32 f_h = 0; f_h < filter.shape.f_h; f_h++){
                        for (ui32 f_w = 0; f_w < filter.shape.f_w; f_w++){
                            ui32 in_h = h+f_h-offset_h;
                            ui32 in_w = w+f_w-offset_w;
                            // Uses the wrap-around property of ui32 to discard negative
                            bool zero = (same && (in_h >= in.shape.h || in_w >= in.shape.w));
                            ui64 in_act = zero ? 0:in.act[m][in_h][in_w];
                            out.act[n][h][w] += (filter.w[n][m][f_h][f_w]*in_act);
                        }
                    }
                }
                out.act[n][h][w] = out.act[n][h][w] % p;
            }
        }
    }

    return out;
}

ConvLayer conv_2d_ptU01(const ConvLayer& in, Filter2D& filter, bool same, const ui32 p){
    ui32 out_h = in.shape.h - ((same) ? 0 : (filter.shape.f_h - 1));
    ui32 out_w = in.shape.w - ((same) ? 0 : (filter.shape.f_w - 1));

    ui32 offset_h = (same) ? (filter.shape.f_h-1)/2 : 0;
    ui32 offset_w = (same) ? (filter.shape.f_w-1)/2 : 0;
    ConvLayer out(filter.shape.out_chn, in.shape.h, in.shape.w);
    for(ui32 n=0; n<filter.shape.out_chn; n++){ 
        for (ui32 h = 0; h < out_h; h++){
            for (ui32 w = 0; w < out_w; w++){
                int64_t sum=0;
                if(filter.b[n]>(opt::p)/2){
                    sum= filter.b[n]-opt::p;
                }else{
                    sum=filter.b[n];
                }
                for(ui32 m=0; m<filter.shape.in_chn; m++){
                    for (ui32 f_h = 0; f_h < filter.shape.f_h; f_h++){
                        for (ui32 f_w = 0; f_w < filter.shape.f_w; f_w++){
                            ui32 in_h = h+f_h-offset_h;
                            ui32 in_w = w+f_w-offset_w;
                            bool zero = (same && (in_h >= in.shape.h || in_w >= in.shape.w));
                            ui64 in_act = zero ? 0:in.act[m][in_h][in_w];
                            int64_t temp=0;
                            if(filter.w[n][m][f_h][f_w]>(opt::p)/2){
                                temp=filter.w[n][m][f_h][f_w]-opt::p;
                            }else{
                                temp=filter.w[n][m][f_h][f_w];
                            }
                            sum += temp*in_act;
                                                     
                        }
                    }
                }
                cout<<sum<<" ";

            }
            cout<<endl;     
        }
        cout<<endl;
  
    }
    return out;
}

uv64 readBias(string path, const FVParams & params){
    uv64 bias;
    ifstream in(path);
    string line;
    regex pat_regex("[-]?([0-9]+.[0-9]+)"); 
    while(getline(in, line)) {
            for (sregex_iterator it(line.begin(), line.end(), pat_regex), end_it; it != end_it; ++it) {
                float num=Round(stof(it->str().c_str()),3)*1000;       
                if(num<0){
                    num=num+params.p;
                }
                bias.push_back(num); 
            }
            if(line.size()==0){
                break;
            }          
    }            
    return bias;
}
//从文件里读小数
vector<vector<vector<vector<uint64_t>>>>readInputWeight(string path,int in_chn,const FVParams & params){
    
    vector<uint64_t> temp_line;
    vector<vector<uint64_t> > temp_matrx;
    vector<vector<vector<uint64_t>>> temp_weight;
    vector<vector<vector<vector<uint64_t>>>> weight;
    string line;
    ifstream in(path);
    regex pat_regex("[-]?([0-9]+.[0-9]+)");
    while(getline(in, line)) { 
            for (sregex_iterator it(line.begin(), line.end(), pat_regex), end_it; it != end_it; ++it) {
                float num=Round(stof(it->str().c_str()),2)*100;       
                if(num<0){
                    num=num+params.p;
                }
                temp_line.push_back(num); 
            }
            if(temp_line.size() > 0){
                temp_matrx.push_back(temp_line);
                temp_line.clear();  
            }else{
                temp_weight.push_back(temp_matrx);
                temp_matrx.clear();
                if(temp_weight.size()==in_chn){
                    weight.push_back(temp_weight); 
                    temp_weight.clear(); 
                }                  
            }              
        }
             
    return weight;
}

Ciphertext EvalSubPlainU(const Ciphertext& ct, const uv64& pt, const FVParams& params)
{
    uv64 pt_c(pt.size());

    for (ui32 i=0; i < pt.size(); i++)
    {
        pt_c[i] = mod((params.p-pt[i]), params.p);
    }

    uv64 pt_enc = packed_encode(pt_c, params.p, params.logn);


    for(ui32 i=0; i<pt_enc.size(); i++)
    {
           pt_enc[i] = pt_enc[i]*params.delta;
    }

    uv64 temp=NullEncrypt(pt_enc, params);


    return EvalAddPlain(ct, temp, params);
}


void recover_ctmat(CTMat &ct_mat, const uv64 &r, const ui32 ct_idx,
                   const ui32 window_size, const ui32 num_windows, const FVParams params)
{
    ct_mat[0][ct_idx]=EvalSubPlainU(ct_mat[0][ct_idx], r, params);
    
    for ( int w = 1; w < num_windows; w ++) 
    {
        uv64 pt(params.phim);
        for (ui32 n=0; n < pt.size(); n++)
        {
            pt[n]=(ui32)(1 << (w*window_size));
        }

        uv64 pt_enc = packed_encode(pt, params.p, params.logn);

	    uv64 temp=NullEncrypt(pt_enc, params);

        ct_mat[w][ct_idx]= EvalMultPlain(ct_mat[0][ct_idx], temp, params);

    }
}

CTMat preprocess_recover_ifmap_2stage(CTMat &ifmap, const ConvLayer& noise_map, const ConvShape & shape,
        const ui32 window_size, const ui32 num_windows, const FVParams & params)
{
    ui32 chn_pow2 = nxt_pow2(shape.h*shape.w);
    ui32 row_pow2 = nxt_pow2(shape.w);

    if (row_pow2*2 > params.phim){
        throw std::logic_error("Rows larger than half a ciphertext not supported");
    } else if(chn_pow2*2 > params.phim) {
        ui32 chn_pixels = row_pow2*shape.h;
        ui32 num_ct_chn = div_ceil(chn_pixels, params.phim);
        ui32 num_ct = num_ct_chn*2*div_ceil(shape.chn, 2);
        ui32 rows_per_ct = params.phim/2/row_pow2;
        for(ui32 curr_set=0; curr_set<div_ceil(shape.chn, 2); curr_set++){
            for(ui32 ct_offset=0; ct_offset<num_ct_chn*2; ct_offset++){
                uv64 packed_rows(params.phim, 0);
                for(ui32 row_offset=0; row_offset<rows_per_ct; row_offset++){
                    ui32 curr_h = ct_offset+2*num_ct_chn*row_offset;
                    if(curr_h < shape.h){
                        for(ui32 curr_seg=0; curr_seg<2; curr_seg++){
                            ui32 dest = row_offset*row_pow2+curr_seg*(params.phim >> 1);
                            ui32 curr_chn = 2*curr_set + curr_seg;
                            for(ui32 curr_w=0; curr_w<shape.w; curr_w++){
                                 
                                packed_rows[dest] = noise_map.act[curr_chn][curr_h][curr_w];
                                dest++;
                            }
                            curr_chn++;
                            if(curr_chn == noise_map.shape.chn){
                                break;
                            }
                        }
                    }
                }
                ui32 ct_idx = ct_offset + 2*num_ct_chn*curr_set;
                // Encode the packed channel and then encrypt its scaled copies
                recover_ctmat(ifmap, packed_rows, ct_idx, window_size, num_windows, params);
            }
        }  
        return ifmap;

    }   
                
}


CTMat preprocess_recover_ifmap(CTMat &ifmap, const vector<uv64> &vec_r, const ConvShape & shape,
        const ui32 window_size, const ui32 num_windows, const FVParams & params)
{
    ui32 chn_pow2 = nxt_pow2(shape.h*shape.w);
    ui32 row_pow2 = nxt_pow2(shape.w);

    ui32 tot_pixels = chn_pow2*shape.chn;

    ui32 num_ct = div_ceil(tot_pixels, params.phim);

    ui32 chn_per_ct = params.phim/chn_pow2;;

    CTMat ct_mat(num_windows, std::vector<Ciphertext>(num_ct, Ciphertext(params.phim)));
    for(ui32 ct_idx=0; ct_idx<num_ct; ct_idx++){
        uv64 packed_chn(params.phim, 0);
        for(ui32 chn_offset=0; chn_offset<chn_per_ct; chn_offset++){
            ui32 curr_chn = ct_idx*chn_per_ct+chn_offset;
            if(curr_chn == shape.chn) {
                break;
            }
        ui32 dest = chn_offset*chn_pow2;
            for(ui32 i=0; i<chn_pow2; i++){
                    packed_chn[dest] = vec_r[curr_chn%chn_per_ct][i];
                    dest++;
                }
            }
            recover_ctmat(ifmap, packed_chn, ct_idx, window_size, num_windows, params);
        }
        return ifmap;
        
                
}
//从文件里读取多个矩阵
vector<vector<uv64>> readInputTensor(string path){
    bool flag=false;
    uv64 temp_line;
    vector<uv64 > temp_matrx;
    vector<vector<uv64> > Vec_tensor;
    string line;
    ifstream in(path); 
    regex pat_regex("-?\\d+");
    while(getline(in, line)) {  
            flag=false;
            for (sregex_iterator it(line.begin(), line.end(), pat_regex), end_it; it != end_it; ++it) { 
                uint64_t num= strtoull(it->str().c_str(), NULL,10);
                temp_line.push_back(num); 
                flag=true;
            }
            if(!flag){
                Vec_tensor.push_back(temp_matrx);
                temp_matrx.clear();
                continue;
            }
            temp_matrx.push_back(temp_line);
            temp_line.clear();
            
    }
    return Vec_tensor;
}

//读取参数
vector<ui64> read_paramater(string path_parameter ){
    ifstream inFile(path_parameter);
    vector<ui64> v_paramater;
    if (inFile)
    {
        string strLine;
        while(getline(inFile, strLine)) 
        {  
            v_paramater.push_back(stoi(strLine));
        }   
    } 
    return v_paramater;
}
//获取补码
uv64 get_nagative(uv64 vec,const FVParams& params){
    uv64 r2_nagtive(vec.size());
    for(ui32 n=0; n<vec.size(); n++){
        r2_nagtive[n] = mod(params.p-vec[n],params.p);
    }  
    return r2_nagtive;   
}

void print_multi_matrix(vector<vector<uv64>> matrix){
    for(int i=0;i<matrix.size();i++){
        for(int j=0;j<matrix[0].size();j++){
             cout<<vec_to_str(matrix[i][j])<<endl;
        }
        cout<<endl;
    }
}
//转换为矩阵
vector<vector<uv64>> trans_to_matrix(vector<uv64> vec, ui32 in_h,ui32 in_w){
    vector<vector<uv64>> matrix;
    vector<uv64>  temp_matrix;
    uv64 temp;
    for(int k=0;k<vec.size();k++){
        for(int i=0;i<in_h*in_w;i+=in_h){
            for(int j=i;j<i+in_h;j++){
                temp.push_back(vec[k][j]);
            }
            temp_matrix.push_back(temp);
            temp.clear();
        }
        matrix.push_back(temp_matrix);
        temp_matrix.clear();
    }
    return matrix;
}
vector<vector<vector<uv64>>> trans_to_matrix_group(ui32 group_num,vector<uv64> vec, ui32 in_h,ui32 in_w){
    vector<vector<vector<uv64>>> group_matrix;
    vector<vector<uv64>> matrix;
    vector<uv64>  temp_matrix;
    uv64 temp;
    for(int k=0;k<vec.size();k++){
        for(int i=0;i<in_h*in_w;i+=in_h){
            for(int j=i;j<i+in_h;j++){
                temp.push_back(vec[k][j]);
            }
            temp_matrix.push_back(temp);
            temp.clear();
        }
        matrix.push_back(temp_matrix);
        temp_matrix.clear();

        if(matrix.size()==group_num){
            group_matrix.push_back(matrix);
            matrix.clear();
        }
    }
    return group_matrix;
}

uv64 empty_encrypt(uv64 vec,const FVParams & params){
    auto pt_enc = packed_encode(vec, params.p, params.logn);
    for(ui32 n=0; n<params.phim; n++){
        pt_enc[n] = pt_enc[n]*params.delta;
    }
    uv64 ct=NullEncrypt(pt_enc, params);
    return ct;
}

CTVec addBias(CTVec &ct_conv,const Filter2D& filter,int in_h,int in_w,const FVParams & params){
    CTVec new_ct_conv;
    ui32 chn_pow2 = nxt_pow2(in_h*in_w);
    ui32 chn_per_ct = params.phim/chn_pow2;
    ui32 out_ct = div_ceil(filter.shape.out_chn, chn_per_ct);
    vector<uv64> bias_vec;
    uv64 temp1;
    for(int i=0;i<filter.shape.out_chn;i++){
        for(int j=0;j<params.phim;j++){
            temp1.push_back(filter.b[i]);
        }
        bias_vec.push_back(temp1);
        temp1.clear();
    }
    vector<uv64> bias_joint;
    uv64 temp;
    for(int j=0;j<filter.shape.out_chn;j=j+chn_per_ct){
        for(int k=0;k<chn_per_ct;k++){
             if(filter.shape.out_chn<=j+k){
                break;        
            }
            for(int i=k*chn_pow2;i<(k+1)*chn_pow2;i++){
                temp.push_back(bias_vec[j+k][i]);
            }
        }
       
        bias_joint.push_back(temp);
        temp.clear();  
    }
    //如果某一个向量没填满2048位，需要用0填满
    int end=bias_joint.size();
    if(filter.shape.out_chn%chn_per_ct!=0){
        int num=filter.shape.out_chn%chn_per_ct;
        for(int i=num*chn_pow2;i<params.phim;i++){
            bias_joint[end-1].push_back(0);
        }
    }
    
    for(int i=0;i<ct_conv.size();i++){
        uv64 ct_bias=empty_encrypt(bias_joint[i],params);
        new_ct_conv.push_back( EvalAddPlain(ct_conv[i], ct_bias, params));
    }
    return new_ct_conv;
}


CTVec addBias_2stage(CTVec &ct_conv,const Filter2D& filter,int in_h,int in_w,const FVParams & params){
    ui32 chn_pow2 = nxt_pow2(in_h*in_w);
    ui32 row_pow2 = nxt_pow2(in_w);
    ui32 chn_pixels = row_pow2*in_h;
    ui32 num_ct_chn = div_ceil(chn_pixels, params.phim);
    ui32 rows_per_ct = params.phim/2/row_pow2;
    vector<vector<uv64>> bias_vec;
    vector<uv64> bias_temp;
    uv64 temp;
    for(int j=0; j<filter.shape.out_chn; j=j+2){
        for(int k=0;k<num_ct_chn*2;k++){
            for(int curr_seg=0; curr_seg<2; curr_seg++){
                for(int i=curr_seg*params.phim/2;i<(curr_seg+1)*params.phim/2;i++){
                    if(j+curr_seg>=filter.shape.out_chn){
                        temp.push_back(0);
                    }else{
                        temp.push_back(filter.b[j+curr_seg]);
                    }
                    
                }
                
            }
            bias_temp.push_back(temp);
            temp.clear(); 
        }
        bias_vec.push_back(bias_temp);
        bias_temp.clear();
        
    }
    CTVec new_ct_conv;
    int j=0;
    for(int i=0;i<ct_conv.size();i=i+num_ct_chn*2){
        for(int k=0;k<num_ct_chn*2;k++){
                uv64 ct_bias=empty_encrypt(bias_vec[j][k],params);
                new_ct_conv.push_back(EvalAddPlain(ct_conv[i+k],ct_bias,params));
        }
        j++;
    }
    
    return new_ct_conv;
}

void print_map(vector<vector<uv64>> map){
    for(int chn=0;chn<map.size();chn++){
        for(int h=0;h<map[chn].size();h++){
            cout<<vec_to_str(map[chn][h])<<endl;      
        }
        cout<<endl;
    }

}
void print_weight(int out_chn,int in_chn,int f_h,Filter2D filter){
    for(ui32 ochn=0; ochn<out_chn; ochn++){
        for(ui32 ichn=0; ichn<in_chn; ichn++){
            for(ui32 h=0; h<f_h; h++){
                cout<<vec_to_str(filter.w[ochn][ichn][h])<<endl;
            }
        cout<<endl;
        }
        cout<<endl;
    }

}

EncMat preprocess_filter_2stageU(int in_set,int in_row_idx, int f_w, int f_h, int out_set, int curr_loop,const Filter2D& filter, const ConvShape& shape,
         const ui32 window_size, const ui32 num_windows, const FVParams& params){
    ui32 chn_pow2 = nxt_pow2(shape.h*shape.w);
    ui32 row_pow2 = nxt_pow2(shape.w);

    ui32 offset_h = (filter.shape.f_h-1)/2;
    ui32 offset_w = (filter.shape.f_w-1)/2;

    if (row_pow2*2 > params.phim){
        throw std::logic_error("Rows larger than half a ciphertext not supported");
    } else if(chn_pow2*2 > params.phim) {
        ui32 chn_pixels = row_pow2*shape.h;
        ui32 num_ct_chn = div_ceil(chn_pixels, params.phim);
        if(num_ct_chn < offset_h){
            throw std::logic_error("Unsupported filter and input combination");
        }

        ui32 rows_per_ct = params.phim/2/row_pow2;
        ui32 in_ct = num_ct_chn*2*div_ceil(filter.shape.in_chn, 2);
        ui32 rot_per_in = filter.shape.f_h*filter.shape.f_w;
        ui32 out_ct = 2*div_ceil(filter.shape.out_chn, 2);

        // Create the diagonal rotation of the plaintext matrix
        ui32 enc_row = 0;
        ui32 num_filter_rows = in_ct*rot_per_in*out_ct;
        EncMat enc_filter(1, std::vector<uv64>(num_windows, uv64(params.phim)));
        bool first_skip = (in_row_idx+offset_h < f_h);              
        bool last_skip = (in_row_idx+offset_h >= (2*num_ct_chn+f_h));
        uv64 filter_base(params.phim, 0);


        for(ui32 curr_offset=0; curr_offset<2; curr_offset++){
            ui32 curr_in = 2*in_set + curr_offset;
            ui32 curr_out = 2*out_set + (curr_offset+curr_loop)%2;

            ui64 coeff = ((curr_in >= filter.shape.in_chn) || (curr_out >= filter.shape.out_chn))? 0:
                    filter.w[curr_out][curr_in][f_h][f_w];
            
            ui32 dest = curr_offset*params.phim/2;
            for(ui32 curr_h=0; curr_h<rows_per_ct; curr_h++){
                for(ui32 curr_w=0; curr_w<shape.w; curr_w++){
                    bool zero = (((curr_h == 0) && last_skip) ||
                            ((curr_h == rows_per_ct-1) && first_skip) ||
                            ((curr_w+f_w) < offset_w) ||
                            ((curr_w+f_w) >= (offset_w+shape.w)));
                    filter_base[dest] = zero? 0: coeff;
                    dest++;
                }
            }
        }
        auto pt_row = packed_encode(filter_base, params.p, params.logn);
        auto decomposed_row = base_decompose(pt_row, window_size, num_windows);
        for(ui32 w=0; w<num_windows; w++){
            enc_filter[0][w] = NullEncrypt(decomposed_row[w], params);
        }
        return enc_filter;
    }
}

CTVec conv_2d_2stage_onlineU(const Filter2D& filter, const ConvShape& shape,
         const ui32 window_size, const ui32 num_windows,const CTMat& ct_mat,
        const Filter2DShape& filter_shape, const ConvShape& in_shape, const FVParams& params){
    ui32 chn_pow2 = nxt_pow2(in_shape.h*in_shape.w);
    ui32 row_pow2 = nxt_pow2(in_shape.w);

    ui32 offset_h = (filter_shape.f_h-1)/2;
    ui32 offset_w = (filter_shape.f_w-1)/2;
    
    if (row_pow2*2 > params.phim){
        throw std::logic_error("Rows larger than half a ciphertext not supported");
    } else if(chn_pow2*2 > params.phim) {
        ui32 chn_pixels = row_pow2*in_shape.h;
        ui32 num_ct_chn = div_ceil(chn_pixels, params.phim);
        EncMat enc_matU(1, std::vector<uv64>(num_windows, uv64(params.phim)));
        if(num_ct_chn < offset_h){
            throw std::logic_error("Unsupported filter and input combination");
        }
        ui32 out_ct = num_ct_chn*2*div_ceil(filter_shape.out_chn, 2);
        CTVec ct_mid(out_ct*2, Ciphertext(params.phim));
        CTVec rot_vec(2, Ciphertext(params.phim));
        for(ui32 w=0; w<ct_mat.size(); w++){
            ui32 filter_row = 0;
            for(ui32 in_set=0; in_set<div_ceil(filter_shape.in_chn, 2); in_set++){
                for(ui32 in_row_idx=0; in_row_idx<2*num_ct_chn; in_row_idx++){
                    ui32 in_ct_idx = in_row_idx + in_set*2*num_ct_chn;

                    std::vector<uv64> digits_vec_w;
                    if((filter_shape.f_w > 1) || (in_row_idx < offset_h) ||
                            (in_row_idx >= (2*num_ct_chn-offset_h))) {
                        digits_vec_w = HoistedDecompose(ct_mat[w][in_ct_idx], params);
                    }

                    for(ui32 f_w=0; f_w<filter_shape.f_w; f_w++){
                        ui32 rot_w = (f_w-offset_w);
                        ui32 rot_h = 0;
                        if(in_row_idx < offset_h) {
                            rot_h = in_shape.w;
                        } else if (in_row_idx >= (2*num_ct_chn-offset_h)) {
                            rot_h = (params.phim >> 1)-in_shape.w;
                        }
                        ui32 rot_a = (rot_w & ((params.phim >> 1) - 1));
                        const Ciphertext *base_vec = &ct_mat[w][in_ct_idx];
                        if(rot_a != 0){
                            auto rk = GetAutomorphismKey(rot_a);
                            rot_vec[0] = EvalAutomorphismDigits(rot_a, *rk, *base_vec, digits_vec_w, params);
                            base_vec = &rot_vec[0];
                        }

                        ui32 rot_b = ((rot_h + rot_w) & ((params.phim >> 1) - 1));
                        const Ciphertext *alt_vec = &ct_mat[w][in_ct_idx];
                        if(rot_b != 0){
                            auto rk = GetAutomorphismKey(rot_b);
                            rot_vec[1] = EvalAutomorphismDigits(rot_b, *rk, *alt_vec, digits_vec_w, params);
                            alt_vec = &rot_vec[1];
                        }

                        for(ui32 f_h=0; f_h<filter_shape.f_h; f_h++){
                            ui32 out_row_idx = in_row_idx+offset_h;
                           
                            const Ciphertext *curr_vec;
                            if(out_row_idx < f_h) {
                                out_row_idx += (2*num_ct_chn-f_h);
                                curr_vec = alt_vec;
                              
                            } else if (out_row_idx >= (2*num_ct_chn+f_h)) {
                                out_row_idx -= (2*num_ct_chn+f_h);
                                curr_vec = alt_vec;
                            } else {
                                out_row_idx -= f_h;
                                curr_vec = base_vec;
                            }

                            // Accumulate to all the outputs
                            for(ui32 out_set=0; out_set<div_ceil(filter_shape.out_chn, 2); out_set++){
                                ui32 out_ct_idx = out_row_idx + out_set*2*num_ct_chn;

                                for(ui32 inner_loop=0; inner_loop<2; inner_loop++){
                                    enc_matU=preprocess_filter_2stageU(in_set,in_row_idx, f_w, f_h, out_set, inner_loop, filter, shape,window_size, num_windows, params);
                                    ui32 mid_ct_idx = 2*out_ct_idx + inner_loop;
                                    auto mult = EvalMultPlain(*curr_vec, enc_matU[0][w], params);
                                    ct_mid[mid_ct_idx] = EvalAdd(ct_mid[mid_ct_idx], mult, params);
                                    filter_row++;
                                    cout<<"filter_row:"<<filter_row<<endl;
                                    enc_matU.clear();

                                }
                            }
                        }
                    }
                }
            }
        }
        CTVec ct_vec(out_ct, Ciphertext(params.phim));
        for(ui32 curr_out_ct=0; curr_out_ct<out_ct; curr_out_ct++){
            auto rot_vec = EvalAutomorphism(params.phim/2, ct_mid[curr_out_ct*2+1], params);
            ct_vec[curr_out_ct] = EvalAdd(ct_mid[curr_out_ct*2], rot_vec, params);
        }

        return ct_vec;
    } 
}

vector<uv64> get_vec_nagative(vector<uv64> r,const FVParams& params){
    vector<uv64> vec_nagative;
    uv64 temp;
    for(int i=0;i<r.size();i++){
        temp=get_nagative(r[i],params);
        vec_nagative.push_back(temp);
        temp.clear();
    }
    return vec_nagative;

}
void print_result_2stage(int in_h, int in_w,int out_chn, vector<uv64> output,vector<uv64> r_nagative,const FVParams& params){
    uv64 y(in_h*in_w);
    //噪声生成的有问题，需要噪声对应起来
    for(int i=0;i<output.size();i++){
        for(int j=0;j<in_h*in_w;j++){
                y[j]=mod(output[i][j]+r_nagative[i][j],opt::p);
            }
       
        for(int k=0;k<in_h*in_w;k=k+in_h){
                for(int j=k;j<k+in_h;j++){
                    cout<<y[j]<<" ";
                }
                cout<<endl;
            }
        cout<<endl;
     } 
}

uv64 get(ui32 start,ui32 end, uv64 out,const FVParams& params){
    uv64 temp(params.phim);
    int i=0;
    for(start;start<end;start++){
        temp[i]=out[start];
        i++;
    }
    for(int j=end;j<params.phim;j++){
        temp[j]=0;
    }
    return temp;
}
vector<uv64> split_vec(uv64 vec_orgin,int in_w, int in_h,const FVParams& params){
    vector<uv64> vec;
    uv64 temp;
    ui32 chn_pow2 = nxt_pow2(in_w*in_h);
    int split_num=params.phim/chn_pow2;
    for(int i=0;i<split_num;i++){
         temp=get(chn_pow2*i,chn_pow2*i+in_w*in_h,vec_orgin,params);
         vec.push_back(temp);
         temp.clear();
    }

    return vec;
}
vector<uv64> eliminate_noise_2stage(vector<uv64> vec,vector<uv64> nagative,const FVParams& params){
    vector<uv64> matrix;
    uv64 x(vec[0].size());
    for(int i=0;i<vec.size();i++){

        for(int j=0;j<vec[i].size();j++){
            x[j]=mod(vec[i][j]+nagative[i][j],opt::p);
        }
        matrix.push_back(x);
    }
    return matrix;
}

vector<uv64> eliminate_noise(ui32 in_h,ui32 in_w,vector<uv64> x_r_vec,vector<uv64> nagative,const FVParams& params){
    ui32 chn_pow2 = nxt_pow2(in_w*in_h);
    int split_num=params.phim/chn_pow2;
    vector<uv64> matrix;
    uv64 x(chn_pow2);
    for(int j=0;j<x_r_vec.size();j=j+split_num){
        ui32 curr_r=0;
        for(int i=j;i<j+split_num;i++){
            for(int k=0;k<chn_pow2;k++){
                x[k]=mod(x_r_vec[i][k]+nagative[curr_r][k],params.p);
            }
            curr_r++;
            matrix.push_back(x);
        }
    }
    return matrix;
}

CTVec comput_convolution(int conv_type,const SecretKey& sk,const vector<uv64>& noise,const ConvLayer& noise_map,const ConvLayer& ifmap, Filter2D& filter,const FVParams& params){
    double start, stop;
    ui32 pt_window_size=10;
    ui32 pt_num_windows = 1 + floor(log2(params.p))/pt_window_size;
    start = currentDateTime();
    ui32 chn_pow2 = nxt_pow2(ifmap.shape.h*ifmap.shape.w);
    ui32 offset_h = (filter.shape.f_h-1)/2;
    ui32 offset_w = (filter.shape.f_w-1)/2;

    ui32 chn_per_ct = opt::phim/chn_pow2;
    ui32 inner_loop = chn_per_ct;

    uv32 index_list;
    if(conv_type == 0){
        for(ui32 curr_loop=0; curr_loop<inner_loop; curr_loop++){
            ui32 rot_base = curr_loop*chn_pow2;
            for(ui32 f_y=0; f_y<filter.shape.f_h; f_y++){
                ui32 rot_h = (f_y-offset_h)*ifmap.shape.w;
                for(ui32 f_x=0; f_x<filter.shape.f_w; f_x++){
                    ui32 rot_w = (f_x-offset_w);
                    ui32 rot_f = ((rot_base + rot_h + rot_w) & ((opt::phim >> 1) - 1));
                    ui32 rot = (rot_base & (opt::phim >> 1)) + rot_f;

                    index_list.push_back(rot);
                }
            }
        }
    } else {
        for(ui32 f_y=0; f_y<filter.shape.f_h; f_y++){
            ui32 rot_h = (f_y-offset_h)*ifmap.shape.w;
            for(ui32 f_x=0; f_x<filter.shape.f_w; f_x++){
                ui32 rot_w = (f_x-offset_w);
                ui32 rot = ((rot_h + rot_w) & ((opt::phim >> 1) - 1));

                index_list.push_back(rot);
            }
        }

        for(ui32 curr_loop=1; curr_loop<inner_loop; curr_loop++){
            ui32 rot_base = curr_loop*chn_pow2;
            ui32 rot_r = ((opt::phim >> 1) - rot_base) & ((opt::phim >> 1) - 1);
            ui32 rot = (rot_base & (opt::phim >> 1)) + rot_r;

            index_list.push_back(rot);
        }
        index_list.push_back(opt::phim/2);
    }
    EvalAutomorphismKeyGen(sk, index_list, params);
    index_list.clear();
    index_list.shrink_to_fit();
    //------------------------client:对输入的特征map进行加密---------------------
    auto ct_mat = preprocess_ifmapU(sk, ifmap, pt_window_size, pt_num_windows, params);
    //------------------------server:加密的结果需要减去r----------------- 
   
    if(conv_type==0){
        ct_mat=preprocess_recover_ifmap(ct_mat, noise, ifmap.shape, pt_window_size, pt_num_windows, params);
    }else{
        ct_mat = preprocess_recover_ifmap_2stage(ct_mat, noise_map, ifmap.shape, pt_window_size, pt_num_windows, params);
    }
    //------------------- Post-Process ---------------------

    //------------------------server:对滤波器进行加密-----------------------------
    auto enc_filter = (conv_type) ?
            preprocess_filter_2stage(filter, ifmap.shape, pt_window_size, pt_num_windows, params):
            preprocess_filter(filter, ifmap.shape, pt_window_size, pt_num_windows, params);
    //----------------------- Conv2D ----------------------------------------
    //------------------------server:加密情况下卷积运算-------------------------
    auto ct_conv = (conv_type) ?
            conv_2d_2stage_online(ct_mat, enc_filter, filter.shape, ifmap.shape, params):
            conv_2d_online(ct_mat, enc_filter, filter.shape, ifmap.shape, params);
    return ct_conv;
}

void save_result_relu(string path,vector<vector<uv64>> act, const FVParams& params){
    
    ofstream cheack(path,ios::out);
    float value=0;
    vector<float> temp;
    for(int i=0;i<act.size();i++){
        for(int j=0;j<act[i].size();j++){
            for(int k=0;k<act[i][j].size();k++){
                 if(act[i][j][k]<params.p/2){
                    value=act[i][j][k]/10.0;
                }else{
                    ui64 a=act[i][j][k];
                    value=((params.p-act[i][j][k]))*(-1)/10.0;
                }
                temp.push_back(value);
            }
            cheack<<vec_to_str(temp)<<endl;
            temp.clear();
        }
        cheack<<endl;
    }
    cheack<<endl;
    cheack.close();
}

void save_result(string path,vector<vector<uv64>> act, const FVParams& params){
    
    ofstream cheack(path,ios::out);
    float value=0;
    vector<float> temp;
    for(int i=0;i<act.size();i++){
        for(int j=0;j<act[i].size();j++){
            for(int k=0;k<act[i][j].size();k++){
                 if(act[i][j][k]<params.p/2){
                    value=act[i][j][k]/1000.0;
                }else{
                    value=((params.p-act[i][j][k])/1000.0)*(-1);
                }
                temp.push_back(value);
            }
            cheack<<vec_to_str(temp)<<endl;
            temp.clear();
        }
        cheack<<endl;
    }
    cheack<<endl;
    cheack.close();
}
int  main() {
  
    //----------------------执行conv1_2-----------------------------
    string input_path="../result/relu_5_2_out.txt";
    string path_parameter="../params/con5_3_param.txt";
    string path_output="../result/5_3_output.txt";
    string path_outkey="../params/outkey.txt";
    string conv_filter="../params/5_3w.txt";
    string path_bias="../params/5_3b.txt";
    //----------------------从文件中读入参数---------------------------
    vector<ui64> v_paramater=read_paramater(path_parameter);
    //----------------------从文件中读取参数并打印----------------------
    ui32 out_chn = v_paramater[0];
    ui32 in_chn = v_paramater[1]; 
    ui32 in_h = v_paramater[2];
    ui32 in_w = v_paramater[3];
    ui32 f_h = v_paramater[4];
    ui32 f_w = v_paramater[5];
    ui32 conv_type =v_paramater[6];
    v_paramater.clear();
    v_paramater.shrink_to_fit();
     //---------------------- Setup Parameters -------------------------------------
    ui64 z = RootOfUnity(opt::phim << 1, opt::q);
    ui64 z_p = RootOfUnity(opt::phim << 1, opt::p);
    ftt_precompute(z, opt::q, opt::logn);
    ftt_precompute(z_p, opt::p, opt::logn);
    encoding_precompute(opt::p, opt::logn);
    precompute_automorph_index(opt::phim);

    DiscreteGaussianGenerator dgg = DiscreteGaussianGenerator(4.0);

    FVParams slow_params {
        false,
        opt::q, opt::p, opt::logn, opt::phim,
        (opt::q/opt::p),
        OPTIMIZED, std::make_shared<DiscreteGaussianGenerator>(dgg),
        8
    };

    FVParams fast_params = slow_params;
    fast_params.fast_modulli = true;
    ui32 nRep=1;
    FVParams test_params = fast_params;
    ui32 w_sz = 7;
    test_params.window_size=w_sz;
    ui32 pt_window_size=10;
    ui32 pt_num_windows = 1 + floor(log2(test_params.p))/pt_window_size;
    //------------------------从文件里读取密钥----------------------
    vector<vector<uv64>> vec_key=readInputTensor(path_outkey);
    SecretKey sk(2048);
    sk.s=vec_key[0][0]; 
    
    //-----------------------从relu中读入输入数据-------------------
    vector<vector<uv64>> input;
    input=readInputTensor(input_path);
    for(int i=0;i<input[0].size();i++){
        for(int j=0;j<input[0][i].size();j++){
            input[0][i][j]=input[0][i][j]/100;
        }
    }

    vector<uv64> noise_vec;//in_h*in_w
    vector<uv64> x_r_vec;//in_h*in_w
    ui32 chn_pow2 = nxt_pow2(in_w*in_h);
    int split_num=test_params.phim/chn_pow2;
    if(conv_type==0){
        
        for(int i=0;i<split_num;i++){
            noise_vec.push_back(input[0][i]);
        }
        for(int j=split_num;j<input[0].size();j++){
            x_r_vec.push_back(input[0][j]);
        }

    }else{
        for(int i=0;i<out_chn;i++){
            noise_vec.push_back(input[0][i]);
        }
        for(int j=out_chn;j<2*out_chn;j++){
            x_r_vec.push_back(input[0][j]);
        }
    }
 
    //用作测试
    vector<uv64> vec_x;
    vector<uv64> nagative=get_vec_nagative(noise_vec,test_params);
    //-----------------------将噪声r减去----------------------------------
    if(conv_type==0){
        vec_x=eliminate_noise(in_h,in_w,x_r_vec,nagative, test_params);
    }else{
        vec_x=eliminate_noise_2stage(x_r_vec,nagative,test_params);
    }
    input.clear();
    input.shrink_to_fit();

    CTVec ct_output;
    ui32 chn_per=512;//32
    ui32 map_num_per_group=8;//8
    ui32 group_num=in_chn/map_num_per_group;
    vector<CTVec> ct_vec(group_num);

    //-----------------------将y转换成矩阵的形式--------------------------
    //需要将map分组
    //64*256*256-->8×8×256×256 
    //128*128*128-->16*8*128*128
    //256*64*64-->32*8*64*64
    //512*64*64-->64*8*32*32
    vector<vector<vector<uv64>>> x_matrix;
    x_matrix=trans_to_matrix_group(map_num_per_group,vec_x,in_h,in_w);
    //-----------------------将x_r转换成矩阵的形式-----------------------
    vector<vector<vector<uv64>>> x_r_matrix;
    x_r_matrix=trans_to_matrix_group(map_num_per_group,x_r_vec,in_h,in_w);
    //-----------------------将noise转换为矩阵的形式---------------------
    vector<vector<vector<uv64>>> noise_matrix;
    if(conv_type==1){
        noise_matrix=trans_to_matrix_group(map_num_per_group,noise_vec,in_h,in_w);
    }
    //需要将噪声进行分组
    //64*256*256 map
    ConvLayer ifmap_x(in_chn, in_h, in_w);
    ifmap_x.act=trans_to_matrix(vec_x,in_h,in_w);
    //-------------------------从文件里读取输入卷积核--------------------
    //形状64*64*3*3
    vector<vector<vector<uv64>>> filter_weight;
    filter_weight=readInputWeight(conv_filter,in_chn,test_params);
    //读入偏置
    uv64 bias=readBias(path_bias,test_params);
    Filter2D  filter(chn_per, map_num_per_group, f_h, f_w);
    ConvLayer ifmap(map_num_per_group, in_h, in_w);
    ConvLayer ifmap_pt(map_num_per_group, in_h, in_w);
    ConvLayer noise_map(map_num_per_group, in_h, in_w);
    Filter2D filter_totol(chn_per, in_chn, f_h, f_w);
   

    double start = currentDateTime();
    for(ui32 round=0;round<out_chn/chn_per;round++){
        cout<<"处理："<<round*chn_per<<"--->"<<round*chn_per+chn_per<<"通道"<<endl;
        ui32 curr_chn=0;
        for(ui32 ochn=chn_per*round;ochn<chn_per*round+chn_per;ochn++){
            for(ui32 inch=0;inch<in_chn;inch++){
                filter_totol.w[curr_chn][inch]=filter_weight[ochn][inch];                                   
            } 
            curr_chn++;         
        }
        filter_totol.b=bias;
        // //8*32*8*3*3、16*32*8*3*3、64×512×8×3×3
        vector<vector<vector<vector<uv64>>>>  w(group_num, std::vector<std::vector<std::vector<uv64>>>(chn_per, std::vector<std::vector<uv64>>(map_num_per_group, vector<uv64>(f_h,uv64(f_w)))));
        for(ui32 group=0;group<group_num;group++){
            for(ui32 ochn=0;ochn<chn_per;ochn++){
                ui32 curr=0;
                for(ui32 seg=group*map_num_per_group;seg<group*map_num_per_group+map_num_per_group;seg++){
                    w[group][ochn][curr]=filter_totol.w[ochn][seg];
                    curr++;   
                }
                    
            }   
        }
        for(ui32 ngroup=0;ngroup<group_num;ngroup++){
            //给图像赋值
            ifmap_pt.act=x_matrix[ngroup];
            ifmap.act=x_r_matrix[ngroup];
            if(conv_type==1){
                noise_map.act=noise_matrix[ngroup];
            }
            //给滤波器赋值
            filter.w=w[ngroup]; 
            auto ofmap_ref = conv_2d_pt(ifmap_pt, filter, true, opt::p);
            auto ct=comput_convolution(conv_type,sk,noise_vec,noise_map,ifmap,filter,test_params);
            
            auto ofmap = postprocess_conv(sk, ct, ofmap_ref.shape , test_params);
            auto eq = check_conv(ofmap, ofmap_ref);
            std::cout << "第"<<ngroup<<"组:"<<"Check "<< (eq?"succeeded":"failed") << std::endl;
            //将密文添加到ct_vec
            ct_vec[ngroup]=ct;

        }
        //---------------------------在密文上加入噪声并交给客户端解密---------------------------------
        uv64 r(test_params.phim);
        for(int i=0;i<test_params.phim;i++){
            r[i]=rand()%100;
        }
        CTVec temp_ct;
        uv64 ct_r = empty_encrypt(r, test_params);
        for(int i=0;i<ct_vec.size();i++){
            for(int j=0;j<ct_vec[i].size();j++){
                temp_ct.push_back(EvalAddPlain(ct_vec[i][j], ct_r,test_params));
            }
            ct_vec[i]=temp_ct;
            temp_ct.clear();
        }
        //--------------------------client对[x+r]进行解密进行求和----------------------------------------
        vector<uv64> sum(ct_vec[0].size(),uv64(test_params.phim));
        for(int seg=0;seg<ct_vec.size();seg++){
            for(int i=0;i<ct_vec[seg].size();i++){
                auto pt = packed_decode(Decrypt(sk, ct_vec[seg][i], test_params), test_params.p, test_params.logn);  
                for(int col=0;col<pt.size();col++){
                    if(seg==0){
                        sum[i][col]=pt[col];
                    }else{
                        sum[i][col] = mod(sum[i][col]+pt[col],test_params.p);    
                    }    
                }
            }
        }
        //对求和后的结果进行加密并交给服务器
        uv64 ct_share(test_params.phim);
        Ciphertext new_ct_prod(test_params.phim);
        CTVec ct_vec_sum(ct_vec[0].size(), Ciphertext(test_params.phim));
        for(int i=0;i<sum.size();i++){
            ct_share=packed_encode(sum[i], test_params.p, test_params.logn);
            new_ct_prod=Encrypt(sk,ct_share, test_params);
            ct_vec_sum[i]=new_ct_prod;
        }
        //-------------------------------server:服务器将噪声减掉----------------------------------------
        for(int i=0;i<r.size();i++){
            r[i]=r[i]*group_num;
        }
        for(int i=0;i<ct_vec_sum.size();i++){
            ct_vec_sum[i]=EvalSubPlainU(ct_vec_sum[i],r,test_params);
        }
        
        //明文计算
        auto ofmap_ref_x = conv_2d_pt(ifmap_x, filter_totol, true, opt::p);
        bool flag=(chn_pow2*2>test_params.phim)?true:false;
        auto new_ct_conv=(flag)?
             addBias_2stage(ct_vec_sum,filter_totol,in_h,in_w,test_params):
             addBias(ct_vec_sum,filter_totol,in_h,in_w,test_params);

        //密文解密之后的结果
        auto ofmap_x = postprocess_conv(sk, new_ct_conv, ofmap_ref_x.shape,test_params);
        auto eq = check_conv(ofmap_x, ofmap_ref_x);
        std::cout << "Check " << (eq?"succeeded":"failed") << std::endl;
        for(int i=0;i<new_ct_conv.size();i++)
        {
            ct_output.push_back(new_ct_conv[i]);
        }
    }
    double end = currentDateTime();
    cout<<"运行时间："<<(end-start)/1000/60<<"min"<<endl;
   
    ofstream o(path_output,ios::out);
    for(int i=0;i<ct_output.size();i++)
    {
        o<<vec_to_str(ct_output[i].a)<<endl;
        o<<vec_to_str(ct_output[i].b)<<endl;
    }
    o<<endl;
    o.close();
    //释放内存
    x_r_matrix.clear();
    x_r_matrix.shrink_to_fit();
   
    filter_totol.w.clear();
    
    
    return 0;
}

