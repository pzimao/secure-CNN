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
#include<ctime>
#include <ratio>
#include <chrono>
using namespace std;
using namespace lbcrypto;


bool check_convU(const ConvLayer& ofmap, const ConvLayer& ofmap_ref){


    bool ret = true;

    if((ofmap.shape.chn != ofmap_ref.shape.chn) ||
            (ofmap.shape.h != ofmap_ref.shape.h) ||
            (ofmap.shape.w != ofmap_ref.shape.w)){
        return false;
    } else {
        for(ui32 chn=0; chn<ofmap.shape.chn; chn++){
           
            for (ui32 h = 0; h < ofmap.shape.h; h++){
                int count=0;
                for (ui32 w = 0; w < ofmap.shape.w; w++){
                    if(ofmap.act[chn][h][w] != ofmap_ref.act[chn][h][w]){
                        count++;
                        ret = false;
                        //return false;
                        //std::cout<<"ofmap.act:"<<ofmap.act[chn][h][w]<<"!"<<"ofmap_ref.act"<<ofmap_ref.act[chn][h][w]<<" ";
                    }
                    
                }
                std::cout<<"chn:"<<chn<<":"<<"h:"<<h<<": "<<count<<endl;
               
            }
            
            // std::cout << "channel " << chn << " matched" << std::endl;
        }

        return ret;
    }
}

void update_ct_matU(CTMat& ct_mat, const ui32 ct_idx,
        const SecretKey& sk, const uv64& in,
        const ui32 window_size, const ui32 num_windows, const FVParams& params){
    uv64 pt = packed_encode(in, params.p, params.logn);
    for(ui32 w=0; w<num_windows; w++){
        ct_mat[w][ct_idx] = Encrypt(sk, pt, params);

        // Scale for the next iteration
        for (ui32 i=0; i<params.phim; i++){
            pt[i] = ((pt[i] << window_size) % params.p);
        }
    }
} 
EncMat preprocess_filter_2stageU(const Filter2D& filter, const ConvShape& shape,
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
        EncMat enc_filter(num_filter_rows, std::vector<uv64>(num_windows, uv64(params.phim)));
        // std::cout << "Number of filters: " << num_filter_rows << std::endl;

        for(ui32 in_set=0; in_set<div_ceil(filter.shape.in_chn, 2); in_set++){
            for(ui32 in_row_idx=0; in_row_idx<2*num_ct_chn; in_row_idx++){
                for(ui32 f_w=0; f_w<filter.shape.f_w; f_w++){
                    for(ui32 f_h=0; f_h<filter.shape.f_h; f_h++){
                        for(ui32 out_set=0; out_set<div_ceil(filter.shape.out_chn, 2); out_set++){
                            bool first_skip = (in_row_idx+offset_h < f_h);
                            bool last_skip = (in_row_idx+offset_h >= (2*num_ct_chn+f_h));

                            for(ui32 curr_loop=0; curr_loop<2; curr_loop++){
                                // Create a vector with filter_coeff and zeros
                                uv64 filter_base(params.phim, 0);
                                for(ui32 curr_offset=0; curr_offset<2; curr_offset++){
                                    ui32 curr_in = 2*in_set + curr_offset;
                                    ui32 curr_out = 2*out_set + (curr_offset+curr_loop)%2;

                                    ui64 coeff = ((curr_in >= filter.shape.in_chn) || (curr_out >= filter.shape.out_chn))? 0:
                                            filter.w[curr_out][curr_in][f_h][f_w];

                                    ui32 dest = curr_offset*params.phim/2;
                                    /*if(coeff != 0){
                                        std::cout << "coeff: " << coeff << " dest: " << dest << std::endl;
                                    }*/
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
                                    enc_filter[enc_row][w] = NullEncrypt(decomposed_row[w], params);
                                }
                                enc_row++;
                            }
                        }
                    }
                }
            }
        }

        return enc_filter;
    } else {
        ui32 chn_per_ct = params.phim/chn_pow2;
        ui32 chn_per_seg = chn_per_ct/2;

        ui32 in_ct = div_ceil(filter.shape.in_chn, chn_per_ct);
        ui32 out_ct = div_ceil(filter.shape.out_chn, chn_per_ct);
        ui32 inner_loop = chn_per_ct;
        ui32 rot_per_in = inner_loop*filter.shape.f_h*filter.shape.f_w;

        // Create the diagonal rotation of the plaintext matrix
        ui32 enc_row = 0;
        ui32 num_filter_rows = in_ct*rot_per_in*out_ct;
        EncMat enc_filter(num_filter_rows, std::vector<uv64>(num_windows, uv64(params.phim)));

        for(ui32 in_chn_base=0; in_chn_base<filter.shape.in_chn; in_chn_base+=chn_per_ct){
            for(ui32 f_h=0; f_h<filter.shape.f_h; f_h++){
                for(ui32 f_w=0; f_w<filter.shape.f_w; f_w++){
                    for(ui32 out_chn_base=0; out_chn_base<filter.shape.out_chn; out_chn_base+=chn_per_ct){
                        for(ui32 curr_loop=0; curr_loop<inner_loop; curr_loop++){
                            // Create a vector with filter_coeff and zeros
                            uv64 filter_base(params.phim, 0);
                            for(ui32 curr_offset=0; curr_offset<chn_per_ct; curr_offset++){
                                ui32 delta_in = (curr_offset % chn_per_ct);
                                ui32 delta_out = ((curr_offset+curr_loop) % chn_per_seg +
                                    (curr_loop/chn_per_seg)*chn_per_seg +
                                    (curr_offset/chn_per_seg)*chn_per_seg) % chn_per_ct;

                                ui32 curr_in = in_chn_base + delta_in;
                                ui32 curr_out = out_chn_base + delta_out;

                                ui64 coeff = ((curr_in >= filter.shape.in_chn) || (curr_out >= filter.shape.out_chn))? 0:
                                        filter.w[curr_out][curr_in][f_h][f_w];
                                /* std::cout << "curr_in: " << curr_in
                                        << " curr_out: " << curr_out
                                        << " coeff: " << coeff << std::endl; */

                                ui32 dest = curr_offset*chn_pow2;
                                /*if(coeff != 0){
                                    std::cout << "coeff: " << coeff << " dest: " << dest << std::endl;
                                }*/
                                for(ui32 curr_h=0; curr_h<shape.h; curr_h++){
                                    for(ui32 curr_w=0; curr_w<shape.w; curr_w++){
                                        bool zero = ((curr_w+f_w) < offset_w) ||
                                                ((curr_w+f_w) >= (offset_w+shape.w)) ||
                                                ((curr_h+f_h) < offset_h) ||
                                                ((curr_h+f_h) >= (offset_h+shape.h));
                                        filter_base[dest] = zero? 0: coeff;
                                        dest++;
                                    }
                                }
                            }

                            auto pt_row = packed_encode(filter_base, params.p, params.logn);
                            auto decomposed_row = base_decompose(pt_row, window_size, num_windows);
                            for(ui32 w=0; w<num_windows; w++){
                                enc_filter[enc_row][w] = NullEncrypt(decomposed_row[w], params);
                            }
                            enc_row++;
                        }
                    }
                }
            }
        }

        return enc_filter;
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
                                if(curr_chn<in.shape.chn){
                                    packed_rows[dest] = in.act[curr_chn][curr_h][curr_w];
                                    dest++;
                                }else{
                                    packed_rows[dest]=0;
                                }
                                
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

ConvLayer conv_2d_ptU(const ConvLayer& in, const Filter2D& filter, bool same, const ui32 p){
    ui32 out_h = in.shape.h - ((same) ? 0 : (filter.shape.f_h - 1));
    ui32 out_w = in.shape.w - ((same) ? 0 : (filter.shape.f_w - 1));
    cout<<"p:"<<p<<endl;
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
                            bool zero = (same && (in_h >= in.shape.h || in_w >= in.shape.w));
                            ui64 in_act = zero ? 0:in.act[m][in_h][in_w];
                            out.act[n][h][w] += (filter.w[n][m][f_h][f_w]*in_act);
                        }
                    }
                }
                //cout<<out.act[n][h][w]<<" ";
                out.act[n][h][w] = out.act[n][h][w]%p;
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
                // cout<<filter.b[n]<<" ";
                if(filter.b[n]>(opt::p)/2){
                    sum= filter.b[n]-opt::p;
                }else{
                    sum=filter.b[n];
                } 
                //out.act[n][h][w] = filter.b[n];
                for(ui32 m=0; m<filter.shape.in_chn; m++){
                    for (ui32 f_h = 0; f_h < filter.shape.f_h; f_h++){
                        for (ui32 f_w = 0; f_w < filter.shape.f_w; f_w++){
                            ui32 in_h = h+f_h-offset_h;
                            ui32 in_w = w+f_w-offset_w;
                            // Uses the wrap-around property of ui32 to discard negative
                            bool zero = (same && (in_h >= in.shape.h || in_w >= in.shape.w));
                            ui64 in_act = zero ? 0:in.act[m][in_h][in_w];
                            // 如果这个数是负数，则用带符号数处理
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

//从文件里读小数
vector<vector<vector<vector<uint64_t>>>>readInputWeight(string path,int in_chn,const FVParams & params){

    vector<uint64_t> temp_line;
    vector<vector<uint64_t> > temp_matrx;
    vector<vector<vector<uint64_t>>> temp_weight;
    vector<vector<vector<vector<uint64_t>>>> weight;
    string line;
    ifstream in(path);
    //regex pat_regex("[-|+]?([0-9]*.[0-9]+|[0-9]+)"); 
    regex pat_regex("[-]?([0-9]+.[0-9]+)"); 
    //regex pat_regex("[[:digit:]]+");  
    while(getline(in, line)) { 
            for (sregex_iterator it(line.begin(), line.end(), pat_regex), end_it; it != end_it; ++it) {
                float num=Round(stof(it->str().c_str()),3)*1000;  
                //cout<<num<<" ";  
                //uint64_t num= strtoull(it->str().c_str(), NULL,10);
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
//读取偏置
uv64 readBias(string path,const FVParams & params){
    uv64 bias;
    ifstream in(path);
    string line;
    regex pat_regex("[-]?([0-9]+.[0-9]+)"); 
    //regex pat_regex("[[:digit:]]+");  
    while(getline(in, line)) {
            for (sregex_iterator it(line.begin(), line.end(), pat_regex), end_it; it != end_it; ++it) {
                ui64 number=0;
                float num=Round(stof(it->str().c_str()),5)*100000;      
                if(num<0){
                    number=num+params.p;
                }else{
                    number=num;
                }
                bias.push_back(number); 
            }
            if(line.size()==0){
                break;
            }          
    }            
    return bias;
}
//从文件里读取多个矩阵
vector<vector<uv64>> readInputTensor(string path){
    bool flag=false;
    uv64 temp_line;
    vector<uv64 > temp_matrx;
    vector<vector<uv64> > Vec_tensor;
    string line;
    ifstream in(path); 
    regex pat_regex("[[:digit:]]+"); 
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
vector<vector<uv64>> readInputImage(string path,const FVParams & params){

    uv64 temp_line;
    vector<uv64> temp_matrx;
    vector<vector<uv64>>image;
    string line;
    ifstream in(path);
    //regex pat_regex("[-|+]?([0-9]*.[0-9]+|[0-9]+)"); 
    regex pat_regex("[-]?([0-9]+.[0-9]+)"); 
    //regex pat_regex("[[:digit:]]+"); 
    while(getline(in, line)) { 
            for (sregex_iterator it(line.begin(), line.end(), pat_regex), end_it; it != end_it; ++it) {
                float num=Round(stof(it->str().c_str()),2)*100;
                if(num<0){
                    num=num+params.p;
                }
                temp_line.push_back(num); 
            }
            //cout<<vec_to_str(temp_line)<<endl;
            if(temp_line.size() >0){
                temp_matrx.push_back(temp_line);
                temp_line.clear();  
            }else{
                image.push_back(temp_matrx);
                temp_matrx.clear();             
            }              
        }
             
    return image;
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

void print_result(ui32 out_chn, ui32 in_h, ConvLayer ofmap_ref){
    for(ui32 chn=0; chn<out_chn; chn++){
        for(ui32 h=0; h<in_h; h++){
            std::cout << vec_to_str(ofmap_ref.act[chn][h]) << std::endl;
        }
        cout<<endl;
    }

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
    ui32 chn_per_ct = params.phim/chn_pow2;//每个密文处理的map数量
    ui32 out_ct = div_ceil(filter.shape.out_chn, chn_per_ct);
    vector<uv64> bias_vec;
    uv64 temp1;
    // cout<<"filter.shape.out_chn:"<<filter.shape.out_chn<<endl;
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
    ui32 chn_pixels = row_pow2*in_h;//一个map的像素点个数
    ui32 num_ct_chn = div_ceil(chn_pixels, params.phim);//每个map需要的密文数
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
ConvLayer postprocess_convU(const SecretKey& sk, const CTVec& ct_vec,
         const ConvShape& shape, const FVParams& params){
    ui32 chn_pow2 = nxt_pow2(shape.h*shape.w);
    ui32 row_pow2 = nxt_pow2(shape.w);

    if (row_pow2*2 > params.phim){
        throw std::logic_error("Rows larger than half a ciphertext not supported");
    } else if(chn_pow2*2 > params.phim) {
        ui32 chn_pixels = row_pow2*shape.h;
        ui32 num_ct_chn = div_ceil(chn_pixels, params.phim);
        ui32 rows_per_ct = params.phim/2/row_pow2;

        ConvLayer ofmap(shape.chn, shape.h, shape.w);
        for(ui32 out_set=0; out_set<div_ceil(shape.chn, 2); out_set++){
            for(ui32 out_row_idx = 0; out_row_idx < 2*num_ct_chn; out_row_idx++){
                ui32 curr_out_ct = out_row_idx + out_set*2*num_ct_chn;
                auto pt = packed_decode(Decrypt(sk, ct_vec[curr_out_ct], params), params.p, params.logn);
                ui32 src = 0;
                for(ui32 curr_seg=0; curr_seg<2; curr_seg++){
                    ui32 curr_chn = out_set*2 + curr_seg;
                    for(ui32 h_offset=0; h_offset<rows_per_ct; h_offset++){
                        ui32 h = out_row_idx+2*num_ct_chn*h_offset;
                        for(ui32 w=0; w<shape.w; w++){
                            ofmap.act[curr_chn][h][w] = pt[src];
                            src++;
                        }
                    }
                    curr_chn++;
                    if(curr_chn == shape.chn){
                        break;
                    }
                }
                
            }
        }

        return ofmap;
    } else {
        ui32 curr_chn = 0;
        ConvLayer ofmap(shape.chn, shape.h, shape.w);
        for(ui32 curr_out_ct = 0; curr_out_ct < ct_vec.size(); curr_out_ct++){
            auto pt = packed_decode(Decrypt(sk, ct_vec[curr_out_ct], params), params.p, params.logn);
            for(ui32 src_base=0; src_base<params.phim; src_base+=chn_pow2){
                ui32 src = src_base;
                for(ui32 h=0; h<shape.h; h++){
                    for(ui32 w=0; w<shape.w; w++){
                        ofmap.act[curr_chn][h][w] = pt[src];
                        src++;
                    }
                }
                curr_chn++;
                if(curr_chn == shape.chn){
                    break;
                }
            }
        }

        return ofmap;
    }
}

void save_result(const ConvLayer & map, const FVParams& params){
    string path="./cheak_result/1_1_conv.txt";
    ofstream cheack(path,ios::out);
    float value=0;
    vector<float> temp;
    for(int i=0;i<map.act.size();i++){
        for(int j=0;j<map.act[i].size();j++){
            for(int k=0;k<map.act[i][j].size();k++){
                 if(map.act[i][j][k]<params.p/2){
                    value=map.act[i][j][k]/100000.0;
                }else{
                    ui64 a=map.act[i][j][k];
                    value=((params.p-map.act[i][j][k])/100000.0)*(-1);
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

void myConv2d() {
    //------------------------执行第一层卷积层--------------------------"<<endl;
    string path_parameter="../params/con1_1_param.txt";
    string path_inputMat="../params/image.txt";
    string path_filter="../params/1_1w.txt";
    string path_output="../result/1_1_output.txt";
    string path_outkey="../params/outkey.txt";
    string path_bias="../params/1_1b.txt";
    vector<ui64> v_paramater;
    v_paramater=read_paramater(path_parameter);
    //------------------------从文件中读取参数并打印-------------------------------------
    ui32 out_chn = v_paramater[0];
    ui32 in_chn = v_paramater[1]; 
    ui32 in_h = v_paramater[2];
    ui32 in_w = v_paramater[3];
    ui32 f_h = v_paramater[4];
    ui32 f_w = v_paramater[5];
    ui32 conv_type =v_paramater[6];
    v_paramater.clear();
    v_paramater.shrink_to_fit();
  
    //------------------ -----Setup Parameters --------------------------------
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
    FVParams test_params = fast_params;
    ui32 w_sz = 8;
    test_params.window_size=w_sz;
    ui32 pt_window_size=10;
    ui32 pt_num_windows = 1 + floor(log2(test_params.p))/pt_window_size;
    
   //----------------------从文件里读取输入图像--------------------------------
    vector<vector<uv64>> image;
    image=readInputImage(path_inputMat,test_params);
    //-----------------------Synthetic Data -------------------
    ConvLayer ifmap(in_chn, in_h, in_w);
    for(ui32 chn=0; chn<in_chn; chn++){
        for(ui32 h=0; h<in_h; h++){
            ifmap.act[chn][h]=image[chn][h];
        }
    }
    image.clear();
    image.shrink_to_fit();
    //-----------------------读取输入卷积核以及偏置-----------------------
    vector<vector<vector<uv64>>> Vec_filter;
    uv64 bias;
    Vec_filter=readInputWeight(path_filter,in_chn,test_params);
    bias=readBias(path_bias,test_params);
    Filter2D filter(out_chn, in_chn, f_h, f_w);
    for(ui32 ochn=0; ochn<out_chn; ochn++){
    //--------------------get_prng 这个函数的副作用---------------------------
        std::normal_distribution<double> distribution(0,40.0);
        auto& prng = get_prng();
        distribution(prng);
    //-----------------------------------------------------------------
        for(ui32 ichn=0; ichn<in_chn; ichn++){
            for(ui32 h=0; h<f_h; h++){
                
                filter.w[ochn][ichn][h]=Vec_filter[ochn][ichn][h];
            }
        }
    }
    filter.b=bias;
    Vec_filter.clear();
    Vec_filter.shrink_to_fit();
    auto ofmap_ref = conv_2d_pt(ifmap, filter, true, opt::p);
    //-----------------------打印没有加密时的输出map--------------------------
    double start, stop;
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
    //-----------------------KeyGen -----------------------
    auto kp = KeyGen(test_params);
    EvalAutomorphismKeyGen(kp.sk, index_list, test_params);
    
    index_list.clear();
    index_list.shrink_to_fit();
    //---------------------Preprocess Filter ------------------
    auto enc_filter = (conv_type) ?
            preprocess_filter_2stage(filter, ifmap.shape, pt_window_size, pt_num_windows, test_params):
            preprocess_filter(filter, ifmap.shape, pt_window_size, pt_num_windows, test_params);

    //--------------------Preprocess Vector ------------------
    auto ct_mat = preprocess_ifmapU(kp.sk, ifmap, pt_window_size, pt_num_windows, test_params);
    //--------------------Conv2D ------------------------------
    auto ct_conv = (conv_type) ?
            conv_2d_2stage_online(ct_mat, enc_filter, filter.shape, ifmap.shape, test_params):
            conv_2d_online(ct_mat, enc_filter, filter.shape, ifmap.shape, test_params);

    //需要在结果上加入偏置的值
    bool flag=(chn_pow2*2>test_params.phim)?true:false;

    auto new_ct_conv=(flag)?
        addBias_2stage(ct_conv,filter,in_h,in_w,test_params):
        addBias(ct_conv,filter,in_h,in_w,test_params);

    stop = currentDateTime();
    cout << " conv2d_1 cost_time: " << (stop-start)<<endl;

    //将密文下卷积的结果写入文件中
    ofstream out(path_output,ios::out);
    int len = new_ct_conv.size();
    for(int i=0;i<len;i++)
    {
        out<<vec_to_str(new_ct_conv[i].a)<<endl;
        out<<vec_to_str(new_ct_conv[i].b)<<endl;
    }
    out<<endl;
    out.close();
     //将密钥保存到文件中
    ofstream outkey(path_outkey,ios::out);
    outkey<<vec_to_str(kp.sk.s)<<endl;
    outkey<<endl;
    outkey.close();
   
    //------------------- Post-Process ---------------------
    auto ofmap = postprocess_convU(kp.sk, new_ct_conv, ofmap_ref.shape, test_params);
    //----------------------- Check ------------------------
    auto eq = check_conv(ofmap, ofmap_ref);
    std::cout << "Check " << (eq?"succeeded!!!":"failed") << std::endl;

    ifmap.act.clear();
    ifmap.act.shrink_to_fit();

    filter.b.clear();
    filter.b.shrink_to_fit();
    filter.w.clear();
    filter.w.shrink_to_fit();

    enc_filter.clear();
    enc_filter.shrink_to_fit();

    ct_mat.clear();
    ct_mat.shrink_to_fit();

    ct_conv.clear();
    ct_conv.shrink_to_fit();

    ofmap_ref.act.clear();
    ofmap_ref.act.shrink_to_fit();
    
    ofmap.act.clear();
    ofmap.act.shrink_to_fit();


}

int main(){
    myConv2d();
    return 0;
}