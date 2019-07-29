/*
 This file is part of JustGarble.

    JustGarble is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    JustGarble is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with JustGarble.  If not, see <http://www.gnu.org/licenses/>.

*/
#include <iostream>
#include <algorithm>
#include <random>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <time.h>

#include "pke/gazelle.h"
#include "gc/gc.h"
#include "gc/util.h"
#include "gc/gazelle_circuits.h"
#include <cryptoTools/Common/Defines.h>
#include <fstream>
#include <regex>
#include<ctime>
#include <ratio>
#include <chrono>
using namespace osuCrypto;
using namespace lbcrypto;
using namespace std;

void relu_refU(uv64& din, uv64& dref, ui64 mask, ui64 p){
    dref[0] = (std::min((din[0]+din[1]) % p, p/2) + din[2]) % p;
}
void MAXCircuitU(GarbledCircuit *gc, BuildContext *garblingContext,
        const uv64& in_a, const uv64& in_b,const uv64 & zero,uv64& out) {
    ui64 leq;
    GEQCircuit(gc, garblingContext, in_a, in_b, leq);
    MUXCircuit(gc, garblingContext, in_a, zero, leq, out);
}

void ReLUCircuitU(GarbledCircuit *gc, BuildContext *context, const uv64& s_p,
        const uv64& s_p_2,const uv64 &zero,const uv64& s_c_x, const uv64& s_s_x, const uv64& s_s_y,
        uv64& s_c_y) {
    uv64 s_x, s_y;
    A2BCircuit(gc, context, s_p, s_c_x, s_s_x, s_x);
    MAXCircuitU(gc, context,  s_x ,s_p_2,zero,s_y);
    B2ACircuit(gc, context, s_p, s_y, s_s_y, s_c_y);
}


void buildRELULayerU(GarbledCircuit& gc, BuildContext& context,
        ui64 width, ui64 n_circ, ui64 p) {
    std::vector<uv64> in(3, uv64(width));
    uv64 out(width);
    uv64 s_p(width), s_p_2(width);

    uv64 zero(width);
    int n = n_circ*width*3;
    int m = n_circ*width;
    startBuilding(&gc, &context, n, m, n_circ*1000);
    gc.n_c = n_circ*width;
    CONSTCircuit(&gc, &context, p, width, s_p);
    CONSTCircuit(&gc, &context, p/2, width, s_p_2);
    CONSTCircuit(&gc, &context, (ui64)0, width, zero);
    for(ui64 i=0; i<n_circ; i++){
        for(ui64 j=0; j<3; j++){
            fill_vector(in[j], (j*n_circ+i)*width);
        }
        ReLUCircuitU(&gc, &context, s_p, s_p_2,zero, in[0], in[1], in[2], out);
        addOutputs(&gc, &context, out);
    }
    finishBuilding(&gc, &context);

    return;
}

vector<vector<uv64>> readInputTensor(string path){
    uv64 temp_line;
    vector<uv64 > temp_matrx;
    vector<vector<uv64> > Vec_tensor;
    string line;
    ifstream in(path); 
    regex pat_regex("[[:digit:]]+"); 
    while(getline(in, line)) {  
            for (sregex_iterator it(line.begin(), line.end(), pat_regex), end_it; it != end_it; ++it) { 
                uint64_t num= strtoull(it->str().c_str(), NULL,10);
                temp_line.push_back(num); 
            }
            if(temp_line.size()==0){
                Vec_tensor.push_back(temp_matrx);
                temp_matrx.clear();
            }else{
                temp_matrx.push_back(temp_line);
                temp_line.clear();       
            }
    }
    return Vec_tensor;
}

void save_result(vector<vector<uv64>> out,string path_output){
    ofstream output(path_output,ios::app);
    for(int i=0;i<out.size();i++){     
        for(int j=0;j<out[i].size();j++){
            output<<vec_to_str(out[i][j])<<endl;
        }
        
    }
    output.close();
}
void save_result_2stage(vector<uv64> noise,vector<uv64> out,string path_output){
    ofstream output(path_output,ios::out);
    int len=noise.size();
    for(int i=0;i<len;i++){
        output<<vec_to_str(noise[i])<<endl;
    }
    len=out.size();
    for(int i=0;i<len;i++){
        output<<vec_to_str(out[i])<<endl;
    }
    output<<endl;
    output.close();
}
uv64 execute_gc(SecretKey & sk,Ciphertext ct_vec, uv64 r2,const FVParams& params){
    
     //由于参数不一致，只能先解密再用自己的参数加密
    auto pt_x=packed_decode(Decrypt(sk, ct_vec, params), params.p, params.logn);
    auto pt_enc011= packed_encode(pt_x, params.p, params.logn);
    Ciphertext ct(params.phim);
    ct=Encrypt(sk, pt_enc011, params);
  
    //---------------------------server:产生噪声r,并将r加入到密文[x]中-----------------
    uv64 r=get_dug_vector(params.phim, params.p);

    auto pt_enc = packed_encode(r, params.p, params.logn);
    for(ui32 n=0; n<params.phim; n++){
        pt_enc[n] = pt_enc[n]*params.delta;
    }
    uv64 rr=NullEncrypt(pt_enc, params);
    Ciphertext ct_share = EvalAddPlain(ct, rr, params);

    //---------------------------client:对密文[x+r]进行解密得到明文x+r------------------
    auto pt = packed_decode(Decrypt(sk, ct_share, params), params.p, params.logn);
    //注：server拿到分享r,cliant拿到分享x+r
    uv64 r_nagtive(2048);
     for(ui32 n=0; n<r.size(); n++){
        r_nagtive[n] = mod(opt::p-r[n],opt::p);
    }

    ui64 n_circ = 2048;
    ui64 in_args = 3, out_args = 1;
    ui64 width = 25;
    ui64 p = params.p;
   
    std::vector<uv64> din = std::vector<uv64>(n_circ, uv64(in_args));
    std::vector<uv64> dref = std::vector<uv64>(n_circ, uv64(out_args));
    std::vector<uv64> dout_pt = std::vector<uv64>(n_circ, uv64(out_args));
    std::vector<uv64> dout = std::vector<uv64>(n_circ, uv64(out_args));

    //----------------------------给din赋值-----------------------------
    for(int i=0;i<2048;i++){
        din[i][0]=pt[i];
        din[i][1]=r_nagtive[i];
        din[i][2]=r2[i];
    }
    // }
    for(ui32 n=0; n<n_circ; n++){
        ui64 mask = ((1 << width)-1);
        relu_refU(din[n], dref[n], mask, p);
    }
   
    GarbledCircuit gc;
    BuildContext context;

    buildRELULayerU(gc, context, width, n_circ, p);
 
    BitVector inputBitMap(gc.n);
    BitVector outputBitMap(gc.m);

    // Pack plaintext into bits
    pack_inputs(din, inputBitMap, width);

    // Evaluate plaintext
    evaluate_pt(&gc, inputBitMap, outputBitMap);
    unpack_outputs(outputBitMap, dout_pt, width);

    // Garble the circuit
    InputLabels inputLabels(gc.n);
    OutputMap outputOTPBitMap(gc.m);

    garbleCircuit(&gc, inputLabels, outputOTPBitMap); 

    // Extract the input labels
    ExtractedLabels extractedLabels(gc.n);
    extractLabels(extractedLabels, inputLabels, inputBitMap);
    // Evaluate garbled circuits
    
    OutputLabels eval_outputs(gc.m);
    evaluate(&gc, extractedLabels, eval_outputs);
    // Map the outputs to 
    BitVector extractedMap(gc.m);
    mapOutputs(outputOTPBitMap, eval_outputs, extractedMap);
    unpack_outputs(extractedMap, dout, width);

    uv64 out;
    for(int i=0;i<2048;i++){
        out.push_back(dout[i][0]);
    }
    return out;

}
//读取参数
void save_noise(vector<uv64> out,string path_output){
    ofstream output(path_output,ios::app);
    int len=out.size();
    for(int i=0;i<len;i++){
        output<<vec_to_str(out[i])<<endl;
    } 
    //output<<vec_to_str(r)<<endl;
    output.close();
}
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
void print_result(int in_h, int in_w, vector<vector<uv64>> output,vector<uv64> r_nagative,const FVParams& params){
    uv64 y(params.phim);
    for(int i=0;i<output.size();i++){
        for(int j=0;j<output[i].size();j++){
            for(int k=0;k<params.phim;k++){
                y[k]=mod(output[i][j][k]+r_nagative[j][k],opt::p);
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
uv64 get_nagative(uv64 vec,const FVParams& params){
    uv64 r2_nagtive(vec.size());
    for(ui32 n=0; n<vec.size(); n++){
        r2_nagtive[n] = mod(params.p-vec[n],params.p);
    }  
    return r2_nagtive;   
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
vector<uv64> split_r_stage(uv64 r,int out_chn, int in_w, int in_h,const FVParams& params){
    ui32 chn_pow2 = nxt_pow2(in_h*in_w);
    ui32 row_pow2 = nxt_pow2(in_w);
    ui32 chn_pixels = row_pow2*in_h;
    ui32 num_ct_chn = div_ceil(chn_pixels, params.phim);
    ui32 rows_per_ct = params.phim/2/row_pow2;
    vector<uv64> vec;
    uv64 vec_temp;
    ConvLayer ofmap(out_chn, in_h, in_w);
    for(ui32 out_set=0; out_set<div_ceil(out_chn, 2); out_set++){
        for(ui32 out_row_idx = 0; out_row_idx < 2*num_ct_chn; out_row_idx++){
            ui32 curr_out_ct = out_row_idx + out_set*2*num_ct_chn;
            auto pt = r;
            ui32 src = 0;
            for(ui32 curr_seg=0; curr_seg<2; curr_seg++){
                ui32 curr_chn = out_set*2 + curr_seg;
                for(ui32 h_offset=0; h_offset<rows_per_ct; h_offset++){
                    ui32 h = out_row_idx+2*num_ct_chn*h_offset;
                    for(ui32 w=0; w<in_w; w++){
                        ofmap.act[curr_chn][h][w] = pt[src];
                        ofmap.act[curr_chn][h][w] = pt[src];
                        src++;
                    }
                }
                curr_chn++;
                if(curr_chn == out_chn){
                    break;
                }
            }
            
        }
    } 
    //将ofmap拼接成向量
    for(int chn=0;chn<ofmap.shape.chn;chn++){
        for(int h=0;h<ofmap.shape.h;h++){
            for(int w=0;w<ofmap.shape.w;w++){
                vec_temp.push_back(ofmap.act[chn][h][w]);
            }
            
        }
        vec.push_back(vec_temp);
        vec_temp.clear();
    }
    ofmap.act.clear();
    ofmap.act.shrink_to_fit();
    return vec;
}
vector<uv64> split_vec_stage(vector<uv64> vec_orgin,int out_chn, int in_w, int in_h,const FVParams& params){
    ui32 chn_pow2 = nxt_pow2(in_h*in_w);
    ui32 row_pow2 = nxt_pow2(in_w);
    ui32 chn_pixels = row_pow2*in_h;
    ui32 num_ct_chn = div_ceil(chn_pixels, params.phim);
    ui32 rows_per_ct = params.phim/2/row_pow2;
    vector<uv64> vec;
    uv64 vec_temp;
    ConvLayer ofmap(out_chn, in_h, in_w);
    for(ui32 out_set=0; out_set<div_ceil(out_chn, 2); out_set++){
        for(ui32 out_row_idx = 0; out_row_idx < 2*num_ct_chn; out_row_idx++){
            ui32 curr_out_ct = out_row_idx + out_set*2*num_ct_chn;
            auto pt = vec_orgin[curr_out_ct];
            ui32 src = 0;
            for(ui32 curr_seg=0; curr_seg<2; curr_seg++){
                ui32 curr_chn = out_set*2 + curr_seg;
                for(ui32 h_offset=0; h_offset<rows_per_ct; h_offset++){
                    ui32 h = out_row_idx+2*num_ct_chn*h_offset;
                    for(ui32 w=0; w<in_w; w++){
                        ofmap.act[curr_chn][h][w] = pt[src];
                        ofmap.act[curr_chn][h][w] = pt[src];
                        src++;
                    }
                }
                curr_chn++;
                if(curr_chn == out_chn){
                    break;
                }
            }
            
        }
    } 
    //已经恢复成图像，将ofmap拼接成向量
    for(int chn=0;chn<ofmap.shape.chn;chn++){
        for(int h=0;h<ofmap.shape.h;h++){
            for(int w=0;w<ofmap.shape.w;w++){
                vec_temp.push_back(ofmap.act[chn][h][w]);
            }
            
        }
        vec.push_back(vec_temp);
        vec_temp.clear();
    }
    ofmap.act.clear();
    ofmap.act.shrink_to_fit();
    return vec;
}

uv64 eliminate_noise(uv64 vec,uv64 r,const FVParams& params){
    uv64 y(params.phim);
    uv64 r_nagative=get_nagative(r,params);
    for(int i=0;i<params.phim;i++){
        y[i]=mod(vec[i]+r_nagative[i],opt::p);
    }
    return y;
}


void relu(SecretKey & sk,CTVec input_ct,string path_output,int in_h,int in_w,const FVParams& params){
    vector<vector<uv64>> vec_output;
    uv64 out;
    uv64 r2(params.phim);
    for(int i=0;i<params.phim;i++){
        r2[i]=rand()%100;
    }
    vector<uv64> vec_r;    
    vec_r=split_vec(r2,in_h,in_w,params);
    vector<uv64> vec_r_nagative=get_vec_nagative(vec_r,params);
    vector<uv64> temp;
    save_noise(vec_r,path_output);

    double start, stop;
    start = currentDateTime();
    for(int i=0;i<input_ct.size();i++){
        out=execute_gc(sk,input_ct[i],r2,params);
        temp=split_vec(out,in_h,in_w,params);
        vec_output.push_back(temp);
        temp.clear();
        out.clear();
    }
    stop= currentDateTime();
    save_result(vec_output,path_output); 
    ofstream out0(path_output,ios::app);
    out0<<endl;
    out0.close();
    out.clear();
    out.shrink_to_fit();
    vec_output.clear();
    vec_output.shrink_to_fit();
    r2.clear();
    r2.shrink_to_fit();
    temp.clear();
    temp.shrink_to_fit();
    vec_r.clear();
    vec_r.shrink_to_fit();
    vec_r_nagative.clear();
    vec_r_nagative.shrink_to_fit();
}
void relu_2stage(SecretKey & sk,CTVec input_ct,string path_output,int out_chn,int in_h,int in_w,const FVParams& params){
    uv64 out;
    vector<uv64> vec_out;
    uv64 r2(params.phim);
    for(int i=0;i<params.phim;i++){
            r2[i]=rand()%100;
    }
    vector<uv64> r2_split=split_r_stage(r2,out_chn, in_w, in_h, params);
    vector<uv64> temp;
    double start, stop;
    start = currentDateTime();
    
    for(int i=0;i<input_ct.size();i++){
        out=execute_gc(sk,input_ct[i],r2,params);//每个都加上了噪声r2
        vec_out.push_back(out);
        out.clear();
    }
    //对图像进行恢复形状
    vector<uv64> new_vec_out=split_vec_stage(vec_out,out_chn,in_w, in_h,params);
    
    stop= currentDateTime();
    // cout << " relu_2 cost_time: " << (stop-start)<<endl;;
    
    vector<uv64> nagative=get_vec_nagative(r2_split,params);
    save_result_2stage(r2_split,new_vec_out,path_output); ;
    out.clear();
    out.shrink_to_fit();
    vec_out.clear();
    vec_out.shrink_to_fit();
    r2.clear();
    r2.shrink_to_fit();
    temp.clear();
    temp.shrink_to_fit();
    nagative.clear();
    nagative.shrink_to_fit();
}

using namespace std::chrono;
int main() {
    //cout<<"---------------------------执行非线性relu函数-------------------------------"<<endl;
    ui64 z = RootOfUnity(opt::phim << 1, opt::q);
    ui64 z_p = RootOfUnity(opt::phim << 1, opt::p);
    ftt_precompute(z, opt::q, opt::logn);
    ftt_precompute(z_p, opt::p, opt::logn);
    encoding_precompute(opt::p, opt::logn);
    precompute_automorph_index(opt::phim);

    ui64 nRep = 1;
    ui32 vec_size = 2048;
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
    string path_input="../result/5_3_output.txt";
    string path_input_key="../params/outkey.txt";
    string path_output="../result/relu_5_3_out.txt";
    string path_parameter="../params/con5_3_param.txt";
    //读入参数
    vector<ui64> v_paramater;
    v_paramater=read_paramater(path_parameter);
    ui32 out_chn = v_paramater[0];
    ui32 in_chn = v_paramater[1]; 
    ui32 in_h = v_paramater[2];
    ui32 in_w = v_paramater[3];
    ui32 f_h = v_paramater[4];
    ui32 f_w = v_paramater[5];
    ui32 conv_type =v_paramater[6];
    v_paramater.clear();
    v_paramater.shrink_to_fit();
    //---------------------------从文件里读取密钥--------------------------
    vector<vector<uv64>> vec=readInputTensor(path_input_key);
    SecretKey sk(2048);
    sk.s=vec[0][0]; 
    vec.clear();
    vec.shrink_to_fit();
    
    //---------------------------server:读取密文--------------------------
    CTVec input_ct;
    Ciphertext ct(test_params.phim);
    vector<vector<uv64> > ct_conv=readInputTensor(path_input);
    for(int i=0;i<ct_conv[0].size();i=i+2){
        ct.a=ct_conv[0][i];
        ct.b=ct_conv[0][i+1];
        input_ct.push_back(ct);
        ct.a.clear();
        ct.b.clear();
    }
    ui32 chn_pow2 = nxt_pow2(in_h*in_w);
    bool flag=(chn_pow2*2>test_params.phim)?true:false;
    (flag)?relu_2stage(sk,input_ct,path_output,out_chn,in_h,in_w,test_params):
           relu(sk,input_ct,path_output,in_h,in_w,test_params);
    //--------------------------server:产生噪声r2------------------------
    cout<<"执行完毕！"<<endl;
    input_ct.clear();
    input_ct.shrink_to_fit();
    ct.a.clear();
    ct.b.shrink_to_fit();
    return 0;
}

