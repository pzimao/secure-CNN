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
uv64 get_nagative(uv64 vec,const FVParams& params){
    uv64 r2_nagtive(vec.size());
    for(ui32 n=0; n<vec.size(); n++){
        r2_nagtive[n] = mod(params.p-vec[n],params.p);
    }  
    return r2_nagtive;   
}
uv64 execute_gc(uv64 x_r,uv64 r, uv64 r2,const FVParams& params){
    //---------------------------client:对密文[x+r]进行解密得到明文x+r------------------
    uv64 r_nagaive=get_nagative(r,params);

    ui64 n_circ = x_r.size();
    ui64 in_args = 3, out_args = 1;
    ui64 width = 25;
    ui64 p = params.p;
   
    std::vector<uv64> din = std::vector<uv64>(n_circ, uv64(in_args));
    std::vector<uv64> dref = std::vector<uv64>(n_circ, uv64(out_args));
    std::vector<uv64> dout_pt = std::vector<uv64>(n_circ, uv64(out_args));
    std::vector<uv64> dout = std::vector<uv64>(n_circ, uv64(out_args));

    //----------------------------给din赋值-----------------------------
    for(int i=0;i<n_circ;i++){
        din[i][0]=x_r[i];
        din[i][1]=r_nagaive[i];
        din[i][2]=r2[i];
    }
   
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
    for(int i=0;i<n_circ;i++){
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


uv64 eliminate_noise(uv64 vec,uv64 r,const FVParams& params){
    uv64 y(vec.size());
    uv64 r_nagative=get_nagative(r,params);
    for(int i=0;i<vec.size();i++){
        y[i]=mod(vec[i]+r_nagative[i],opt::p);
    }
    return y;
}


void relu(uv64 x_r,uv64 r,uv64 r2,string path_output,int num_row,const FVParams& params){
    
    string path_descriptor="../result/descriptor_relu.txt";
    uv64 out_x_r=execute_gc(x_r,r,r2,params);
    
    uv64 vec_y=eliminate_noise(out_x_r,r2,params);
    cout<<vec_to_str(vec_y);

    vector<float> feature_descritor;
    for(int i=0;i<vec_y.size();i++){
        int64_t temp=0;
        if(vec_y[i] > opt::p/2){
            temp=vec_y[i]-opt::p;
        }else{
            temp=vec_y[i];
        }
        feature_descritor.push_back((float)temp/10000);
    }
    // //将结果输出到文件中
    ofstream out1(path_descriptor,ios::out);
    out1<<vec_to_str(feature_descritor)<<endl;
    out1.close();

    //将x+r和r传到下一层
    ofstream out(path_output,ios::out);
    out<<vec_to_str(r2)<<endl;
    out<<vec_to_str(out_x_r)<<endl;
    out<<endl;
    out.close();
}

uv64 preprocess_r_vec(uv64 r, const ui32 vec_size, const ui32 num_rows, const FVParams& params){

    auto prod = uv64(num_rows);
    ui32 sz_pow2 = nxt_pow2(vec_size);
    ui32 pack_factor = (params.phim / nxt_pow2(vec_size));
    ui32 set_size = nxt_pow2(num_rows)/pack_factor;
    for(ui32 row=0; row<num_rows; row++){
        ui32 curr_set = (row / set_size);
        prod[row] = r[(row % set_size) + sz_pow2*curr_set];
    }
    return prod;
}

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
    string path_input="../result/fc_1_out.txt";
    string path_input_key="../params/outkey.txt";
    string path_output="../result/relu_fc1_out.txt";
    string path_parameter="../params/fc_1_param.txt";
    //读入参数
    vector<ui64> v_paramater;
    v_paramater=read_paramater(path_parameter);
    ui32 num_row = v_paramater[0];
    ui32 num_col = v_paramater[1]; 
    ui32 in_h = v_paramater[2];
    ui32 in_w = v_paramater[3];
    v_paramater.clear();
    v_paramater.shrink_to_fit();

    //--------------------------从文件里读取密钥--------------------------
    
    vector<vector<uv64>> vec=readInputTensor(path_input_key);
    SecretKey sk(2048);
    sk.s=vec[0][0]; 
    vec.clear();
    vec.shrink_to_fit();
    
    //--------------------------server:读取密文--------------------------
    //--------------------------server:产生噪声r2-----------------------------
    uv64 r2(num_row);
    for(int i=0;i<num_row;i++){
        r2[i]=rand()%100;
    }
    uv64 r_nagative=get_nagative(r2,test_params);
    //---------------------------server:从fc1中读取数据--------------------------
    vector<vector<uv64>> input=readInputTensor(path_input);
    uv64 r=input[0][0];
    uv64 x_r=input[0][1];
    uv64 rr=preprocess_r_vec(r, test_params.phim, num_row, test_params);
    relu(x_r,rr,r2,path_output,num_row,test_params);
    return 0;
}

