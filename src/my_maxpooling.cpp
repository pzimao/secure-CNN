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
#include <fstream>
#include <regex>
#include "pke/gazelle.h"
#include "gc/gc.h"
#include "gc/util.h"
#include "gc/gazelle_circuits.h"
#include <cryptoTools/Common/Defines.h>

using namespace osuCrypto;
using namespace lbcrypto;
using namespace std;


void pool2_refU(uv64& din, uv64& dref, ui64 mask, ui64 p){
    ui64 curr_max = 0;
    curr_max = std::min((din[0] + din[4]) % p, p/2);
    curr_max = std::max((din[1] + din[5]) % p, curr_max);
    curr_max = std::max((din[2] + din[6]) % p, curr_max);
    curr_max = std::max((din[3] + din[7]) % p, curr_max);
    dref[0] = (curr_max + din[8]) % p;
}
void MAXCircuitU1(GarbledCircuit *gc, BuildContext *garblingContext,
        const uv64& in_a, const uv64& in_b, uv64& out) {
    ui64 leq;
    LEQCircuit(gc, garblingContext, in_a, in_b, leq);
    MUXCircuit(gc, garblingContext, in_a, in_b, leq, out);
}
void MAXCircuitU2(GarbledCircuit *gc, BuildContext *garblingContext,
        const uv64& in_a, const uv64& in_b, uv64& out) {
    ui64 leq;
    GEQCircuit(gc, garblingContext, in_a, in_b, leq);
    MUXCircuit(gc, garblingContext, in_a, in_b, leq, out);
}
void Pool2CircuitUback(GarbledCircuit *gc, BuildContext *context, const uv64& s_p,
        const uv64& s_p_2, const std::vector<uv64>& s_c_x,
        const std::vector<uv64>& s_s_x, const uv64& s_s_y, uv64& s_c_y) {
    std::vector<uv64> s_x(4);
    uv64 s_in = s_p_2;
    uv64 s_out;
    for(ui64 i=0; i<4; i++){    
        A2BCircuit(gc, context, s_p, s_c_x[i], s_s_x[i], s_x[i]);
        if(i==0){
            MAXCircuitU2(gc, context, s_x[i], s_in, s_out);
        }else{
            MAXCircuitU1(gc, context, s_x[i], s_in, s_out);
        }
     
        s_in = s_out;
    }
    B2ACircuit(gc, context, s_p, s_out, s_s_y, s_c_y);
}
void Pool2CircuitU(GarbledCircuit *gc, BuildContext *context, const uv64& s_p,
        const uv64& s_p_2, const std::vector<uv64>& s_c_x,
        const std::vector<uv64>& s_s_x, const uv64& s_s_y, uv64& s_c_y) {
    std::vector<uv64> s_x(4);
    uv64 s_in = s_p_2;
    uv64 s_out_1;   
    uv64 s_out_2;
    uv64 s_out;
    
    A2BCircuit(gc, context, s_p, s_c_x[0], s_s_x[0], s_x[0]);
    A2BCircuit(gc, context, s_p, s_c_x[1], s_s_x[1], s_x[1]);
    A2BCircuit(gc, context, s_p, s_c_x[2], s_s_x[2], s_x[2]);
    A2BCircuit(gc, context, s_p, s_c_x[3], s_s_x[3], s_x[3]);


    MAXCircuitU1(gc, context, s_x[0], s_x[1], s_out_1);
    MAXCircuitU1(gc, context, s_x[2], s_x[3], s_out_2);
    MAXCircuitU1(gc, context, s_out_1, s_out_2, s_out);  
      

    B2ACircuit(gc, context, s_p, s_out, s_s_y, s_c_y);
}
void buildPool2LayerU(GarbledCircuit& gc, BuildContext& context,
        ui64 width, ui64 n_circ, ui64 p) {
    std::vector<uv64> c_x(4, uv64(width));
    std::vector<uv64> s_x(4, uv64(width));
    uv64 s_y(width);
    uv64 c_y(width);

    int n = n_circ*width*9;
    int m = n_circ*width;

    startBuilding(&gc, &context, n, m, n_circ*2200);
    gc.n_c = 4*n_circ*width;
    uv64 s_p, s_p_2;
    CONSTCircuit(&gc, &context, p, width, s_p);
    CONSTCircuit(&gc, &context, p/2, width, s_p_2);
    for(ui64 i=0; i<n_circ; i++){
        for(ui64 j=0; j<4; j++){
            fill_vector(c_x[j], (j*n_circ+i)*width);
            fill_vector(s_x[j], ((4+j)*n_circ+i)*width);
        }
        fill_vector(s_y, (8*n_circ+i)*width);
        Pool2CircuitU(&gc, &context, s_p, s_p_2, c_x, s_x, s_y, c_y);
        addOutputs(&gc, &context, c_y);
    }
    finishBuilding(&gc, &context);

    return;
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

vector<uv64> get_matrix(ui32 in_h,ui32 in_w, uv64 vec){
    vector<uv64> matrix;
    uv64 templine;
    for(int j=0;j<in_h*in_w;j=j+in_w){
        for(int i=j;i<j+in_w;i++){
            templine.push_back(vec[i]);
        }
        matrix.push_back(templine);
        templine.clear();
    }
    return matrix;
}


uv64 execute_maxpool(ui32 in_h,ui32 in_w,uv64 x_r,uv64 r,uv64 r1,const FVParams& params){
    uv64 r_negative(r.size());
    for(int i=0;i<r.size();i++){
        r_negative[i]=mod(opt::p-r[i],opt::p);
    }
    //首先将x+r和r变成矩阵形式
    vector<uv64> matrix_x_r=get_matrix(in_h,in_w,x_r);
    vector<uv64> matrix_r=get_matrix(in_h,in_w,r_negative);
    ui64 n_circ = (in_h*in_w)/4;
    ui64 in_args = 9, out_args = 1;
    ui64 width = 22;
    ui64 p = params.p;
    std::vector<uv64> din = std::vector<uv64>(n_circ, uv64(in_args));
    std::vector<uv64> dref = std::vector<uv64>(n_circ, uv64(out_args));
    std::vector<uv64> dout_pt = std::vector<uv64>(n_circ, uv64(out_args));
    std::vector<uv64> dout = std::vector<uv64>(n_circ, uv64(out_args));
    

    //---------------------------------对din进行赋值----------------------------
    int i=0;
    for(int j=0;j<in_h/2;j++){
        for(int k=0;k<in_w/2;k++){
            din[i][0]=matrix_x_r[2*j][2*k];
            din[i][1]=matrix_x_r[2*j][2*k+1];
            din[i][2]=matrix_x_r[2*j+1][2*k];
            din[i][3]=matrix_x_r[2*j+1][2*k+1];
            din[i][4]=matrix_r[2*j][2*k];
            din[i][5]=matrix_r[2*j][2*k+1];
            din[i][6]=matrix_r[2*j+1][2*k];
            din[i][7]=matrix_r[2*j+1][2*k+1];
            din[i][8]=r1[i]; 
            i++;      
        }  
        if(i>=n_circ){
            break;
        }   
    }
    
    for(ui32 n=0; n<n_circ; n++){
        ui64 mask = ((1 << width)-1);
        pool2_refU(din[n], dref[n], mask, p);
    }
     
    GarbledCircuit gc;
    BuildContext context;
    buildPool2LayerU(gc, context, width, n_circ, p);
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

    // Print Circuit Info
    // print_gc(gc);

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
    //uv64 ref;
    for(int i=0;i<n_circ;i++){
        out.push_back(dout[i][0]);
    }
    return out;

}

//维度扩展
void expand_dem(vector<uv64> &output,int in_h, int in_w, uv64 &r3,const FVParams& params){
    for(int k=0;k<output.size();k++){
        for(int i=(in_w*in_h)/4;i<params.phim;i++){
            output[k].push_back(0);
        }
    }
    for(int i=(in_w*in_h)/4;i<params.phim;i++){
         r3.push_back(0);
    }
}
uv64 get_nagative(uv64 vec,const FVParams& params){
    uv64 r2_nagtive(vec.size());
    for(ui32 n=0; n<vec.size(); n++){
        r2_nagtive[n] = mod(params.p-vec[n],params.p);
    }  
    return r2_nagtive;   
}
void print_y(uv64 x_r,uv64 r, const FVParams& params){
    uv64 y(2048);
    uv64 r1_nagative=get_nagative(r,params);
    for(int i=0;i<2048;i++){
        y[i]=mod(x_r[i]+r1_nagative[i],params.p);
    }
    cout<<vec_to_str(y)<<endl;
    cout<<endl;
}
void print_result(vector<uv64> output,int in_h,int in_w, uv64 r3,const FVParams& params){
    uv64 r3_negative(2048);
    r3_negative=get_nagative(r3,params);
    vector<uv64> vec_y;
    uv64 y1(2048);
    for(int k=0;k<output.size();k++){
        for(int i=0;i<(in_h*in_w)/4;i++){
            y1[i]=mod(output[k][i]+r3_negative[i],opt::p);
        }
        vec_y.push_back(y1);
    }
    for(int i=0;i<vec_y.size();i++){
        for(int k=0;k<(in_h*in_w)/4;k=k+in_h/2){
            for(int j=k;j<k+in_h/2;j++){
                cout<<vec_y[i][j]<<" ";
            }
            cout<<endl;
        }
        cout<<endl;
    }

}
void print_result_2stage(vector<uv64> output,int in_h,int in_w, uv64 r3,const FVParams& params){
    uv64 r3_negative=get_nagative(r3,params);
    vector<uv64> vec_y;
    uv64 y1(in_h*in_w/4);
    for(int k=0;k<output.size();k++){
        for(int i=0;i<(in_h*in_w)/4;i++){
            y1[i]=mod(output[k][i]+r3_negative[i],opt::p);
        }
        vec_y.push_back(y1);
    }

    for(int i=0;i<vec_y.size();i++){
        for(int k=0;k<(in_h*in_w)/4;k=k+in_h/2){
            for(int j=k;j<k+in_h/2;j++){
                cout<<vec_y[i][j]<<" ";
            }
            cout<<endl;
        }
        cout<<endl;
    }

}
void save_result(vector<uv64> noise,vector<uv64> out,string path_output){
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

vector<uv64> get_noise(vector<uv64> vec,int in_w,int in_h,const FVParams& params){
    vector<uv64> vec_noise;
    ui32 chn_pow2 = nxt_pow2(in_w*in_h);
    int num_noise=params.phim/chn_pow2;
    for(int i=0;i<num_noise;i++){
        vec_noise.push_back(vec[i]);
    }
    return vec_noise;
}

vector<vector<uv64>> get_x_r(vector<uv64> vec,int in_w,int in_h,const FVParams& params){
    ui32 chn_pow2 = nxt_pow2(in_w*in_h);
    int num_noise=params.phim/chn_pow2;
    vector<uv64> temp_x_r;
    vector<vector<uv64>> x_r;
    for(int i=num_noise;i<vec.size();i=i+num_noise){
        for(int j=i;j<i+num_noise;j++){
            temp_x_r.push_back(vec[j]);
        }
        x_r.push_back(temp_x_r);
        temp_x_r.clear();
    }
    return x_r;
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
void maxpool(vector<vector<uv64>> vec,uv64 r3,int in_w,int in_h, string path_output,const FVParams& params){
    vector<uv64> output;
    vector<uv64> noise=get_noise(vec[0],in_w,in_h,params);
    uv64 out;
    vector<vector<uv64>> x_r2=get_x_r(vec[0],in_w,in_h,params);
    
    //--------------------检测-------------------------------
    vector<uv64> matrix;
    uv64 y(params.phim);
    vector<uv64> vec_r_nagative=get_vec_nagative(noise,params);
    for(int i=0;i<x_r2.size();i++){
        for(int j=0;j<x_r2[i].size();j++){
            for(int k=0;k<params.phim;k++){
                y[k]=mod(x_r2[i][j][k]+vec_r_nagative[j][k],opt::p);
            }         
            matrix.push_back(y);
        }
    }


    double start, stop;
    start = currentDateTime();
    for(int i=0;i<x_r2.size();i++){
        for(int j=0;j<x_r2[i].size();j++){
            out=execute_maxpool(in_h,in_w,x_r2[i][j],noise[j],r3,params);
            output.push_back(out);
            out.clear();
        }
    }
    stop= currentDateTime();
    expand_dem(output,in_h,in_w,r3,params);//要对y+r3进行扩充维度到2048
    //将y+r3和r3传入到下层卷积层
    vector<uv64> vec_r;
    vec_r.push_back(r3);
    save_result(vec_r,output,path_output);
}

uv64 eliminate_noise(uv64 vec,uv64 r,const FVParams& params){
    uv64 y(r.size());
    uv64 r_nagative=get_nagative(r,params);
    for(int i=0;i<r.size();i++){
        y[i]=mod(vec[i]+r_nagative[i],opt::p);
    }
    return y;
}
void save_result(string path,vector<vector<uv64>> act){
    
    ofstream cheack(path,ios::out);
    float value=0;
    vector<float> temp;
    for(int i=0;i<act.size();i++){
        for(int j=0;j<act[i].size();j++){
            for(int k=0;k<act[i][j].size();k++){
                 if(act[i][j][k]<opt::p/2){
                    value=act[i][j][k]/100.0;
                }else{
                    ui64 a=act[i][j][k];
                    value=((opt::p-act[i][j][k])/100.0)*(-1);
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

void maxpool_2stage(vector<vector<uv64>> vec,int out_chn,uv64 r3,int in_w,int in_h, string path_output,const FVParams& params){
    vector<uv64> noise_vec;
    vector<uv64> x_r_vec;
    uv64 out;
    vector<uv64> output;
    for(int i=0;i<out_chn;i++){
        noise_vec.push_back(vec[0][i]);
    }
    for(int j=out_chn;j<2*out_chn;j++){
        x_r_vec.push_back(vec[0][j]);
    }
    
    for(int i=0;i<x_r_vec.size();i++){
            out=execute_maxpool(in_h,in_w,x_r_vec[i],noise_vec[i],r3,params);
            output.push_back(out);
            out.clear();
    }
    // //将y+r3和r3传入到下层卷积层
    vector<uv64> vec_r;
    vec_r.push_back(r3);
    save_result(vec_r,output,path_output);


    noise_vec.clear();
    noise_vec.shrink_to_fit();
    x_r_vec.clear();
    x_r_vec.shrink_to_fit();
    output.clear();
    output.shrink_to_fit();
    out.clear();
    out.shrink_to_fit();
    vec_r.clear();
    vec_r.shrink_to_fit();
}
void save_result_maxpool(string path,vector<vector<uv64>> act, const FVParams& params){
    
    ofstream cheack(path,ios::out);
    float value=0;
    vector<float> temp;
    for(int i=0;i<act.size();i++){
        for(int j=0;j<act[i].size();j++){
            for(int k=0;k<act[i][j].size();k++){
                 if(act[i][j][k]<params.p/2){
                    value=act[i][j][k];
                }else{
                    ui64 a=act[i][j][k];
                    value=((params.p-act[i][j][k]))*(-1);
                    //cout<<value<<" ";
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
int main() {
    //cout<<"------------------------执行maxpooling-------------------"<<endl;
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

    string path_input="../result/relu_5_3_out.txt";
    string path_parameter="../params/con5_3_param.txt";
    string path_input_key="../params/outkey.txt";
    string path_output="../result/maxpool_5_out.txt";

    //从卷积层读入输出map的维度
    uv64 v_paramater;
    v_paramater=read_paramater(path_parameter);
    ui32 out_chn = v_paramater[0];
    ui32 in_chn = v_paramater[1]; 
    ui32 in_h = v_paramater[2];
    ui32 in_w = v_paramater[3];
    ui32 f_h = v_paramater[4];
    ui32 f_w = v_paramater[5];
    ui32 conv_type =v_paramater[6];
    //--------------------------------server:产生噪声r3-----------------------------
    for(int i=0;i<(in_h*in_w)/4;i++){
        r3[i]=rand()%100;
    }
    //--------------------------------读入relu的输出结果x+r'以及r'--------------------
    vector<vector<uv64>> vec;
    vec=readInputTensor(path_input);
    ui32 chn_pow2 = nxt_pow2(in_h*in_w);
    bool flag=(chn_pow2*2>test_params.phim)?true:false;
    (flag)?
        maxpool_2stage(vec,out_chn,r3,in_w,in_h,path_output,test_params):
        maxpool(vec,r3,in_w,in_h,path_output,test_params);

    cout<<"maxpool执行成功！"<<endl;
    vec.clear();
    vec.shrink_to_fit();
    r3.clear();
    r3.shrink_to_fit();

    return 0;
}

