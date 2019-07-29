/*
NN-Layers-Benchmarking: This code benchmarks FC and Conv layers for a neural network

List of Authors:
Chiraag Juvekar, chiraag@mit.edu

License Information:
MIT License
Copyright (c) 2017, Massachusetts Institute of Technology (MIT)

*/

#include <pke/gazelle.h>
#include <iostream>
#include <random>
#include "math/bit_twiddle.h"
#include <fstream>
#include <regex>
#include <cmath>
using namespace std;
using namespace lbcrypto;

uv64 postprocess_prodU(const SecretKey& sk, const Ciphertext& ct_prod,
        const ui32 vec_size, const ui32 num_rows, const FVParams& params){
    auto pt = packed_decode(Decrypt(sk, ct_prod, params), params.p, params.logn);
    auto prod = uv64(num_rows);

    ui32 sz_pow2 = nxt_pow2(vec_size);
    ui32 pack_factor = (params.phim / nxt_pow2(vec_size));
    ui32 set_size = nxt_pow2(num_rows)/pack_factor;
    for(ui32 row=0; row<num_rows; row++){
        ui32 curr_set = (row / set_size);
        prod[row] = (pt[(row % set_size) + sz_pow2*curr_set])%params.p;
    }

    return prod;
}
uv64 empty_encrypt(uv64 vec,const FVParams & params){
    auto pt_enc = packed_encode(vec, params.p, params.logn);
    for(ui32 n=0; n<vec.size(); n++){
        pt_enc[n] = pt_enc[n]*params.delta;
    }
    uv64 ct=NullEncrypt(pt_enc, params);
    return ct;
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
uv64 readBias(string path,const FVParams & params){
    uv64 bias;
    ifstream in(path);
    string line;
    regex pat_regex("[-]?([0-9]+.[0-9]+)"); 
    while(getline(in, line)) {
            for (sregex_iterator it(line.begin(), line.end(), pat_regex), end_it; it != end_it; ++it) {
                float num=Round(stof(it->str().c_str()),1)*10;
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
// corrected subplain function
Ciphertext EvalSubPlainU(const Ciphertext& ct, const uv64& pt, const FVParams& params)
{
    uv64 pt_c(pt.size());
    //求补
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

uv64 my_preprocess_vec(uv64 b, const ui32 vec_size, const ui32 num_rows, const FVParams& params){
    uv64 ret = uv64(params.phim);

    ui32 sz_pow2 = nxt_pow2(vec_size);
    ui32 pack_factor = (params.phim / nxt_pow2(vec_size));
    ui32 set_size = nxt_pow2(num_rows)/pack_factor;

    for(ui32 row=0; row<num_rows; row++){
        ui32 curr_set = (row / set_size);
        ret[(row % set_size) + sz_pow2*curr_set] = b[row];
    }

    return ret;

}

// performed by server, recover x from encrypted [x+r] and plaintext r
CTVec preprocess_recover_vec(const SecretKey& sk, CTVec &x_r, const uv64 r,
        const ui32 window_size,const ui32 num_windows, const FVParams& params) 
{
    //先减去r(恢复x)再移位
    uv64 pt(params.phim);
    ui32 sz_pow2 = nxt_pow2(r.size());
    ui32 pack_factor = (params.phim / sz_pow2);
    for(ui32 col=0; col<r.size(); col++){
        for(ui32 n=0; n<pack_factor; n++){
            pt[col + sz_pow2*n] = r[col];
        }
    }
    x_r[0]=EvalSubPlainU(x_r[0], pt, params);
    //密文下需要乘以移位后的值，也就是2^10=1024
    for ( ui32 w = 1; w <num_windows; w++)
    {
        
        for (ui32 n=0; n < pt.size(); n++)
        {   
            pt[n]=(ui32)(1 << (w*window_size));
        }
        uv64 pt_enc = packed_encode(pt, params.p, params.logn);

        uv64 temp=NullEncrypt(pt_enc, params);

        x_r[w]= EvalMultPlain(x_r[0], temp, params);

    }

    return x_r;
}
CTVec preprocess_vecU(const SecretKey& sk, const uv64 &vec,
        const ui32 window_size, const ui32 num_windows, const FVParams& params){
    uv64 pt(params.phim);
    ui32 sz_pow2 = nxt_pow2(vec.size());//
    ui32 pack_factor = (params.phim / sz_pow2);
    for(ui32 col=0; col<vec.size(); col++){
        for(ui32 n=0; n<pack_factor; n++){
            pt[col + sz_pow2*n] = vec[col];
        }
    }
    pt = packed_encode(pt, params.p, params.logn);  

    // Expand the input with multiples of the plaintext base
    std::vector<uv64> pt_scaled(num_windows, uv64(params.phim));
    for (ui32 w=0; w<num_windows; w++){
        for (ui32 i=0; i<params.phim; i++){
            pt_scaled[w][i] = ((pt[i] << (w*window_size)) % params.p);
        }
    }

    CTVec ct_vec(num_windows, Ciphertext(params.phim));
    for (ui32 w=0; w<num_windows; w++){

        ct_vec[w] = Encrypt(sk, pt_scaled[w], params);
    }

    return ct_vec;
}
//从文件里读取多个矩阵
vector<uv64> readInputWeight(string path,const FVParams& params){
    uv64 temp_line;
    vector<uv64> weight ;
    string line;
    ifstream in(path);
    regex pat_regex("[-]?([0-9]+.[0-9]+)");
    while(getline(in, line)) { 
            for (sregex_iterator it(line.begin(), line.end(), pat_regex), end_it; it != end_it; ++it) {
                float num=Round(stof(it->str().c_str()),1)*10; 
                cout<<num;
                cout<<" ";
                if(num<0){
                    num=num+params.p;
                }
                temp_line.push_back(num); 
            }
            cout<<endl;
            if(temp_line.size()>0){
                weight.push_back(temp_line);
                temp_line.clear();
            }else{
                temp_line.clear();
            }           
        }      
    return weight;
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
vector<uv64> get_x_r(vector<uv64> vec){
    vector<uv64> vec_x_r;
    for(int i=1;i<vec.size();i++){
        vec_x_r.push_back(vec[i]);
    }
    return vec_x_r;
}

uv64 get_nagative(uv64 vec,const FVParams& params){
    uv64 r2_nagtive(vec.size());
    for(ui32 n=0; n<vec.size(); n++){
        r2_nagtive[n] = mod(params.p-vec[n],params.p);
    }  
    return r2_nagtive;   
}

uv64 eliminate_noise(uv64 vec,int num_rows,uv64 nagative,const FVParams& params){
    uv64 pt(num_rows);
    for(int j=0;j<num_rows;j++){
        pt[j]=mod(vec[j]+nagative[j],params.p);
    }
    return pt;
}
uv64 joint(vector<uv64> vec,int num_cols,int len){
    uv64 vec_joint(num_cols);
    int i=0;
    for(int k=0;k<vec.size();k++){
        for(int j=0;j<len;j++){
            vec_joint[i]=vec[k][j];
            i=i+1;
        }
    }
    return vec_joint;
}
uv64 joint_r(uv64 r,int in_chn,int in_w, int in_h){
    uv64 vec;
    for(int j=0;j<in_chn;j++){
        for(int i=0;i<in_h*in_w;i++){
            vec.push_back(r[i]);
        }
    }
    return vec; 
}
uv64 mat_mul_ptU(const uv64& vec, uv64 b,const std::vector<uv64> mat, const ui64 p){
    ui32 rows = mat.size();
    ui32 cols = vec.size();
    cout<<"rows:"<<rows<<endl;
    cout<<"cols:"<<cols<<endl;
    uv64 product(rows, 0);
    for (ui32 row = 0; row < rows; row++){
        for (ui32 col = 0; col < cols; col++){
            ui64 partial = mod(vec[col]*mat[row][col], p);
            product[row] = mod(product[row] + partial, p);
        }
    }
    for(ui32 row=0;row<rows;row++){
        product[row]=mod(product[row]+b[row],p);
    }
    return product;
}


void mat_mul_ptU01(const uv64& vec, uv64 b,const std::vector<uv64>& mat, const ui64 p){
    ui32 rows = mat.size();
    ui32 cols = vec.size();
    for (ui32 row = 0; row < rows; row++){
        int64_t sum=0;
        for (ui32 col = 0; col < cols; col++){
            int64_t temp=0;
            if(mat[row][col]>p/2){
                temp=mat[row][col]-p;
            }else{
                temp=mat[row][col];
            }
            int64_t partial =vec[col]*temp;
            sum=sum + partial;
        }
        if(b[row]>p/2){
            sum+=(b[row]-p);
        }else{
            sum+=b[row];
        }
        cout<<sum<<" ";
    }

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
                if(num<0){
                    num=num+opt::p;
                }
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
Ciphertext mat_mul_online01(const CTVec& ct_vec, uv64 b, const EncMat& enc_mat,
        const ui32 num_cols, const FVParams& params){

    auto ct_b = empty_encrypt(b, params);

    cout << "endterng emptey_encrypted" << endl;

    Ciphertext ret(params.phim);
    ui32 padded_rows = enc_mat.size();
    cout<<"padded_rows"<<padded_rows<<endl;
    for(ui32 w=0; w<ct_vec.size(); w++){
        auto digits_vec_w = HoistedDecompose(ct_vec[w], params);
        Ciphertext curr_vec(params.phim);
        Ciphertext cur_vec_b(params.phim);
        for(ui32 row=0; row<padded_rows; row++){
            if(row == 0){
                curr_vec = ct_vec[w];
                
            } else {
                auto rk = GetAutomorphismKey(row);
                curr_vec = EvalAutomorphismDigits(row, *rk, ct_vec[w], digits_vec_w, params);
            }
            auto mult = EvalMultPlain(curr_vec, enc_mat[row][w], params);
            ret = EvalAdd(ret, mult, params);
            
        }
        
    }

    // Rotate and add the partial sums
    ui32 pack_factor = (params.phim / nxt_pow2(num_cols));
    for (ui32 rot = padded_rows; rot < (params.phim/pack_factor); rot *= 2){
        auto rotated_ret = EvalAutomorphism(rot, ret, params);
        ret = EvalAdd(ret, rotated_ret, params);
    }

    ret=EvalAddPlain(ret, ct_b, params);

    return ret;
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

vector<vector<uv64>> get_matrix(vector<uv64> vec,int in_h,int in_w){
    vector<vector<uv64>> matrix;
    vector<uv64> temp_matrix;
    uv64 templine;
    for(int i=0;i<vec.size();i++){
        for(int j=0;j<in_h*in_w;j=j+in_h){
            for(int k=j;k<j+in_h;k++){
                templine.push_back(vec[i][k]);
            }
            temp_matrix.push_back(templine);
            templine.clear();
            
        }
        matrix.push_back(temp_matrix);
        temp_matrix.clear();
    }
    return matrix;

}

uv64 transToVec(vector<vector<uv64>> matrix, int in_h, int in_w){
    uv64 vec;
    for(int j=0;j<in_h;j++){
        for(int i=0;i<in_w;i++){
            for(int k=0;k<matrix.size();k++){
                vec.push_back(matrix[k][j][i]);
            }
        }
    }
    return vec;
}
void print_map(vector<vector<uv64>> map){
    for(int chn=0;chn<map.size();chn++){
        for(int h=0;h<map[chn].size();h++){
            cout<<vec_to_str(map[chn][h])<<endl;      
        }
        cout<<endl;
    }

}
Ciphertext excute_fc(const SecretKey& sk,int num_rows,int num_cols, vector<uv64> &weight,const uv64 &x,const uv64 &x_r,const uv64 &vec_r,FVParams& params){
    std::vector<uv64> mat(num_rows, uv64(num_cols));
    for(ui32 row=0; row<weight.size(); row++){
        mat[row]=weight[row];
    }
    double start, stop;
    start = currentDateTime();
    ui32 num_rot = nxt_pow2(num_rows)*nxt_pow2(num_cols)/opt::phim;
    uv32 index_list;
    for (ui32 i = 1; i < num_rot; i++){
        index_list.push_back(i);
    }
    for(ui32 i=num_rot; i<num_cols; i*=2){
        index_list.push_back(i);
    }
    
    EvalAutomorphismKeyGen(sk, index_list, params);
    //-----------------------客户端对向量进行加密------------------------------------
    //-----------------------Preprocess Vector -----------------------------------
    ui32 mat_window_size = 10;
    ui32 mat_num_windows = 1 + floor(log2(params.p))/mat_window_size;
    auto ct_vec = preprocess_vecU(sk, x_r, mat_window_size, mat_num_windows, params);
    //-----------------------服务端处理噪声r(减去r)--------------------------------------
    auto new_ct_vec = preprocess_recover_vec(sk,ct_vec, vec_r, mat_window_size,mat_num_windows, params);
    //-----------------------服务端对矩阵进行预处理--------------------------------
    //----------------- -----Preprocess Matrix ------------------
    auto enc_mat = preprocess_matrix(mat, mat_window_size, mat_num_windows, params);
    //--------------------- --Multiply -----------------------
    //-----------------------服务器执行明文矩阵和加密向量的乘法--------------------------
    
    auto ct_prod = mat_mul_online(new_ct_vec, enc_mat, num_cols, params);
    stop= currentDateTime();
    //cout << " fc_1 cost_time: " << (stop-start)<<endl;
    return ct_prod;

}

void fc(uv64 x, uv64 ct_x_r,uv64 vec_r,const SecretKey& sk,vector<uv64> weight, uv64 b,int num_rows, int num_cols, string output,FVParams& params){
    //第一个为噪声
    std::vector<uv64> mat(num_rows, uv64(num_cols));
    for(ui32 row=0; row<weight.size(); row++){
        mat[row]=weight[row];
    }
    
    Ciphertext ct_prod=excute_fc(sk,num_rows,num_cols,mat, x,ct_x_r, vec_r,params);
    // preprocess vec bias b
    uv64 bias = my_preprocess_vec(b, num_cols, num_rows, params);
    uv64 ct_bias = empty_encrypt(bias, params);
    ct_prod = EvalAddPlain(ct_prod, ct_bias, params);
    //-----------------------server:产生噪声r,并在密文[x]上加上噪声r,将[x+r]交给客户端----------------------------------------
    uv64 r2=get_dug_vector(params.phim, params.p);
    uv64 ct_r=empty_encrypt(r2,params);
    Ciphertext ct_share = EvalAddPlain(ct_prod, ct_r,params);
    //------------------------client:用密钥sk对[x+r]进行解密获得x+r
    auto  x_r2 = postprocess_prod(sk, ct_share, num_cols, num_rows, params);
    //------------------- ---Post-Process client---------------------
    auto prod = postprocess_prod(sk, ct_prod, num_cols, num_rows, params);
    
    //---------------------- Multiply PT ----------------------
    auto prod_ref = mat_mul_ptU(x, b, mat, opt::p);

    vector<float> feature_descritor_fc2;
    for(int i=0;i<prod_ref.size();i++){
        int64_t temp=0;
        if(prod_ref[i] > opt::p/2){
            temp=prod_ref[i]-opt::p;
        }else{
            temp=prod_ref[i];
        }
        feature_descritor_fc2.push_back((float)temp/10);
    }
    //将结果输出到文件中
    ofstream out1(output,ios::out);
    out1<<vec_to_str(feature_descritor_fc2)<<endl;
    out1.close();
    //mat_mul_ptU01(x,b, mat, opt::p);
    // cout<<"------------------打印明文下的结果：----------------------------------"<<endl;
    cout<<vec_to_str(prod_ref)<<endl;
    check_vec_eq(prod_ref, prod, "mat_mul mismatch:\n");
    cout<<"核对成功！"<<endl;

    weight.clear();
    weight.shrink_to_fit();

}

Ciphertext TwoPartySum(vector<Ciphertext> & CTvec,int num_rows,const SecretKey& sk, FVParams& params){
    //-----------------------server:产生噪声r,并在密文[x]上加上噪声r,将[x+r]交给客户端----------------------------------------
    uv64 r(params.phim);
    for(int i=0;i<params.phim;i++){
        r[i]=rand()%100;
    }
    uv64 ct_r = empty_encrypt(r, params);
    for(int i=0;i<CTvec.size();i++){
        CTvec[i]=EvalAddPlain(CTvec[i], ct_r,params);
    }
    //------------------------client:用密钥sk对[x+r]进行解密获得x+r
    uv64 sumValue;
    for ( int seg = 0; seg<CTvec.size();seg++)
    { 
        auto pt = packed_decode(Decrypt(sk, CTvec[seg], params), params.p, params.logn);
        for ( int i = 0; i < pt.size(); i++)
        {
            if ( seg == 0)
            {
                sumValue.push_back(pt[i]);

            }
            else
            {
                sumValue[i] = mod(sumValue[i]+pt[i],params.p);                
            }
        }
    }
    //------------------------client:用密钥sk对[x+r]进行解密获得x+r

    //客户端对结果加密
    auto ct_share=packed_encode(sumValue, params.p, params.logn);  
    auto new_ct_prod=Encrypt(sk,ct_share, params);

    //------------------------server:将求和的结果发送给服务器,server remove r---------------
   
    for(int i=0;i<r.size();i++){
        r[i]=r[i]*CTvec.size();
    }
    auto ct_result=EvalSubPlainU(new_ct_prod,r,params);
    return ct_result;
}

void fc_2stage(uv64 x, uv64 ct_x_r, uv64 vec_r,const SecretKey& sk,vector<uv64> weight, uv64 b,int num_rows, int num_cols, FVParams& params){

    //---------------------- Multiply PT ----------------------
    auto prod_ref = mat_mul_ptU(x, b, weight, opt::p);

    int num_part=num_cols/params.phim;
    //需要将拼接成的向量划分为num_part
    vector<uv64> split_vec;
    vector<uv64> split_x_r_vec;
    vector<uv64> split_r_vec;
    uv64 temp_vec_x;
    uv64 temp_vec_x_r;
    uv64 temp_vec_r;
    for(int seg=0;seg<num_part;seg++){
        for(int col=seg*params.phim;col<(seg+1)*params.phim;col++){
            temp_vec_x.push_back(x[col]);
            temp_vec_x_r.push_back(ct_x_r[col]);
            temp_vec_r.push_back(vec_r[col]);
        }
        split_vec.push_back(temp_vec_x);
        split_x_r_vec.push_back(temp_vec_x_r);
        split_r_vec.push_back(temp_vec_r);
        temp_vec_x.clear();
        temp_vec_x_r.clear();
        temp_vec_r.clear();
    }

    //将weight分成num_cols/params.phim份
    vector<vector<uv64>> split_matrix;
    uv64 temp_line;
    vector<uv64> temp_matrix;
    for(int seg=0;seg<num_part;seg++){
        for(int h=0;h<weight.size();h++){
            for(int col=seg*params.phim;col<(seg+1)*params.phim;col++){
                temp_line.push_back(weight[h][col]);
            }
            temp_matrix.push_back(temp_line);
            temp_line.clear();
        }
        split_matrix.push_back(temp_matrix);
        temp_matrix.clear();
    }
    //cout<<split_matrix.size()<<endl;
    //分别单独计算每个矩阵
    vector<Ciphertext> CTVec;
    std::vector<uv64> mat(num_rows, uv64(num_cols));
    for(int seg=0;seg<split_matrix.size();seg++){
        Ciphertext ct_prod=excute_fc(sk,num_rows, params.phim, split_matrix[seg], split_vec[seg],split_x_r_vec[seg], split_r_vec[seg], params);
        CTVec.push_back(ct_prod);
    }
    
    // preprocess vec bias b
    Ciphertext ct_new=TwoPartySum(CTVec,num_rows,sk,params);
    uv64 bias = my_preprocess_vec(b, params.phim, num_rows, params);
    //在这里加上
    uv64 ct_bias = empty_encrypt(bias, params);
    auto ct_prod_bias = EvalAddPlain(ct_new, ct_bias, params);
    //------------------- ---Post-Process client---------------------
    auto prod = postprocess_prod(sk, ct_prod_bias, params.phim, num_rows, params);
    cout<<"打印密文的结果"<<endl;
    //cout<<vec_to_str(prod)<<endl;
    
    cout<<"------------------打印明文下的结果：----------------------------------"<<endl;
    //cout<<vec_to_str(prod_ref)<<endl;
    check_vec_eq(prod_ref, prod, "mat_mul mismatch:\n");
    cout<<"核对成功！"<<endl;

    weight.clear();
    weight.shrink_to_fit();

    split_vec.clear();
    split_vec.shrink_to_fit();

    split_x_r_vec.clear();
    split_x_r_vec.shrink_to_fit();

    split_r_vec.clear();
    split_r_vec.shrink_to_fit();

    temp_vec_x.clear();
    temp_vec_x.shrink_to_fit();

    temp_vec_x_r.clear();
    temp_vec_x_r.shrink_to_fit();

    temp_vec_r.clear();
    temp_vec_r.shrink_to_fit();

    split_matrix.clear();
    split_matrix.shrink_to_fit();
}

 //上一层应该是非线性层过来的
int main() {
    //cout<<"--------------------------执行线性层------------------------------------"<<endl;
    //-------------------------------------Setup Parameters -------------------------------------
    ui64 z = opt::z;
    ui64 z_p = opt::z_p;

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

    string input_path="./result/relu_3_out.txt";
    string path_parameter="./params/fc_2_paramater.txt";
    string path_weight="./params/fc_2_weight.txt";
    string path_input_key="./params/outkey.txt";
    string path_input_b="./params/fc_2_b.txt";
    string output="result/feature_descritor_fc2.txt";
    vector<ui64> v_paramater=read_paramater(path_parameter);
    ui32 num_rows = v_paramater[0], num_cols =v_paramater[1];

    cout<<num_cols<<endl;
    v_paramater.clear();
    v_paramater.shrink_to_fit();
    //----------------------------读入密钥-------------------------------
    cout<<"enter1"<<endl;
    vector<vector<uv64>> vec_key=readInputTensor(path_input_key);
    SecretKey sk(2048);
    sk.s=vec_key[0][0]; 

    
    //------------------- Synthetic Data ---------------------------------
    ui32 window_size = 4;
    test_params.window_size = window_size;
    //-------------------从maxpooling中读入输入数据--------------------- 
    //第一个为噪声
    vector<vector<uv64>> input=readInputTensor(input_path);
    for(int i=0;i<input[0].size();i++){
        for(int j=0;j<input[0][i].size();j++){
            input[0][i][j]=input[0][i][j]/10;
        }
    }
    uv64 r=input[0][0];
    uv64 x_r=input[0][1];
    uv64 r_new(num_cols);
    uv64 x_r_new(num_cols);
    for(int i=0;i<num_cols;i++){
        r_new[i]=r[i];
        x_r_new[i]=x_r[i];
    }
    //----------------------------验证----------------------------
    uv64 r_nagaive(num_cols);
    r_nagaive=get_nagative(r_new,test_params);
    uv64 vec_y=eliminate_noise(x_r_new,100,r_nagaive,test_params);
    cout<<vec_to_str(vec_y);

      //--------------------------服务器读入weight数据-------------------------
    vector<uv64> weight;
    cout<<weight.size()<<endl;
    weight=readInputWeight(path_weight,test_params);
    //----------------------------读入b-----------------------------------
    cout<<"enter2"<<endl;
    uv64 b=readBias(path_input_b,test_params);
    cout<<vec_to_str(b)<<endl;
    //----------------------------执行fc-------------------------------------
    bool flag=(num_cols>test_params.phim)?true:false;
    (flag)?
        fc_2stage(vec_y, x_r_new, r_new, sk, weight, b, num_rows, num_cols , test_params):
        fc(vec_y, x_r_new, r_new, sk, weight, b, num_rows, num_cols, output, test_params);


    x_r.clear();
    x_r.shrink_to_fit();


    r.clear();
    r.shrink_to_fit();

    x_r_new.clear();
    x_r_new.shrink_to_fit();

    r_new.clear();
    r_new.shrink_to_fit();
    
    weight.clear();
    weight.shrink_to_fit();

    vec_y.clear();
    vec_y.shrink_to_fit();

    b.clear();
    b.shrink_to_fit();

    r_nagaive.clear();
    r_nagaive.shrink_to_fit();

    return 0;

}

