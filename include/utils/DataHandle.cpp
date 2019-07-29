#include <iostream>
#include <fstream>
#include <regex>
#include <string>
#include <vector>
#include "DataHandle.h"
using namespace std;

int readTxt(string file) {
    vector<int> temp_line;
    vector<vector<int>> Vec_Dti;
    string line;
    ifstream in(file);  //读入文件
    regex pat_regex("[[:digit:]]+");  //匹配原则，这里代表一个或多个数字
    //cout<<"hello"<<endl;
    while(getline(in, line)) {  //按行读取
        for (sregex_iterator it(line.begin(), line.end(), pat_regex), end_it; it != end_it; ++it) {  //表达式匹配，匹配一行中所有满足条件的字符
            cout << it->str() << " ";  //输出匹配成功的数据
            temp_line.push_back(stoi(it->str()));  //将数据转化为int型并存入一维vector中
        }
        cout << endl;
        Vec_Dti.push_back(temp_line);  //保存所有数据
        temp_line.clear();
    }
    cout << endl << endl;

    for(auto i : Vec_Dti) {  //输出存入vector后的数据
        for(auto j : i) {
            cout << j << " ";
        }
        cout << endl;
    }
    return 0;
}
