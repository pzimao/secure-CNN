import torch.nn as nn
from torch.autograd import Function
import os
from utils import saveImage


class conv_1_1_F(Function):
    def forward(self, input, out_chn, in_chn, in_h, in_w, f_h, f_w, conv_type):
        #print("进程执行之前先把参数写进文件中,然后开启一个进程在c程序中将参数读入到程序中")
        saveImage(input)
        print("执行1_1_conv")
        if(os.system("../excute/1_1_conv.out")==0):
            #print("进程结束之后,从输出的文件中读取结果")
            pass

class conv_1_1(nn.Module):
    def __init__(self, out_chn, in_chn, in_h, in_w, f_h, f_w, conv_type):
        super().__init__()
        self.out_chn = out_chn
        self.in_chn = in_chn
        self.in_h=in_h
        self.in_w=in_w
        self.f_h=f_h
        self.f_w=f_w
        self.conv_type=conv_type

    def __call__(self,input):
        return self.forward(input)

    def forward(self,input):
        return conv_1_1_F.forward(self,input,self.out_chn, self.in_chn, self.in_h, self.in_w, self.f_h, self.f_w, self.conv_type)

#-------------------------------------------------------------------------------------
class relu_1_1_F(Function):
    def forward(self):
        #print("开启一个新的进程,并执行c++程序")
        print("执行1_1_relu")
        if (os.system("../excute/1_1_relu.out") == 0):
            #print("进程结束之后,从输出的文件中读取结果")
            pass

class relu_1_1(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self):
        #print("调用relu_1_1_F Function")
        return relu_1_1_F.forward(self)
#-------------------------------------------------------------------------------------
class conv_1_2_F(Function):
    def forward(self, out_chn,  in_chn,  in_h,  in_w,  f_h, f_w, conv_type):
        #print("开启一个新的进程,并执行c++程序")
        print("执行1_2_conv")
        if(os.system("../excute/1_2_conv.out")==0):
            #print("进程结束之后,从输出的文件中读取结果")
            pass

class conv_1_2(nn.Module):
    def __init__(self, out_chn, in_chn, in_h, in_w, f_h, f_w, conv_type):
        super().__init__()
        self.out_chn = out_chn
        self.in_chn = in_chn
        self.in_h=in_h
        self.in_w=in_w
        self.f_h=f_h
        self.f_w=f_w
        self.conv_type=conv_type

    def __call__(self):
        return self.forward()

    def forward(self):
        #print("调用conv2d Function")
        return conv_1_2_F.forward(self,self.out_chn, self.in_chn, self.in_h, self.in_w, self.f_h, self.f_w, self.conv_type)
#------------------------------------------------------------------------------
class relu_1_2_F(Function):
    def forward(self):
        #print("开启一个新的进程,并执行c++程序")
        print("执行1_2_relu")
        if (os.system("../excute/1_2_relu.out") == 0):
            #print("进程结束之后,从输出的文件中读取结果")
            pass

class relu_1_2(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self):
        #print("调用MyReluF Function")
        return relu_1_2_F.forward(self)
#------------------------------------------------------------------------------
class maxpool_1_F(Function):
    def forward(self):
        #print("开启一个新的进程,并执行c++程序")
        print("执行maxpool_1")
        if (os.system("../excute/maxpool_1.out") == 0):
            #print("进程结束之后,从输出的文件中读取结果")
            pass

class maxpool_1(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self):
        #print("调用MyMaxpoolF Function")
        return maxpool_1_F.forward(self)
#-------------------------------------------------------------------------------------

class conv_2_1_F(Function):
    def forward(self, out_chn,  in_chn,  in_h,  in_w,  f_h, f_w, conv_type):
        #print("开启一个新的进程,并执行c++程序")
        print("执行2_1_conv")
        if(os.system("../excute/2_1_conv.out")==0):
            #print("进程结束之后,从输出的文件中读取结果")
            pass

class conv_2_1(nn.Module):
    def __init__(self, out_chn, in_chn, in_h, in_w, f_h, f_w, conv_type):
        super().__init__()
        self.out_chn = out_chn
        self.in_chn = in_chn
        self.in_h=in_h
        self.in_w=in_w
        self.f_h=f_h
        self.f_w=f_w
        self.conv_type=conv_type

    def __call__(self):
        return self.forward()

    def forward(self):
        #print("调用conv2d Function")
        return conv_2_1_F.forward(self,self.out_chn, self.in_chn, self.in_h, self.in_w, self.f_h, self.f_w, self.conv_type)
#----------------------------------------------------------------------------------------------------------
class relu_2_1_F(Function):
    def forward(self):
        #print("开启一个新的进程,并执行c++程序")
        print("执行2_1_relu")
        if (os.system("../excute/2_1_relu.out") == 0):
            #print("进程结束之后,从输出的文件中读取结果")
            pass

class relu_2_1(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self):
        #print("调用MyReluF Function")
        return relu_2_1_F.forward(self)
#-------------------------------------------------------------------------------------------------------------
class conv_2_2_F(Function):
    def forward(self, out_chn,  in_chn,  in_h,  in_w,  f_h, f_w, conv_type):
        #print("开启一个新的进程,并执行c++程序")
        print("执行2_2_conv")
        if(os.system("../excute/2_2_conv.out")==0):
            #print("进程结束之后,从输出的文件中读取结果")
            pass

class conv_2_2(nn.Module):
    def __init__(self, out_chn, in_chn, in_h, in_w, f_h, f_w, conv_type):
        super().__init__()
        self.out_chn = out_chn
        self.in_chn = in_chn
        self.in_h=in_h
        self.in_w=in_w
        self.f_h=f_h
        self.f_w=f_w
        self.conv_type=conv_type

    def __call__(self):
        return self.forward()

    def forward(self):
        #print("调用conv2d Function")
        return conv_2_2_F.forward(self,self.out_chn, self.in_chn, self.in_h, self.in_w, self.f_h, self.f_w, self.conv_type)
#-----------------------------------------------------------------------------------------------------------------------
class relu_2_2_F(Function):
    def forward(self):
        #print("开启一个新的进程,并执行c++程序")
        print("执行2_2_relu")
        if (os.system("../excute/2_2_relu.out") == 0):
            #print("进程结束之后,从输出的文件中读取结果")
            pass

class relu_2_2(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self):
        #print("调用MyReluF Function")
        return relu_2_2_F.forward(self)
#--------------------------------------------------------------------------------------------------------------------------
class maxpool_2_F(Function):
    def forward(self):
        #print("开启一个新的进程,并执行c++程序")
        print("执行maxpool_2")
        if (os.system("../excute/maxpool_2.out") == 0):
            #print("进程结束之后,从输出的文件中读取结果")
            pass

class maxpool_2(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self):
        #print("调用MyMaxpoolF Function")
        return maxpool_2_F.forward(self)
#-------------------------------------------------------------------------------------
class conv_3_1_F(Function):
    def forward(self, out_chn,  in_chn,  in_h,  in_w,  f_h, f_w, conv_type):
        #print("开启一个新的进程,并执行c++程序")
        print("执行3_1_conv")
        if(os.system("../excute/3_1_conv.out")==0):
            #print("进程结束之后,从输出的文件中读取结果")
            pass

class conv_3_1(nn.Module):
    def __init__(self, out_chn, in_chn, in_h, in_w, f_h, f_w, conv_type):
        super().__init__()
        self.out_chn = out_chn
        self.in_chn = in_chn
        self.in_h=in_h
        self.in_w=in_w
        self.f_h=f_h
        self.f_w=f_w
        self.conv_type=conv_type

    def __call__(self):
        return self.forward()

    def forward(self):
        #print("调用conv2d Function")
        return conv_3_1_F.forward(self,self.out_chn, self.in_chn, self.in_h, self.in_w, self.f_h, self.f_w, self.conv_type)
#-------------------------------------------------------------------------------------------------------------------------
class relu_3_1_F(Function):
    def forward(self):
        #print("开启一个新的进程,并执行c++程序")
        print("执行3_1_relu")
        if (os.system("../excute/3_1_relu.out") == 0):
            #print("进程结束之后,从输出的文件中读取结果")
            pass

class relu_3_1(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self):
        #print("调用MyReluF Function")
        return relu_3_1_F.forward(self)
#--------------------------------------------------------------------------------------------------------------------------
class conv_3_2_F(Function):
    def forward(self, out_chn,  in_chn,  in_h,  in_w,  f_h, f_w, conv_type):
        #print("开启一个新的进程,并执行c++程序")
        print("执行3_2_conv.out")
        if(os.system("../excute/3_2_conv.out")==0):
            #print("进程结束之后,从输出的文件中读取结果")
            pass

class conv_3_2(nn.Module):
    def __init__(self, out_chn, in_chn, in_h, in_w, f_h, f_w, conv_type):
        super().__init__()
        self.out_chn = out_chn
        self.in_chn = in_chn
        self.in_h=in_h
        self.in_w=in_w
        self.f_h=f_h
        self.f_w=f_w
        self.conv_type=conv_type

    def __call__(self):
        return self.forward()

    def forward(self):
        #print("调用conv2d Function")
        return conv_3_2_F.forward(self,self.out_chn, self.in_chn, self.in_h, self.in_w, self.f_h, self.f_w, self.conv_type)
#----------------------------------------------------------------------------------------------------------------------------
class relu_3_2_F(Function):
    def forward(self):
        #print("开启一个新的进程,并执行c++程序")
        print("执行3_2_relu")
        if (os.system("../excute/3_2_relu.out") == 0):
            #print("进程结束之后,从输出的文件中读取结果")
            pass

class relu_3_2(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self):
        #print("调用MyReluF Function")
        return relu_3_2_F.forward(self)
#----------------------------------------------------------------------------------------------------------------------------
class conv_3_3_F(Function):
    def forward(self, out_chn,  in_chn,  in_h,  in_w,  f_h, f_w, conv_type):
        #print("开启一个新的进程,并执行c++程序")
        print("执行3_3_conv")
        if(os.system("../excute/3_3_conv.out")==0):
            #print("进程结束之后,从输出的文件中读取结果")
            pass

class conv_3_3(nn.Module):
    def __init__(self, out_chn, in_chn, in_h, in_w, f_h, f_w, conv_type):
        super().__init__()
        self.out_chn = out_chn
        self.in_chn = in_chn
        self.in_h=in_h
        self.in_w=in_w
        self.f_h=f_h
        self.f_w=f_w
        self.conv_type=conv_type

    def __call__(self):
        return self.forward()

    def forward(self):
        #print("调用conv2d Function")
        return conv_3_3_F.forward(self,self.out_chn, self.in_chn, self.in_h, self.in_w, self.f_h, self.f_w, self.conv_type)
#----------------------------------------------------------------------------------------------------------------------------
class relu_3_3_F(Function):
    def forward(self):
        #print("开启一个新的进程,并执行c++程序")
        print("执行3_3_relu")
        if (os.system("../excute/3_3_relu.out") == 0):
            #print("进程结束之后,从输出的文件中读取结果")
            pass

class relu_3_3(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self):
        #print("调用MyReluF Function")
        return relu_3_3_F.forward(self)
#----------------------------------------------------------------------------------------------------------------------------
class maxpool_3_F(Function):
    def forward(self):
        #print("进程执行之前先把参数写进文件中,然后开启一个进程在c程序中将参数读入到程序中")
        #print("开启一个新的进程,并执行c++程序")
        print("执行maxpool_3")
        if (os.system("../excute/maxpool_3.out") == 0):
            #print("进程结束之后,从输出的文件中读取结果")
            pass

class maxpool_3(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self):
        #print("调用MyMaxpoolF Function")
        return maxpool_3_F.forward(self)
#-------------------------------------------------------------------------------------
class conv_4_1_F(Function):
    def forward(self, out_chn,  in_chn,  in_h,  in_w,  f_h, f_w, conv_type):
        #print("开启一个新的进程,并执行c++程序")
        print("执行4_1_conv")
        if(os.system("../excute/4_1_conv.out")==0):
            #print("进程结束之后,从输出的文件中读取结果")
            pass

class conv_4_1(nn.Module):
    def __init__(self, out_chn, in_chn, in_h, in_w, f_h, f_w, conv_type):
        super().__init__()
        self.out_chn = out_chn
        self.in_chn = in_chn
        self.in_h=in_h
        self.in_w=in_w
        self.f_h=f_h
        self.f_w=f_w
        self.conv_type=conv_type

    def __call__(self):
        return self.forward()

    def forward(self):
        #print("调用conv2d Function")
        return conv_4_1_F.forward(self,self.out_chn, self.in_chn, self.in_h, self.in_w, self.f_h, self.f_w, self.conv_type)
#----------------------------------------------------------------------------------------------------------------------------
class relu_4_1_F(Function):
    def forward(self):
        #print("开启一个新的进程,并执行c++程序")
        print("执行4_1_relu")
        if (os.system("../excute/4_1_relu.out") == 0):
            #print("进程结束之后,从输出的文件中读取结果")
            pass

class relu_4_1(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self):
        #print("调用MyReluF Function")
        return relu_4_1_F.forward(self)
#----------------------------------------------------------------------------------------------------------------------------
class conv_4_2_F(Function):
    def forward(self, out_chn,  in_chn,  in_h,  in_w,  f_h, f_w, conv_type):
        #print("开启一个新的进程,并执行c++程序")
        print("执行4_2_conv")
        if(os.system("../excute/4_2_conv.out")==0):
            #print("进程结束之后,从输出的文件中读取结果")
            pass

class conv_4_2(nn.Module):
    def __init__(self, out_chn, in_chn, in_h, in_w, f_h, f_w, conv_type):
        super().__init__()
        self.out_chn = out_chn
        self.in_chn = in_chn
        self.in_h=in_h
        self.in_w=in_w
        self.f_h=f_h
        self.f_w=f_w
        self.conv_type=conv_type

    def __call__(self):
        return self.forward()

    def forward(self):
        #print("调用conv2d Function")
        return conv_4_2_F.forward(self,self.out_chn, self.in_chn, self.in_h, self.in_w, self.f_h, self.f_w, self.conv_type)
#----------------------------------------------------------------------------------------------------------------------------
class relu_4_2_F(Function):
    def forward(self):
        #print("开启一个新的进程,并执行c++程序")
        print("执行4_2_relu")
        if (os.system("../excute/4_2_relu.out") == 0):
            #print("进程结束之后,从输出的文件中读取结果")
            pass

class relu_4_2(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self):
        #print("调用MyReluF Function")
        return relu_4_2_F.forward(self)
        # return Conv2dF.forward(self)
#----------------------------------------------------------------------------------------------------------------------------
class conv_4_3_F(Function):
    def forward(self, out_chn,  in_chn,  in_h,  in_w,  f_h, f_w, conv_type):
        #print("开启一个新的进程,并执行c++程序")
        print("执行4_3_conv")
        if(os.system("../excute/4_3_conv.out")==0):
            #print("进程结束之后,从输出的文件中读取结果")
            pass

class conv_4_3(nn.Module):
    def __init__(self, out_chn, in_chn, in_h, in_w, f_h, f_w, conv_type):
        super().__init__()
        self.out_chn = out_chn
        self.in_chn = in_chn
        self.in_h=in_h
        self.in_w=in_w
        self.f_h=f_h
        self.f_w=f_w
        self.conv_type=conv_type

    def __call__(self):
        return self.forward()

    def forward(self):
        #print("调用conv2d Function")
        return conv_4_3_F.forward(self,self.out_chn, self.in_chn, self.in_h, self.in_w, self.f_h, self.f_w, self.conv_type)
#----------------------------------------------------------------------------------------------------------------------------
class relu_4_3_F(Function):
    def forward(self):
        #print("开启一个新的进程,并执行c++程序")
        print("执行4_3_relu")
        if (os.system("../excute/4_3_relu.out") == 0):
            #print("进程结束之后,从输出的文件中读取结果")
            pass

class relu_4_3(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self):
        #print("调用MyReluF Function")
        return relu_4_3_F.forward(self)
#----------------------------------------------------------------------------------------------------------------------------
class maxpool_4_F(Function):
    def forward(self):
        #print("开启一个新的进程,并执行c++程序")
        print("执行maxpool_4")
        if (os.system("../excute/maxpool_4.out") == 0):
            #print("进程结束之后,从输出的文件中读取结果")
            pass

class maxpool_4(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self):
        #print("调用MyMaxpoolF Function")
        return maxpool_4_F.forward(self)
        # return Conv2dF.forward(self)
#-------------------------------------------------------------------------------------
class conv_5_1_F(Function):
    def forward(self, out_chn,  in_chn,  in_h,  in_w,  f_h, f_w, conv_type):
        #print("开启一个新的进程,并执行c++程序")
        print("执行5_1_conv")
        if(os.system("../excute/5_1_conv.out")==0):
            #print("进程结束之后,从输出的文件中读取结果")
            pass

class conv_5_1(nn.Module):
    def __init__(self, out_chn, in_chn, in_h, in_w, f_h, f_w, conv_type):
        super().__init__()
        self.out_chn = out_chn
        self.in_chn = in_chn
        self.in_h=in_h
        self.in_w=in_w
        self.f_h=f_h
        self.f_w=f_w
        self.conv_type=conv_type

    def __call__(self):
        return self.forward()

    def forward(self):
        #print("调用conv2d Function")
        return conv_5_1_F.forward(self,self.out_chn, self.in_chn, self.in_h, self.in_w, self.f_h, self.f_w, self.conv_type)
#----------------------------------------------------------------------------------------------------------------------------
class relu_5_1_F(Function):
    def forward(self):
        #print("开启一个新的进程,并执行c++程序")
        print("执行5_1_relu")
        if (os.system("../excute/5_1_relu.out") == 0):
            #print("进程结束之后,从输出的文件中读取结果")
            pass

class relu_5_1(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self):
        #print("调用MyReluF Function")
        return relu_5_1_F.forward(self)
        # return Conv2dF.forward(self)
#----------------------------------------------------------------------------------------------------------------------------
class conv_5_2_F(Function):
    def forward(self, out_chn,  in_chn,  in_h,  in_w,  f_h, f_w, conv_type):
        #print("开启一个新的进程,并执行c++程序")
        print("执行5_2_conv")
        if(os.system("../excute/5_2_conv.out")==0):
            #print("进程结束之后,从输出的文件中读取结果")
            pass

class conv_5_2(nn.Module):
    def __init__(self, out_chn, in_chn, in_h, in_w, f_h, f_w, conv_type):
        super().__init__()
        self.out_chn = out_chn
        self.in_chn = in_chn
        self.in_h=in_h
        self.in_w=in_w
        self.f_h=f_h
        self.f_w=f_w
        self.conv_type=conv_type

    def __call__(self):
        return self.forward()

    def forward(self):
        #print("调用conv2d Function")
        return conv_5_2_F.forward(self,self.out_chn, self.in_chn, self.in_h, self.in_w, self.f_h, self.f_w, self.conv_type)
#----------------------------------------------------------------------------------------------------------------------------
class relu_5_2_F(Function):
    def forward(self):
        #print("开启一个新的进程,并执行c++程序")
        print("执行5_2_relu")
        if (os.system("../excute/5_2_relu.out") == 0):
            #print("进程结束之后,从输出的文件中读取结果")
            pass

class relu_5_2(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self):
        #print("调用MyReluF Function")
        return relu_5_2_F.forward(self)
#----------------------------------------------------------------------------------------------------------------------------
class conv_5_3_F(Function):
    def forward(self, out_chn,  in_chn,  in_h,  in_w,  f_h, f_w, conv_type):
        #print("开启一个新的进程,并执行c++程序")
        print("执行5_3_conv")
        if(os.system("../excute/5_3_conv.out")==0):
            #print("进程结束之后,从输出的文件中读取结果")
            pass

class conv_5_3(nn.Module):
    def __init__(self, out_chn, in_chn, in_h, in_w, f_h, f_w, conv_type):
        super().__init__()
        self.out_chn = out_chn
        self.in_chn = in_chn
        self.in_h=in_h
        self.in_w=in_w
        self.f_h=f_h
        self.f_w=f_w
        self.conv_type=conv_type

    def __call__(self):
        return self.forward()

    def forward(self):
        #print("调用conv2d Function")
        return conv_5_3_F.forward(self,self.out_chn, self.in_chn, self.in_h, self.in_w, self.f_h, self.f_w, self.conv_type)
#----------------------------------------------------------------------------------------------------------------------------
class relu_5_3_F(Function):
    def forward(self):
        #print("开启一个新的进程,并执行c++程序")
        print("执行5_3_relu")
        if (os.system("../excute/5_3_relu.out") == 0):
            #print("进程结束之后,从输出的文件中读取结果")
            pass

class relu_5_3(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self):
        #print("调用MyReluF Function")
        return relu_5_3_F.forward(self)
#----------------------------------------------------------------------------------------------------------------------------
class maxpool_5_F(Function):
    def forward(self):
        #print("开启一个新的进程,并执行c++程序")
        print("执行maxpool_5")
        if (os.system("../excute/maxpool_5.out") == 0):
            #print("进程结束之后,从输出的文件中读取结果")
            pass

class maxpool_5(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self):
        #print("调用MyMaxpoolF Function")
        return maxpool_5_F.forward(self)
#-------------------------------------------------------------------------------------
class fc_1_F(Function):
    def forward(self,input_feature,output_feature):
        print("执行fc_1")
        if (os.system("../excute/fc_1.out") == 0):
            #print("进程结束之后,从输出的文件中读取结果")
            pass

class fc_1(nn.Module):
    def __init__(self,input_feature,output_feature):
        super().__init__()
        self.input_feature=input_feature
        self.output_feature=output_feature
    def forward(self):
        #print("调用MyReluF Function")
        return fc_1_F.forward(self, self.input_feature, self.output_feature)
#----------------------------------------------------------------------------------
class relu_fc_1_F(Function):
    def forward(self):
        #print("开启一个新的进程,并执行c++程序")
        print("执行relu_fc1")
        if (os.system("../excute/relu_fc1.out") == 0):
            #print("进程结束之后,从输出的文件中读取结果")
            pass

class relu_fc_1(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self):
        #print("调用MyReluF Function")
        return relu_fc_1_F.forward(self)

# class fc_2_F(Function):
#     def forward(self,input_feature,output_feature):
#         #print("进程执行之前先把参数写进文件中,然后开启一个进程在c程序中将参数读入到程序中")
#         #print("开启一个新的进程,并执行c++程序")
#         # f = open('/home/liufei/Documents/20190310/test/linear_1_paramater.txt', 'w')
#         # f.writelines([str(input_feature), '\n', str(output_feature)])
#         # f.close()
#         if (os.system("./fc_1.out") == 0):
#             #print("进程结束之后,从输出的文件中读取结果")
#             pass
#
# class fc_2(nn.Module):
#     def __init__(self,input_feature,output_feature):
#         super().__init__()
#         self.input_feature=input_feature
#         self.output_feature=output_feature
#     def forward(self):
#         #print("调用MyReluF Function")
#         return fc_1_F.forward(self, self.input_feature, self.output_feature)
#         # return Conv2dF.forward(self)
#----------------------------------------------------------------------------------