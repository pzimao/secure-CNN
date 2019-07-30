import pymysql
from layer import *
from utils import *
import time
class Vgg(nn.Module):
    def __init__(self):
        super().__init__()
        #第一层
        self.conv1_1=conv_1_1(64,3,256,256,3,3,1)
        self.relu1_1=relu_1_1()
        self.conv1_2 = conv_1_2(64,64,256,256,3,3,1)
        self.relu1_2 = relu_1_2()
        self.maxpool1=maxpool_1()
        #第二层
        self.conv2_1=conv_2_1(128,64,128,128,3,3,1)
        self.relu2_1 = relu_2_1()
        self.conv2_2 = conv_2_2(128,128,128,128,3,3,1)
        self.relu2_2 = relu_2_2()
        self.maxpool2 = maxpool_2()
        #第三层
        self.conv3_1=conv_3_1(256,128,64,64,3,3,1)
        self.relu3_1 = relu_3_1()
        self.conv3_2 = conv_3_2(256,256,64,64,3,3,1)
        self.relu3_2 = relu_3_2()
        self.conv3_3 = conv_3_3(256,256,64,64,3,3,1)
        self.relu3_3 = relu_3_3()
        self.maxpool3 = maxpool_3()
        #第四层
        self.conv4_1=conv_4_1(512,256,32,32,3,3,0)
        self.relu4_1 = relu_4_1()
        self.conv4_2 = conv_4_2(512,512,32,32,3,3,0)
        self.relu4_2 = relu_4_2()
        self.conv4_3 = conv_4_3(512,512,32,32,3,3,0)
        self.relu4_3 = relu_4_3()
        self.maxpool4 = maxpool_4()
        # 第五层
        self.conv5_1 = conv_5_1(512,512,16,16,3,3,0)
        self.relu5_1 = relu_5_1()
        self.conv5_2 = conv_5_2(512,512,16,16,3,3,0)
        self.relu5_2 = relu_5_2()
        self.conv5_3 = conv_5_3(512,512,16,16,3,3,0)
        self.relu5_3 = relu_5_3()
        self.maxpool5 = maxpool_5()
        #全链接层
        self.fc1=fc_1(256,32768)
        self.relu_fc_1 = relu_fc_1()
        # self.fc2 = fc_2(10, 256)
        # self.relu_fc_2 = relu_fc_2()
        # self.linear2=Mylinear_2(10,100);
    def __call__(self, input):
        return self.forward(input)

    def forward(self, x):
        self.conv1_1(x)
        self.relu1_1()
        self.conv1_2()
        self.relu1_2()
        self.maxpool1()
        # 第二层
        self.conv2_1()
        self.relu2_1()
        self.conv2_2()
        self.relu2_2()
        self.maxpool2()
        # 第三层
        self.conv3_1()
        self.relu3_1()
        self.conv3_2()
        self.relu3_2()
        self.conv3_3()
        self.relu3_3()
        self.maxpool3()
        # 第四层
        self.conv4_1()
        self.relu4_1()
        self.conv4_2()
        self.relu4_2()
        self.conv4_3()
        self.relu4_3()
        self.maxpool4()
        # 第五层
        self.conv5_1()
        self.relu5_1()
        self.conv5_2()
        self.relu5_2()
        self.conv5_3()
        self.relu5_3()
        self.maxpool5()
        # 全链接层
        self.fc1()
        self.relu_fc_1()
        # self.fc2 ()
        # self.relu_fc_2()
        #return x

def main():
    #---------------------------------------特征提取器--------------------------------------
    path_descriptor="/home/liufei/Documents/20190310/cnn/result/descriptor.txt"
    path_descriptor_relu = "/home/liufei/Documents/20190310/cnn/result/descriptor_relu.txt"
    CLEAR_PATH = r'/home/liufei/Documents/20190310/cnn/result'
    for i in range(40,1000):
        start=time.time()
        img_path = "/home/liufei/Documents/20190310/cnn/Data/corel_data/" + str(i) + ".jpg"
        data=preprocess_image(img_path)
        vgg = Vgg()  # 执行构造函数
        # 调用forward函数
        vgg(data)
        end = time.time()
        print("\n")
        print(str(end-start)+"s")
        #---------------------------------特征向量存入数据库-----------------------------
        f1=open(path_descriptor,'r')
        f2 = open(path_descriptor_relu, 'r')
        #从文件读出特征向量
        descriptor = np.loadtxt(f1)  # 将文件中数据加载到data数组里
        descriptor_relu = np.loadtxt(f2)
        # 去除字符串两边多余的字符
        descriptor = descriptor.tolist()
        descriptor_relu=descriptor_relu.tolist()

        descriptor = [round(i, 4) for i in descriptor]
        descriptor_relu = [round(i, 4) for i in descriptor_relu]
        str1 = str([descriptor]).strip('[').strip(']')
        str2 = str([descriptor_relu]).strip('[').strip(']')
        # print(len(str1))
        # print("\n")
        # 数据库链接
        db = pymysql.connect("127.0.0.1", "root", "287216", db="cbir", charset="utf8")
        # 使用cursor()方法获取操作游标
        cursor = db.cursor()
        # SQL 插入语句
        sql1 = "INSERT INTO cbir.vgg_descriptor(image_id, feature) VALUES ('%s', '%s')" % (img_path, str1)
        sql2 = "INSERT INTO cbir.vgg_descritor_relu(image_id, feature) VALUES ('%s', '%s')" % (img_path, str2)
        try:
            # 执行sql语句
            result1 = cursor.execute(sql1)
            result2 = cursor.execute(sql2)
            # results = cursor.fetchall()
            # for row in results:
            #     print(row[0])
            db.commit()
        except:
            # Rollback in case there is any error
            print("错误")
            db.rollback()
        # 清空result文件夹
        del_file(CLEAR_PATH)

if __name__ == '__main__':
    main()

