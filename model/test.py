import numpy as np
import pymysql
path_descriptor="/home/liufei/Documents/20190310/cnn/result/descriptor.txt"
path_descriptor_relu = "/home/liufei/Documents/20190310/cnn/result/descriptor_relu.txt"
img_path = "/home/liufei/Documents/20190310/cnn/Data/corel_data/" + str(1) + ".jpg"
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
