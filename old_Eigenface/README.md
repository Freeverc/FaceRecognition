# EigenfaceInPython
A simple face recognation bases on PCA In Python
## 运行环境依赖：
python 3
numpy
PIL
matplotlib

## 各个脚本文件（均直接无参数运行）：
face_train.py ：训练脚本，生成训练集，并计算模型。输入能量百分比，将模型保存在model文件夹中。
理论上先运行才能运行另外两个，但是运行时间较长，建议先运行另外连个脚本。
训练的模型已经保存下来了（./model），另两个脚本可以直接运行
face_testing.py ：测试脚本，输出测试集的准确率，然后输入一张图片，输出分类结果
face_refactoring.py 输入一张图片，输出重构结果（建议输入demo.jpg）；然后输入一张自己的图片，给出重构结果

## 根目录下各个文件夹：
data文件夹为数据集，包含所有数据，40个人脸类别，每个类别有10张图片，其中1-5张用于模型训练，6-10张用于人脸分类测试
model文件夹保存了训练得到的模型，eigValues_real.txt为特征值，eigVectors_real.txt为特征向量，
face_mean.txt 为平均脸的向量数据，model_data.txt为各个人脸类别的PCA提取结果
results文件夹保存了本次实验的所有可视化结果，包括平均脸、特征脸、重构脸、预测精确率曲线图等
根目录下的demo.pgm来自训练集用于人脸重构，其余的demo1、demo2可用于分类，demo1、demo2、demo3用于重构
