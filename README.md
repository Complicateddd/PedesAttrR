# 2021青海省首届河湟杯数据湖算法大赛—行人精细化识别赛道

------

写在前面：今年2月左右关注和更进的一个比赛，尝试整合过一些行人属性识别的文章trick，由于近期临近毕业事情比较多，该比赛延期后不打算继续更进了，分享一些尝试过的代码和想法。

## Introduction

1. 该比赛是接近标准的行人属性识别任务，但是有些互斥类和二分类两种形式的标注，二分类（穿着颜色属性）中是软标签。Baseline尝试过两个Repo: 

   - https://github.com/chufengt/iccv19_attribute (想法不错但精度上尚有提升空间) 
   - https://github.com/valencebond/Rethinking_of_PAR （Trick和参数设置得很好，精度较高）

2. 本项目在Rethinking_of_PAR基础上进行了实验和一些补充：

   - **数据增强**

     - Random rotate/Erase
     - Multi-Scale Traning/Test

   - ##### 尝试的**Backbone**：

     - ResNet
     - CoatNet
     - ConvNext
     - Vit
     - swin-transformer

   - **尝试的Neck**

     - ALM (个人的实现方式) [1]
     - FPN (常规FPN)
   
   - **尝试的Head**
   
     - Linear Layer (wi/o BN)
     - CSRA [2]
     - MSSC [3]

## Get Started

参考 https://github.com/valencebond/Rethinking_of_PAR 


### Reference

[1] Improving Pedestrian Attribute Recognition With Weakly-Supervised Multi-Scale Attribute-Specific Localization (ICCV2019)

[2] Residual Attention: A Simple but Effective Method for Multi-Label Recognition (ICCV2021)

[3] Improving Pedestrian Attribute Recognition with Multi-Scale Spatial Calibration (IJCNN)



***欢迎 star和交流***



