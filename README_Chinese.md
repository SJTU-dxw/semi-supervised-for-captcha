# 半监督学习识别验证码

本项目是《A Semi-supervised Deep Learning-Based Solver for Breaking Text-Based CAPTCHAs》的官方实现。在使用少量标注图片和大量未标注图片的条件下，对文本型验证码进行识别。
</br>
![image](https://github.com/2594306528/semi-supervised-for-captcha/blob/main/images/1.png)
</br>

## 介绍
本项目采用CNN+Seq2Seq模型，以及半监督学习框架Mean-Teacher对文本型验证码进行轻量化识别
</br>
![image](https://github.com/2594306528/semi-supervised-for-captcha/blob/main/images/2.png)
</br>

## 模型
模型由CNN+Seq2Seq组成，由CNN提取特征，Seq2Seq将这些特征翻译为验证码序列
## 半监督学习
[Mean-Teacher](https://github.com/CuriousAI/mean-teacher)是一种非常有效的半监督学习算法，本项目对其进行拆分为两阶段训练。第一阶段仅使用少量标注图片进行预训练，第二阶段再结合Mean-Teacher方法进行半监督训练

## 模型

## 训练

### 1、数据集下载

文件下载后解压
### 2、预训练
使用少量标注图片对模型进行预训练</br>
`python main.py --dataset <dataset-name> --label 700.txt`
### 3、第二阶段训练
使用半监督学习算法对预训练的模型进行训练</br>
`python main_mean_teacher.py --dataset <dataset-name> --label 700.txt`

也可以直接运行go.sh
``` shell
set -e

python main.py --dataset google
python main.py --dataset ganji-1
python main.py --dataset ganji-2
python main.py --dataset sina
python main.py --dataset weibo
python main.py --dataset apple
python main.py --dataset 360
python main.py --dataset yandex
python main.py --dataset wikipedia

python main_mean_teacher.py --dataset google
python main_mean_teacher.py --dataset ganji-1
python main_mean_teacher.py --dataset ganji-2
python main_mean_teacher.py --dataset sina
python main_mean_teacher.py --dataset weibo
python main_mean_teacher.py --dataset apple
python main_mean_teacher.py --dataset 360
python main_mean_teacher.py --dataset yandex
python main_mean_teacher.py --dataset wikipedia
```

## 结果

|验证码|阶段一准确率|阶段二准确率|
|-----|------------|-----------|
|google|19.6%|31.2%|
|ganji-1|93.4%|97.4%|
|ganji-2|88.4%|96.0%|
|sina|89.3%|92.1%|
|weibo|95.4%|96.1%|
|yandex|64.7%|78.5%|
|360|50.9%|70.0%|
|wikipedia|93.4%|97.0%|
|apple|73.4%|85.6%|

举例来说，以360的验证码为例，两阶段的训练过程如下
</br>
阶段一：
![image](https://github.com/2594306528/semi-supervised-for-captcha/blob/main/result/360_700.txt.png)
</br>
阶段二：
![image](https://github.com/2594306528/semi-supervised-for-captcha/blob/main/result/MT_360_700.txt.png)
</br>

## 联系
如有任何问题，请直接提问，或者联系邮箱2594306528@sjtu.edu.cn
