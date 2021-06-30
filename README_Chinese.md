# 半监督学习识别验证码

本项目致力于使用少量标注验证码图片，快速突破各种文本型验证码。在使用少量标注图片和大量未标注图片的条件下，对文本型验证码进行识别。
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
### CNN
CNN模型由ResBlk和Maxpooling组成，将64X128的验证码图片提取为32个特征向量，代表验证码图片从左到右的信息
</br>
![image](https://github.com/2594306528/semi-supervised-for-captcha/blob/main/images/3.png)
</br>
### Seq2Seq
Seq2Seq模型增加了注意力机制
</br>
![image](https://github.com/2594306528/semi-supervised-for-captcha/blob/main/images/4.png)
</br>
## 半监督学习
[Mean-Teacher]()是一种非常有效的半监督学习算法，本项目对其进行拆分为两阶段训练。第一阶段仅使用少量标注图片进行预训练，第二阶段再结合Mean-Teacher方法进行半监督训练

## 运行


