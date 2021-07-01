# semi-supervised-for-captcha

This repository is the official implementation of "A Semi-supervised Deep Learning-Based Solver for Breaking Text-Based CAPTCHAs". Under the condition of using a small number of labeled pictures and a large number of unlabeled pictures, the text-based Captchas are recognized.

[中文版ReadMe](https://github.com/2594306528/semi-supervised-for-captcha/blob/main/README_Chinese.md)

</br>
![image](https://github.com/2594306528/semi-supervised-for-captcha/blob/main/images/1.png)
</br>

## Introduction
This project uses the CNN+Seq2Seq model and the semi-supervised learning framework called Mean-Teacher to perform lightweight recognition of text-based Captchas.
</br>
![image](https://github.com/2594306528/semi-supervised-for-captcha/blob/main/images/2.png)
</br>

## Model
The model is composed of a CNN and a Seq2Seq model. Features are extracted by CNN, and the Seq2Seq model translates these features into sequences.
## Semi-supervised learning
[Mean-Teacher](https://github.com/CuriousAI/mean-teacher) is a very effective semi-supervised learning algorithm. This project splits it into two-stage training. The first stage uses only a small number of labeled images for pre-training, and the second stage combines the Mean-Teacher method for semi-supervised training.

## Trained Model
[Baidu Netdisk](https://pan.baidu.com/s/1yNomSJc9tjq76HfCcupOfw) Extraction code: q5e6</br>
[OneDrive](https://sjtueducn-my.sharepoint.com/:u:/g/personal/2594306528_sjtu_edu_cn/EYYSiq8JP2hLlndRv0d68XIBJRjj7m9PtEOeyIC5xcLCTQ?e=9RHAiq)

## Training

### 1、Download Dataset
[Baidu Netdisk](https://pan.baidu.com/s/1re9qP0sBjZ8DGerNdjDGVQ) Extraction code: u5pb</br>
[OneDrive](https://sjtueducn-my.sharepoint.com/:u:/g/personal/2594306528_sjtu_edu_cn/ETsYouBCbxlKk7FPo-9rafwBJQL7gAwZrUXxYTJXlfx0mg?e=TCp5sl)

Unzip the file after downloading.
### 2、Pretrain Stage
Pre-train the model with a small number of labeled pictures.</br>
`python main.py --dataset <dataset-name> --label 700.txt`
### 3、Second Stage
Use semi-supervised learning algorithm to train the pre-trained model.</br>
`python main_mean_teacher.py --dataset <dataset-name> --label 700.txt`

You can also run go.sh directly.
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

## Result

|Websites|Accuracy(Stage 1)|Accuracy(Stage 2)|
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

Taking the Captchas of 360 as an example, the two-stage training process is as follows.
</br>
Stage 1：
![image](https://github.com/2594306528/semi-supervised-for-captcha/blob/main/result/360_700.txt.png)
</br>
Stage 2：
![image](https://github.com/2594306528/semi-supervised-for-captcha/blob/main/result/MT_360_700.txt.png)
</br>

## Contact
If you have any questions, please ask directly, or contact the email 2594306528@sjtu.edu.cn.
