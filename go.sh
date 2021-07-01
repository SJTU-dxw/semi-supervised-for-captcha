set -e

python main.py --dataset google
python main.py --dataset ganji-1
python main.py --dataset ganji-2
python main.py --dataset sina
python main.py --dataset weibo
python main.py --dataset apple
python main.py --dataset 360
python main.py --dataset yandex --lr 0.03
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
