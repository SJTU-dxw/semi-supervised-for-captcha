import numpy as np
import string
import glob
from PIL import Image
from torch.utils.data import TensorDataset, DataLoader, Dataset
import torch
import torchvision.transforms as transforms
import platform
import random

transform_image = transforms.Compose([
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)
])

if platform.system() == "Windows":
    splitter_token = "\\"
    workers = 0
elif platform.system() == "Linux":
    splitter_token = "/"
    workers = 4
else:
    raise Exception("System must be Linux or Windows")


def load_datasets(args):
    label_dict = get_label_dict(args)

    train_filenames = glob.glob("../final/dataset/" + args.dataset + "/train/*.*")
    train_filenames = [train_filename for train_filename in train_filenames if
                       train_filename.split(splitter_token)[-1] in label_dict]
    test_filenames = glob.glob("../final/dataset/" + args.dataset + "/test/*.*")

    dataloader_train, id2token = get_dataloader(train_filenames, label_dict, args, train=True, label=True)
    dataloader_test, _ = get_dataloader(test_filenames, label_dict, args, train=False, label=True)
    return dataloader_train, dataloader_test, id2token


def load_datasets_mean_teacher(args):
    label_dict = get_label_dict(args)

    train_filenames = glob.glob("../final/dataset/" + args.dataset + "/train/*.*") + glob.glob(
        "../final/dataset/" + args.dataset + "/buchong/*.*")
    labeled_train_filenames = [train_filename for train_filename in train_filenames if
                               train_filename.split(splitter_token)[-1] in label_dict]
    nolabeled_train_filenames = [train_filename for train_filename in train_filenames if
                                 train_filename.split(splitter_token)[-1] not in label_dict]
    nolabeled_train_filenames = random.sample(nolabeled_train_filenames, args.unlabeled_number)
    test_filenames = glob.glob("../final/dataset/" + args.dataset + "/test/*.*")

    dataloader_train_labeled, id2token = get_dataloader(labeled_train_filenames, label_dict, args, train=True,
                                                        label=True)
    dataloader_train_nolabeled, _ = get_dataloader(nolabeled_train_filenames, label_dict, args, train=True, label=False)
    dataloader_test, _ = get_dataloader(test_filenames, label_dict, args, train=False, label=True)

    MAXLEN = max([len(value) for value in label_dict.values()])
    MINLEN = min([len(value) for value in label_dict.values()])
    return dataloader_train_labeled, dataloader_train_nolabeled, dataloader_test, id2token, MAXLEN + 2, MINLEN + 2


def get_label_dict(args):
    label_path = '../final/dataset/' + args.dataset + '/label/' + args.label
    f = open(label_path, 'r')
    lines = f.read().strip().split("\n")
    label_dict = {line.split(" ")[0]: line.split(" ")[1] for line in lines}
    f.close()
    return label_dict


def get_vocab(label_dict):
    all_labels = "".join([value for value in label_dict.values()])
    if all_labels.isdigit():
        return string.digits
    elif all_labels.isalpha():
        return string.ascii_lowercase
    elif all_labels.isalnum():
        return string.digits + string.ascii_lowercase
    elif all_labels.replace("-", "").isalnum():
        return string.digits + string.ascii_lowercase + '-'
    else:
        raise Exception("Label files must consist only of numbers and English letters")


class ImageData(Dataset):
    def __init__(self, image, label, transform):
        self.transform = transform
        self.image = image
        self.label = label

    def __len__(self):
        return len(self.image)

    def __getitem__(self, index):
        img = self.image[index]
        img = self.transform(img)
        lb = self.label[index]

        sample = (img, lb)
        return sample


def get_dataloader(filenames, label_dict, args, train, label):
    MAXLEN = max([len(value) for value in label_dict.values()])
    TARGET_HEIGHT = 64
    TARGET_WIDTH = 128

    vocab = get_vocab(label_dict)
    vocab += ' '

    id2token = {k + 1: v for k, v in enumerate(vocab)}
    id2token[0] = '^'
    id2token[len(vocab) + 1] = '$'
    token2id = {v: k for k, v in id2token.items()}

    img_buffer = np.zeros((len(filenames), TARGET_HEIGHT, TARGET_WIDTH, 3), dtype=np.uint8)
    text_buffer = []

    for i, filename in enumerate(filenames):
        captcha_image = Image.open(filename).resize((TARGET_WIDTH, TARGET_HEIGHT), Image.ANTIALIAS)
        if captcha_image.mode != 'RGB':
            captcha_image = captcha_image.convert("RGB")
        captcha_array = np.array(captcha_image)

        img_buffer[i] = captcha_array

        if train:
            if label:
                text = label_dict[filename.split(splitter_token)[-1]]
        else:
            text = filename.split(splitter_token)[-1].split(".")[0]

        if label:
            text = ("^" + text + "$")
            text_buffer.append([token2id[i] for i in text.ljust(MAXLEN + 2)])
        else:
            text_buffer.append([-1] * (MAXLEN + 2))

    text_buffer = np.array(text_buffer)

    image = img_buffer.astype(np.float32) / 127.5 - 1
    image = torch.Tensor(image).permute(0, 3, 1, 2)
    text = torch.LongTensor(text_buffer)

    if label:
        batch_size = args.batch_size
    else:
        batch_size = args.secondary_batch_size

    if train:
        dataset = ImageData(image, text, transform_image)
        dataloader = DataLoader(dataset, batch_size, shuffle=True, drop_last=True, num_workers=workers)
    else:
        dataset = TensorDataset(image, text)
        dataloader = DataLoader(dataset, batch_size, shuffle=False, drop_last=True, num_workers=workers)
    return dataloader, id2token
