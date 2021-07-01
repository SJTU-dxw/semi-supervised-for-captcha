import argparse
import torch
from torch import optim
import matplotlib
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import random

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from datasets import load_datasets
from models import CNNSeq2Seq
from util import compute_seq_acc, Seq2SeqLoss

parser = argparse.ArgumentParser(description='PyTorch Captcha Training')

parser.add_argument('--dataset', default='google', type=str, help="the name of dataset")
parser.add_argument('--label', default="700.txt", type=str, help='the labels of captcha images used for training')
parser.add_argument('--batch-size', default=64, type=int, help='batch size for training and test')
parser.add_argument('--epoch', default=500, type=int, help='the number of training epochs')
parser.add_argument('--lr', default=0.05, type=float, help='learning rate')

args = parser.parse_args()

USE_CUDA = torch.cuda.is_available()

LR = args.lr
NUM_EPOCHS = args.epoch

dataloader_train, dataloader_test, id2token = load_datasets(args)
print("train number is", len(dataloader_train) * args.batch_size)
print("test number is", len(dataloader_test) * args.batch_size)

print("token:", "".join(list(id2token.values())))


manualSeed = random.randint(1, 10000)
random.seed(manualSeed)
torch.manual_seed(manualSeed)
if USE_CUDA:
    torch.cuda.manual_seed_all(manualSeed)

cudnn.benchmark = True


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


model = CNNSeq2Seq(vocab_size=len(id2token))
model.apply(weights_init)
criterion = Seq2SeqLoss()

if USE_CUDA:
    model = model.cuda()
    criterion = criterion.cuda()

params = list(filter(lambda p: p.requires_grad, model.parameters()))
optimizer = optim.SGD(params, lr=LR, momentum=0.9, weight_decay=1e-4, nesterov=True)
#optimizer = optim.Rprop(params, lr=LR)

train_loss = []
train_accclevel = []
train_accuracy = []

test_loss = []
test_accclevel = []
test_accuracy = []
            
for epoch in range(NUM_EPOCHS):
    loss = accuracy = accclevel = 0
    for num_iter, (x, y) in enumerate(dataloader_train):
        x = Variable(x)
        y = Variable(y)
        if USE_CUDA:
            x = x.cuda()
            y = y.cuda()

        optimizer.zero_grad()
        outputs = model(x, y, is_training=True)

        loss_batch = criterion(outputs, y)
        loss_batch.backward()
        # torch.nn.utils.clip_grad_value_(model.parameters(), 20.0)
        optimizer.step()

        max_len = y.size(1)
        accuracy_clevel, accuracy_all = compute_seq_acc(outputs, y, args.batch_size, max_len)

        loss += loss_batch.item()
        accuracy += accuracy_all
        accclevel += accuracy_clevel

    train_loss.append(loss / len(dataloader_train))
    train_accuracy.append(accuracy / len(dataloader_train))
    train_accclevel.append(accclevel / len(dataloader_train))
    print("{} epoch train loss: {}\n"
          "epoch train accuracy: {} accclevel {}".format(epoch, train_loss[-1], train_accuracy[-1],
                                                         train_accclevel[-1]))

    model = model.eval()
    loss = accuracy = accclevel = 0
    for num_iter, (x, y) in enumerate(dataloader_test):
        x = Variable(x)
        y = Variable(y)
        if USE_CUDA:
            x = x.cuda()
            y = y.cuda()

        outputs = model(x, y, is_training=False)

        loss_batch = criterion(outputs, y)

        max_len = y.size(1)
        accuracy_clevel, accuracy_all = compute_seq_acc(outputs, y, args.batch_size, max_len)

        loss += loss_batch.item()
        accuracy += accuracy_all
        accclevel += accuracy_clevel

    test_loss.append(loss / len(dataloader_test))
    test_accuracy.append(accuracy / len(dataloader_test))
    test_accclevel.append(accclevel / len(dataloader_test))
    print("test loss: {}\n"
          "test accuracy: {} accclevel {}\n".format(test_loss[-1], test_accuracy[-1],
                                                    test_accclevel[-1]))
    model = model.train()


fig = plt.figure(figsize=(20, 10))
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)
ax1.plot(train_loss, 'r', label='train_loss')
ax1.plot(test_loss, 'b', label='test_loss')
ax1.legend()
ax2.plot(train_accuracy, 'r', label='train_acc')
ax2.plot(train_accclevel, 'y', label='train_acccl')
ax2.plot(test_accuracy, 'b', label='test_acc')
ax2.plot(test_accclevel, 'g', label='test_acccl')
ax2.legend()
test_acc_array = np.array(test_accuracy)
show_max = str(test_acc_array[-1].item())
ax2.annotate(show_max, xytext=(int(args.epoch), test_acc_array[-1].item()),
             xy=(int(args.epoch), test_acc_array[-1].item()))

path = args.dataset + "_" + args.label
fig.savefig("result/" + path + ".png")

torch.save(model.state_dict(), "model/" + path + ".pth")
