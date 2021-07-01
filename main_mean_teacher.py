import argparse
import torch
from torch import optim
import matplotlib
from torch.autograd import Variable

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from datasets import load_datasets_mean_teacher
from models import CNNSeq2Seq
from util import compute_seq_acc, ClassLoss, ConsistentLoss

parser = argparse.ArgumentParser(description='PyTorch Captcha Training Using Mean-Teacher')

parser.add_argument('--dataset', default='google', type=str, help="the name of dataset")
parser.add_argument('--label', default="700.txt", type=str, help='the labels of captcha images used for training')
parser.add_argument('--batch-size', default=32, type=int, help='batch size for training and test')
parser.add_argument('--secondary-batch-size', default=64, type=int, help='batch size for unlabel')
parser.add_argument('--unlabeled-number', default=5000, type=int, help='the number of unlabeled images')
parser.add_argument('--epoch', default=500, type=int, help='the number of training epochs')
parser.add_argument('--pretrained', action="store_false", help='whether to use pretrained model(Default: True)')

args = parser.parse_args()

USE_CUDA = torch.cuda.is_available()

LR = 0.001
NUM_EPOCHS = args.epoch

dataloader_train_labeled, dataloader_train_nolabeled, dataloader_test, id2token, _, _ = load_datasets_mean_teacher(args)

print("token:", "".join(list(id2token.values())))

model = CNNSeq2Seq(vocab_size=len(id2token))
model_ema = CNNSeq2Seq(vocab_size=len(id2token))
class_criterion = ClassLoss()
consistent_criterion = ConsistentLoss()

if USE_CUDA:
    model = model.cuda()
    model_ema = model_ema.cuda()
    class_criterion = class_criterion.cuda()
    consistent_criterion = consistent_criterion.cuda()

for param in model_ema.parameters():
    param.detach_()

if args.pretrained:
    path = args.dataset + "_" + args.label
    model.load_state_dict(torch.load("model/" + path + ".pth"))
    model_ema.load_state_dict(torch.load("model/" + path + ".pth"))

params = list(filter(lambda p: p.requires_grad, model.parameters()))
optimizer = optim.SGD(params, lr=LR, momentum=0.9)

torch.backends.cudnn.benchmark = True

train_loss_class = []
train_loss_consistency = []

test_class_loss = []
test_class_loss_ema = []
test_accclevel = []
test_accclevel_ema = []
test_accuracy = []
test_accuracy_ema = []

for epoch in range(NUM_EPOCHS):
    loss_1 = loss_2 = 0

    i = 0
    iter_label = dataloader_train_labeled.__iter__()
    for num_iter, (x_nolabel, y_nolabel) in enumerate(dataloader_train_nolabeled):
        x_label, y_label = iter_label.next()
        i += 1
        if i == len(iter_label):
            i = 0
            iter_label = dataloader_train_labeled.__iter__()

        x = Variable(torch.cat([x_nolabel, x_label], 0))
        y = Variable(torch.cat([y_nolabel, y_label], 0))
        if USE_CUDA:
            x = x.cuda()
            y = y.cuda()

        optimizer.zero_grad()

        max_len = y.size(1)
        outputs = model.forward_2(x, max_len)
        outputs_ema = model_ema.forward_2(x, max_len)

        class_loss = class_criterion(outputs, y)
        consistent_loss = 30 * consistent_criterion(outputs, outputs_ema)
        loss_all = class_loss + consistent_loss
        loss_all.backward()
        torch.nn.utils.clip_grad_value_(model.parameters(), 10.0)
        optimizer.step()
        for ema_param, param in zip(model_ema.parameters(), model.parameters()):
            ema_param.data.mul_(0.99).add_(1 - 0.99, param.data)

        loss_1 += class_loss.item()
        loss_2 += consistent_loss.item()

    train_loss_class.append(loss_1 / len(dataloader_train_nolabeled))
    train_loss_consistency.append(loss_2 / len(dataloader_train_nolabeled))
    print("{} epoch train\n"
          "class loss: {} consistent loss {}".format(epoch, train_loss_class[-1], train_loss_consistency[-1]))

    model = model.eval()
    loss = accuracy = accclevel = 0
    for num_iter, (x, y) in enumerate(dataloader_test):
        x = Variable(x)
        y = Variable(y)
        if USE_CUDA:
            x = x.cuda()
            y = y.cuda()

        outputs = model(x, y, is_training=False)

        loss_batch = class_criterion(outputs, y)

        max_len = y.size(1)
        accuracy_clevel, accuracy_all = compute_seq_acc(outputs, y, args.batch_size, max_len)

        loss += loss_batch.item()
        accuracy += accuracy_all
        accclevel += accuracy_clevel

    test_class_loss.append(loss / len(dataloader_test))
    test_accuracy.append(accuracy / len(dataloader_test))
    test_accclevel.append(accclevel / len(dataloader_test))
    print("test loss: {}\n"
          "test accuracy: {} accclevel {}".format(test_class_loss[-1], test_accuracy[-1],
                                                  test_accclevel[-1]))
    model = model.train()

    model_ema = model_ema.eval()
    loss = accuracy = accclevel = 0
    for num_iter, (x, y) in enumerate(dataloader_test):
        x = Variable(x)
        y = Variable(y)
        if USE_CUDA:
            x = x.cuda()
            y = y.cuda()

        outputs = model_ema(x, y, is_training=False)

        loss_batch = class_criterion(outputs, y)

        max_len = y.size(1)
        accuracy_clevel, accuracy_all = compute_seq_acc(outputs, y, args.batch_size, max_len)

        loss += loss_batch.item()
        accuracy += accuracy_all
        accclevel += accuracy_clevel

    test_class_loss_ema.append(loss / len(dataloader_test))
    test_accuracy_ema.append(accuracy / len(dataloader_test))
    test_accclevel_ema.append(accclevel / len(dataloader_test))
    print("test loss: {}\n"
          "test accuracy: {} accclevel {}\n".format(test_class_loss_ema[-1], test_accuracy_ema[-1],
                                                  test_accclevel_ema[-1]))
    model_ema = model_ema.train()

fig = plt.figure(figsize=(20, 10))
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)
ax1.plot(train_loss_class, 'r', label='train_class_loss')
ax1.plot(train_loss_consistency, 'y', label='train_consistency_loss')
ax1.plot(test_class_loss, 'b', label='test_class_loss')
ax1.plot(test_class_loss_ema, 'g', label='test_ema_class_loss')

ax1.legend()
ax2.plot(test_accuracy, 'm', label='test_acc')
ax2.plot(test_accclevel, 'y', label='test_accl')
ax2.plot(test_accuracy_ema, 'k', label='test_acc_ema')
ax2.plot(test_accclevel_ema, 'w', label='test_accl_ema')
ax2.legend()
test_acc_array = np.array(test_accuracy_ema)
max_indx = np.argmax(test_acc_array)
show_max = '['+str(max_indx)+ " " + str(test_acc_array[max_indx].item())+']'
ax2.annotate(show_max,xytext=(max_indx,test_acc_array[max_indx].item()),xy=(max_indx,test_acc_array[max_indx].item()))

path = "MT_" + args.dataset + "_" + args.label
fig.savefig("result/" + path + ".png")

torch.save(model_ema.state_dict(), "model/" + path + ".pth")
