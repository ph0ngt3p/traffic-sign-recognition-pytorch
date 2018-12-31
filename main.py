import torch
import torchvision
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import time
import os
import sys
import argparse
import torch.utils.data as utilsData
import torchvision.models as models
import numpy as np
import csv
import torchvision.transforms as transforms
import datetime
import time
from tqdm import tqdm
from logger import Logger

from model import *
from utils import *

PROJECT_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.join(PROJECT_DIR, 'data')
TRAINVAL_PATH = os.path.join(DATA_DIR, 'train_val_data.hdf5')
TEST_PATH = os.path.join(DATA_DIR, 'test_data.hdf5')
LOG_DIR = os.path.join(PROJECT_DIR, 'logs')
RESULT_FILE = os.path.join(PROJECT_DIR, 'test_results')

parser = argparse.ArgumentParser()
parser.add_argument('--lr', default=1e-2, type=float, help='Learning Rate')
parser.add_argument('--optim', default='sgd', choices=['adam', 'sgd'], type=str,
                    help='Using Adam or SGD optimizer for model')
parser.add_argument('--predict_option', default=0, type=int, choices=[0, 1],
                    help='0: predict with best acc model -- 1: predict with convergence model')
parser.add_argument('--resume', '-r', action='store_true', help='Resume from checkpoint')
parser.add_argument('--train', '-tr', action='store_true', help='Train the model')
parser.add_argument('--predict', '-pr', action='store_true', help='Predict the data')
parser.add_argument('--inspect', '-ins', action='store_true', help='Inspect saved model')
parser.add_argument('--interval', default=250, type=int, help='Number of epochs to train the model')
parser.add_argument('--conv', default=False, type=bool, help='Use convergence model for predict')
parser.add_argument('--weight_decay', default=5e-6, type=float, help='Weight decay')
parser.add_argument('--batch_size', default=256, type=int, help='Batch size - default: 256')
parser.add_argument('--check_after', default=1, type=int, help='Validate the model after how many epoch - default : 1')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()
print(torch.cuda.is_available())

device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
print('Available device is:{}'.format(device))
print(torch.cuda.current_device)

start_epoch = 0
save_acc = 0
save_loss = 0

transform_train = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

transform_test = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

train_logger = Logger(os.path.join(LOG_DIR, 'train'))
val_logger = Logger(os.path.join(LOG_DIR, 'val'))


# cudnn.benchmark = True
def exp_lr_schedule(args, optimizer, epoch):
    init_lr = args.lr
    lr_decay_epoch = 50
    weight_decay = args.weight_decay
    lr = init_lr * (0.6 ** (min(epoch, 200) // lr_decay_epoch))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        param_group['weight_decay'] = weight_decay

    return optimizer, lr


def save_convergence_model(save_loss, model, epoch):
    print('Saving convergence model at epoch {} with loss {}'.format(epoch, save_loss))
    state = {
        'model' : model.state_dict(),
        'loss'	: save_loss,
        'epoch'	: epoch
    }
    if not os.path.isdir('./checkpoint'):
        os.mkdir('./checkpoint')
    torch.save(state, './checkpoint/convergence.t7')


def save_best_acc_model(save_acc, model, epoch):
    print('Saving best acc model at epoch {} with acc in validation set: {}'.format(epoch, save_acc))
    state = {
        'model'	: model.state_dict(),
        'acc'	: save_acc,
        'epoch' : epoch,
    }

    if not os.path.isdir('./checkpoint'):
        os.mkdir('./checkpoint')
    torch.save(state, './checkpoint/best_acc_model.t7')


def train_validate(epoch, optimizer, model, criterion, train_loader, validate_loader):
    optimizer, lr = exp_lr_schedule(args, optimizer, epoch)
    model.train()
    train_loss = 0
    train_acc = 0
    total = 0

    pbar = tqdm(enumerate(train_loader))
    for idx, (images, labels) in pbar:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)

        total += labels.size(0)
        train_acc += predicted.eq(labels).sum().item()

        if idx % args.log_interval == 0:
            pbar.set_description(
                'Train Epoch: {:3d} [{:8d}/{:8d} ({:3.0f}%)]\tLoss: {:.6f}\tTrain accuracy: {:.6f}\tLR: {:.3}'.format(
                    epoch + 1, idx * len(images), len(train_loader.dataset),
                    100. * idx / len(train_loader),
                    loss.item(), train_acc / total, lr))

    epoch_acc = train_acc / total
    epoch_loss = train_loss / (idx + 1)

    # Log scalar values (scalar summary)
    info = {'loss': epoch_loss, 'accuracy': epoch_acc}

    for tag, value in info.items():
        train_logger.scalar_summary(tag, value, epoch + 1)

    global save_loss
    # print(save_loss)
    if epoch == 0:
        save_loss = epoch_loss
        save_convergence_model(save_loss, model, epoch)
    else:
        if epoch_loss < save_loss:
            save_loss = epoch_loss
            save_convergence_model(save_loss, model, epoch)

    if (epoch + 1) % args.check_after == 0:
        print('==============================================\n')
        print('==> Validate in epoch {} at LR {:.3}'.format(epoch + 1, lr))
        model.eval()
        validate_loss = 0
        validate_acc = 0
        validate_correct = 0
        total = 0
        pbar = tqdm(enumerate(validate_loader))
        for idx, (images, labels) in pbar:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)

            validate_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            validate_correct += predicted.eq(labels).sum().item()

        validate_acc = validate_correct / total
        print('Validation accuracy : {:.6f}'.format(validate_acc))

        # Log scalar values (scalar summary)
        info = {'loss': validate_loss / (idx + 1), 'accuracy': validate_acc}

        for tag, value in info.items():
            val_logger.scalar_summary(tag, value, epoch + 1)

        global save_acc
        # print(save_acc)
        if validate_acc > save_acc:
            save_acc = validate_acc
            save_best_acc_model(save_acc, model, epoch)


def predict(model, test_loader, convergence):
    assert os.path.isdir('./checkpoint'), 'Error: model is not availabel!'
    if convergence:
        ckpt = torch.load('./checkpoint/convergence.t7')
        model.load_state_dict(ckpt['model'])
        loss = ckpt['loss']
        epoch = ckpt['epoch']
        print('Model used to predict converges at epoch {} and loss {:.6}'.format(epoch, loss))
    else:
        ckpt = torch.load('./checkpoint/best_acc_model.t7')
        model.load_state_dict(ckpt['model'])
        acc = ckpt['acc']
        epoch = ckpt['epoch']
        print('Model used to predict has best acc {:.6} on validate set at epoch {}'.format(acc, epoch))

    torch.set_grad_enabled(False)
    model.eval()
    test_correct = 0
    total = 0

    f = open(RESULT_FILE + str(datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')) + '.csv', 'w+')
    f.write('correct,predicted\n')

    for idx, (images, labels) in enumerate(test_loader):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = outputs.max(1)
        test_correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

        outputs = outputs.data.cpu().numpy()
        outputs = np.argsort(outputs, axis=1)[:, -3:][:, ::-1]
        for i, image_id in enumerate(labels):
            tmp = gen_outputline(image_id.data.cpu().numpy(), list(outputs[i]))
            f.write(tmp)

        if idx % 2000 and idx > 1:
            print("Processing {}/{}".format(idx + 1, len(test_loader)))

    print('Accuracy on test set: {:.6}'.format(test_correct/total))


model = Net()
model.to(device)
# model = params_initializer(model)
if device == 'cuda':
    model = torch.nn.DataParallel(model)
criterion = nn.CrossEntropyLoss()
if args.optim == 'sgd':
    optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=0.9)
elif args.optim == 'adam':
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)


if args.train:
    train_set = MyTrainDataset(TRAINVAL_PATH, root_dir='./data', train=True, transform=transform_train)
    val_set = MyTrainDataset(TRAINVAL_PATH, root_dir='./data', train=False, transform=transform_test)
    train_loader = utilsData.DataLoader(dataset=train_set, batch_size=args.batch_size, sampler=None, shuffle=True,
                                        batch_sampler=None)
    val_loader = utilsData.DataLoader(dataset=val_set, batch_size=args.batch_size, sampler=None, shuffle=False,
                                      batch_sampler=None)

    for epoch in range(start_epoch + args.interval):
        train_validate(epoch, optimizer, model, criterion, train_loader, val_loader)

if args.predict:
    test_set = MyTestDataset(TEST_PATH, root_dir='./data', transform=transform_test)
    test_loader = utilsData.DataLoader(dataset=test_set, batch_size=args.batch_size, sampler=None, shuffle=False,
                                       batch_sampler=None)
    if args.predict_option == 0:
        conv = False
    else:
        conv = True
    predict(model, test_loader, conv)

if args.inspect:

    checkpoint = torch.load('./checkpoint/convergence.t7')
    loss = checkpoint['loss']
    epoch = checkpoint['epoch']
    print('Model used to predict converges at epoch {} and loss {:.3}'.format(epoch, loss))

    checkpoint = torch.load('./checkpoint/best_acc_model.t7')
    acc = checkpoint['acc']
    epoch = checkpoint['epoch']
    print('Model used to predict has best acc {:3} on validate set at epoch {}'.format(acc, epoch))

# if args.resume:
#     print('==> Resume model from last checkpoint ...')
#     assert os.path.isdir('./checkpoint'), 'Error: no checkpoint directory found!'
#     checkpoint = torch.load
