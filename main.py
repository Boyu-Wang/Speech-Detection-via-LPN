import argparse

import torch
print(torch.__version__)
import torch.nn as nn
import torch.utils.data as DD
import torchnet as tnt

import os
import gc

from data_loader import LSW

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

parser = argparse.ArgumentParser(description='PyTorch')
parser.add_argument('--gpuid', type=int, default=0, metavar='G',
                    help='gpuid')
parser.add_argument('--batch_size', type=int, default=64, metavar='BS',
                    help='batch size')
parser.add_argument('--data_path', type=str, default='dataset/LSW/', metavar='DP',
                    help='data path')
parser.add_argument('--model_type', type=str, default='lpn', metavar='DT',
                    help='model type: appearance, lpn, appearance_lpn')
parser.add_argument('--lr', '--learning_rate', default=0.0002, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--start_epoch', type=int, default=0, metavar='SE',
                    help='start epoch number')
parser.add_argument('--epochs', type=int, default=100, metavar='NE',
                    help='number of epochs')
parser.add_argument('--is_train', type=str2bool, default=True, metavar='IT',
                    help='whether to train the model')
parser.add_argument('--dropout', type=float, default=0.25, metavar='DO',
                    help='dropout on rnn: 0.25')
parser.add_argument('--seq_len', type=int, default=20, metavar='PN')
parser.add_argument('--log_interval', type=int, default=50, metavar='N',
                    help='interval between training status logs (default: 10)')

args = parser.parse_args()

print(args)
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpuid)

use_cuda = torch.cuda.is_available()
print('use cuda: %s' % (use_cuda))

import log_utils as log_utils
import dir_utils as dir_utils
import pytorch_utils as pt_utils


def to_np(x):
    return x.cpu().data.numpy()


def main():
    best_acc = 0
    num_class = 2

    if args.model_type == 'appearance':
        from model import AppearanceModel
        net = AppearanceModel(dropout=args.dropout, num_class=num_class)
    elif args.model_type == 'lpn':
        from model import LPN
        net = LPN(dropout=args.dropout, num_class=num_class)
    elif args.model_type == 'appearance_lpn':
        from model import AppearanceLPNModel
        net = AppearanceLPNModel(dropout=args.dropout, num_class=num_class)
    else:
        raise NotImplementedError
    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss()
    if use_cuda:
        net.cuda()
        criterion.cuda()
    optimizer = torch.optim.Adam(net.parameters(), weight_decay=1e-4, lr=args.lr)

    path_to_save = dir_utils.get_dir(os.path.join('ckpts', 'model-{:s}_T-{:d}_D-{:.2f}'.
                                                  format(args.model_type, args.seq_len, args.dropout)))

    log_file = os.path.join(path_to_save, 'log-train-{:s}_{:s}.txt'.format(str(args.is_train), dir_utils.get_date_str()))
    logger = log_utils.get_logger(log_file)
    log_utils.print_config(vars(args), logger)

    if args.start_epoch > 0 or not args.is_train:
        logger.info('loading model from ' + path_to_save)
        pt_utils.load_checkpoint(net, path_to_save, 'model_best.pth', optimizer=optimizer)

    # Data loading code
    trSet = LSW(root_path=args.data_path, subset='train', seq_len=args.seq_len)
    testSet = LSW(root_path=args.data_path, subset='test', seq_len=args.seq_len)

    trLD = DD.DataLoader(trSet, batch_size=args.batch_size,
           sampler=DD.sampler.RandomSampler(trSet),
           num_workers=8, pin_memory=True)
    testLD = DD.DataLoader(testSet, batch_size=args.batch_size,
           sampler=DD.sampler.SequentialSampler(testSet),
           num_workers=8, pin_memory=True)

    if not args.is_train:
        _, avgpool_acc, lastpool_acc = run_one_epoch(testLD, net, criterion, None, 0, logger)
        logger.info('test avgpool accuray: %.4f'%(avgpool_acc))
        logger.info('test lastpool accuray: %.4f' % (lastpool_acc))

        return

    patienceThre = 20
    patience = 0
    lr = args.lr
    reduced = 1
    for epoch in range(args.start_epoch, args.epochs):
        # train for one epoch
        epoch_loss, avgpool_acc, lastpool_acc = run_one_epoch(trLD, net, criterion, optimizer, epoch, logger)
        logger.info('train avgpool accuray: %.4f' % (avgpool_acc))
        logger.info('train lastpool accuray: %.4f' % (lastpool_acc))

        # evaluate
        if epoch % 1 == 0:
            _, avgpool_acc, lastpool_acc = run_one_epoch(testLD, net, criterion, None, epoch, logger)
            logger.info('test avgpool accuray: %.4f' % (avgpool_acc))
            logger.info('test lastpool accuray: %.4f' % (lastpool_acc))

            # remember best prec@1 and save checkpoint
            is_best = avgpool_acc > best_acc
            best_acc = max(avgpool_acc, best_acc)

            if is_best:
                logger.info('best avgpool accuray: %.4f' % (avgpool_acc))
                logger.info('best lastpool accuray: %.4f' % (lastpool_acc))
                pt_utils.save_checkpoint(net, optimizer, path_to_save, filename='checkpoint_e%d.pth' % (epoch),
                                         is_best=is_best, bestname='model_best.pth', only_best=True)
                patience = 0
            elif patience > int(patienceThre / reduced + 0.5):
                reduced = reduced * 2
                lr = lr * 0.1
                logger.info('learning rate: %.f' % lr)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
                patience = 0
            else:
                patience += 1

    pt_utils.load_checkpoint(net, path_to_save, 'model_best.pth')
    _, avgpool_acc, lastpool_acc = run_one_epoch(testLD, net, criterion, None, 0, logger)
    logger.info('test avgpool accuray: %.4f' % (avgpool_acc))
    logger.info('test lastpool accuray: %.4f' % (lastpool_acc))


def run_one_epoch(data_loader, net, criterion, optimizer, epoch, logger):
    if optimizer is not None:
        net.train()
        is_training = True
        log_interval = args.log_interval
    else:
        net.eval()
        is_training = False
        log_interval = len(data_loader) - 1

    avgpool_acc = tnt.meter.ClassErrorMeter(accuracy=True)
    lastpool_acc = tnt.meter.ClassErrorMeter(accuracy=True)
    meter_loss = tnt.meter.AverageValueMeter()

    num_samples = 0

    for i, (img_sequence, landmark_sequence, label) in enumerate(data_loader):
        batch_size = img_sequence.size(0)
        num_samples += batch_size

        img_sequence = img_sequence.float().cuda()
        landmark_sequence = landmark_sequence.long().cuda()
        label = label.long().cuda()

        net.zero_grad()
        # compute output
        # avg_output_logits: [batch_size, num_class]
        avg_output_logits, last_output_logits = net(img_sequence, landmark_sequence//2)

        loss = criterion(avg_output_logits, label)

        # add meter_loss
        meter_loss.add(loss.data.item())
        avgpool_acc.add(avg_output_logits.data, label)
        lastpool_acc.add(last_output_logits.data, label)


        # compute gradient and do SGD step
        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            clip_gradient(net, 10)
            optimizer.step()

        if (is_training and i % log_interval == 0) or (not is_training and i == log_interval):
            logger.info('Epoch: [{0}][{1}/{2}]\t'
                        'Loss: {3}\t'
                        'meanpool Accuracy: {4}\t'
                        'last Accuracy: {5}\t'.format(
                epoch, i, len(data_loader), meter_loss.value()[0], avgpool_acc.value()[0], lastpool_acc.value()[0]))

        gc.collect()

    return meter_loss.value()[0], avgpool_acc.value()[0], lastpool_acc.value()[0]


def clip_gradient(model, clip):
    for p in model.parameters():
        p.grad.data.clamp_(-clip, clip)


if __name__ == '__main__':
    main()

