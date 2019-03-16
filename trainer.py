from __future__ import print_function
import pickle
import os
import sys
import time
import shutil
import json
import numpy as np

import torch

import evaluation
import util.data_provider as data
from util.vocab import Vocabulary
from util.text2vec import get_text_encoder
from model import get_model, get_we_parameter

import logging
import tensorboard_logger as tb_logger

import argparse

from basic.constant import ROOT_PATH
from basic.bigfile import BigFile
from basic.common import makedirsforfile, checkToSkip
from basic.util import read_dict, AverageMeter, LogCollector
from basic.generic_utils import Progbar

INFO = __file__


def parse_args():
    # Hyper Parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--rootpath', type=str, default=ROOT_PATH,
                        help='path to datasets. (default: %s)'%ROOT_PATH)
    parser.add_argument('trainCollection', type=str, help='train collection')
    parser.add_argument('valCollection', type=str,  help='validation collection')
    parser.add_argument('testCollection', type=str,  help='test collection')
    parser.add_argument('--n_caption', type=int, default=20, help='number of captions of each image/video (default: 1)')
    parser.add_argument('--overwrite', type=int, default=0, choices=[0,1], help='overwrite existed file. (default: 0)')
    # model
    parser.add_argument('--model', type=str, default='dual_encoding', help='model name. (default: dual_encoding)')
    parser.add_argument('--measure', type=str, default='cosine', help='measure method. (default: cosine)')
    parser.add_argument('--dropout', default=0.2, type=float, help='dropout rate (default: 0.2)')
    # text-side multi-level encoding
    parser.add_argument('--vocab', type=str, default='word_vocab_5', help='word vocabulary. (default: word_vocab_5)')
    parser.add_argument('--word_dim', type=int, default=500, help='word embedding dimension')
    parser.add_argument('--text_rnn_size', type=int, default=512, help='text rnn encoder size. (default: 1024)')
    parser.add_argument('--text_kernel_num', default=512, type=int, help='number of each kind of text kernel')
    parser.add_argument('--text_kernel_sizes', default='2-3-4', type=str, help='dash-separated kernel size to use for text convolution')
    parser.add_argument('--text_norm', action='store_true', help='normalize the text embeddings at last layer')
    # video-side multi-level encoding
    parser.add_argument('--visual_feature', type=str, default='resnet-152-img1k-flatten0_outputos', help='visual feature.')
    parser.add_argument('--visual_rnn_size', type=int, default=1024, help='visual rnn encoder size')
    parser.add_argument('--visual_kernel_num', default=512, type=int, help='number of each kind of visual kernel')
    parser.add_argument('--visual_kernel_sizes', default='2-3-4-5', type=str, help='dash-separated kernel size to use for visual convolution')
    parser.add_argument('--visual_norm', action='store_true', help='normalize the visual embeddings at last layer')
    # common space learning
    parser.add_argument('--text_mapping_layers', type=str, default='0-2048', help='text fully connected layers for common space learning. (default: 0-2048)')
    parser.add_argument('--visual_mapping_layers', type=str, default='0-2048', help='visual fully connected layers  for common space learning. (default: 0-2048)')
    # loss
    parser.add_argument('--loss_fun', type=str, default='mrl', help='loss function')
    parser.add_argument('--margin', type=float, default=0.2, help='rank loss margin')
    parser.add_argument('--direction', type=str, default='all', help='retrieval direction (all|t2i|i2t)')
    parser.add_argument('--max_violation', action='store_true', help='use max instead of sum in the rank loss')
    parser.add_argument('--cost_style', type=str, default='sum', help='cost style (sum, mean). (default: sum)')
    # optimizer
    parser.add_argument('--optimizer', type=str, default='adam', help='optimizer. (default: rmsprop)')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='initial learning rate')
    parser.add_argument('--lr_decay_rate', default=0.99, type=float, help='learning rate decay rate. (default: 0.99)')
    parser.add_argument('--grad_clip', type=float, default=2, help='gradient clipping threshold')
    parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
    parser.add_argument('--val_metric', default='recall', type=str, help='performance metric for validation (mir|recall)')
    # misc
    parser.add_argument('--num_epochs', default=50, type=int, help='Number of training epochs.')
    parser.add_argument('--batch_size', default=128, type=int, help='Size of a training mini-batch.')
    parser.add_argument('--workers', default=5, type=int, help='Number of data loader workers.')
    parser.add_argument('--postfix', default='runs_0', help='Path to save the model and Tensorboard log.')
    parser.add_argument('--log_step', default=10, type=int, help='Number of steps to print and record the log.')
    parser.add_argument('--cv_name', default='cvpr_2019', type=str, help='')

    args = parser.parse_args()
    return args


def main():
    opt = parse_args()
    print(json.dumps(vars(opt), indent = 2))

    rootpath = opt.rootpath
    trainCollection = opt.trainCollection
    valCollection = opt.valCollection
    testCollection = opt.testCollection

    if opt.loss_fun == "mrl" and opt.measure == "cosine":
        assert opt.text_norm is True
        assert opt.visual_norm is True

    # checkpoint path
    model_info = '%s_dp_%.1f_measure_%s' %  (opt.model, opt.dropout, opt.measure)
    # text-side multi-level encoding info
    text_encode_info = 'vocab_%s_word_dim_%s_text_rnn_size_%s_text_norm_%s' % \
            (opt.vocab, opt.word_dim, opt.text_rnn_size, opt.text_norm)
    text_encode_info += "_kernel_sizes_%s_num_%s" % (opt.text_kernel_sizes, opt.text_kernel_num)
    # video-side multi-level encoding info
    visual_encode_info = 'visual_feature_%s_visual_rnn_size_%d_visual_norm_%s' % \
            (opt.visual_feature, opt.visual_rnn_size, opt.visual_norm)
    visual_encode_info += "_kernel_sizes_%s_num_%s" % (opt.visual_kernel_sizes, opt.visual_kernel_num)
    # common space learning info
    mapping_info = "mapping_text_%s_img_%s" % (opt.text_mapping_layers, opt.visual_mapping_layers)
    loss_info = 'loss_func_%s_margin_%s_direction_%s_max_violation_%s_cost_style_%s' % \
                    (opt.loss_fun, opt.margin, opt.direction, opt.max_violation, opt.cost_style)
    optimizer_info = 'optimizer_%s_lr_%s_decay_%.2f_grad_clip_%.1f_val_metric_%s' % \
                    (opt.optimizer, opt.learning_rate, opt.lr_decay_rate, opt.grad_clip, opt.val_metric)

    opt.logger_name = os.path.join(rootpath, trainCollection, opt.cv_name, valCollection, model_info, text_encode_info,
                            visual_encode_info, mapping_info, loss_info, optimizer_info, opt.postfix)
    print(opt.logger_name)

    if checkToSkip(os.path.join(opt.logger_name, 'model_best.pth.tar'), opt.overwrite):
        sys.exit(0)
    if checkToSkip(os.path.join(opt.logger_name, 'val_metric.txt'), opt.overwrite):
        sys.exit(0)
    makedirsforfile(os.path.join(opt.logger_name, 'val_metric.txt'))
    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
    tb_logger.configure(opt.logger_name, flush_secs=5)


    opt.text_kernel_sizes = map(int, opt.text_kernel_sizes.split('-'))
    opt.visual_kernel_sizes = map(int, opt.visual_kernel_sizes.split('-'))
    # collections: trian, val
    collections = {'train': trainCollection, 'val': valCollection}
    cap_file = {'train': '%s.caption.txt'%trainCollection, 
                'val': '%s.caption.txt'%valCollection}
    # caption
    caption_files = { x: os.path.join(rootpath, collections[x], 'TextData', cap_file[x])
                        for x in collections }
    # Load visual features
    visual_feat_path = {x: os.path.join(rootpath, collections[x], 'FeatureData', opt.visual_feature)
                        for x in collections }
    visual_feats = {x: BigFile(visual_feat_path[x]) for x in visual_feat_path}
    opt.visual_feat_dim = visual_feats['train'].ndims
    
    # set bow vocabulary and encoding
    bow_vocab_file = os.path.join(rootpath, opt.trainCollection, 'TextData', 'vocabulary', 'bow', opt.vocab+'.pkl')
    bow_vocab = pickle.load(open(bow_vocab_file, 'rb'))
    bow2vec = get_text_encoder('bow')(bow_vocab)
    opt.bow_vocab_size = len(bow_vocab)

    # set rnn vocabulary 
    rnn_vocab_file = os.path.join(rootpath, opt.trainCollection, 'TextData', 'vocabulary', 'rnn', opt.vocab+'.pkl')
    rnn_vocab = pickle.load(open(rnn_vocab_file, 'rb'))
    opt.vocab_size = len(rnn_vocab)
        
    # initialize word embedding
    opt.we_parameter = None
    if opt.word_dim == 500:
        w2v_data_path = os.path.join(rootpath, "word2vec", 'flickr', 'vec500flickr30m')
        opt.we_parameter = get_we_parameter(rnn_vocab, w2v_data_path)

    # mapping layer structure
    opt.text_mapping_layers = map(int, opt.text_mapping_layers.split('-'))
    opt.text_mapping_layers[0] = opt.bow_vocab_size + opt.text_rnn_size*2 + opt.text_kernel_num * len(opt.text_kernel_sizes) 
    opt.visual_mapping_layers = map(int, opt.visual_mapping_layers.split('-'))
    opt.visual_mapping_layers[0] = opt.visual_feat_dim + opt.visual_rnn_size*2 + opt.visual_kernel_num * len(opt.visual_kernel_sizes)

    # set data loader
    video2frames = {x: read_dict(os.path.join(rootpath, collections[x], 'FeatureData', opt.visual_feature, 'video2frames.txt'))
                    for x in collections }
    data_loaders = data.get_data_loaders(
        caption_files, visual_feats, rnn_vocab, bow2vec, opt.batch_size, opt.workers, opt.n_caption, video2frames=video2frames)
       

    # Construct the model
    model = get_model(opt.model)(opt)
    opt.we_parameter = None
    
    # optionally resume from a checkpoint
    if opt.resume:
        if os.path.isfile(opt.resume):
            print("=> loading checkpoint '{}'".format(opt.resume))
            checkpoint = torch.load(opt.resume)
            start_epoch = checkpoint['epoch']
            best_rsum = checkpoint['best_rsum']
            model.load_state_dict(checkpoint['model'])
            # Eiters is used to show logs as the continuation of another
            # training
            model.Eiters = checkpoint['Eiters']
            print("=> loaded checkpoint '{}' (epoch {}, best_rsum {})"
                  .format(opt.resume, start_epoch, best_rsum))
            validate(opt, data_loaders['val'], model, measure=opt.measure)
        else:
            print("=> no checkpoint found at '{}'".format(opt.resume))


    # Train the Model
    best_rsum = 0
    no_impr_counter = 0
    lr_counter = 0
    best_epoch = None
    fout_val_metric_hist = open(os.path.join(opt.logger_name, 'val_metric_hist.txt'), 'w')
    for epoch in range(opt.num_epochs):
        print('Epoch[{0} / {1}] LR: {2}'.format(epoch, opt.num_epochs, get_learning_rate(model.optimizer)[0]))
        print('-'*10)
        # train for one epoch
        train(opt, data_loaders['train'], model, epoch)

        # evaluate on validation set
        rsum = validate(opt, data_loaders['val'], model, measure=opt.measure)


        # remember best R@ sum and save checkpoint
        is_best = rsum > best_rsum
        best_rsum = max(rsum, best_rsum)
        print(' * Current perf: {}'.format(rsum))
        print(' * Best perf: {}'.format(best_rsum))
        print('')
        fout_val_metric_hist.write('epoch_%d: %f\n' % (epoch, rsum))
        fout_val_metric_hist.flush()
        
        if is_best:
            save_checkpoint({
                'epoch': epoch + 1,
                'model': model.state_dict(),
                'best_rsum': best_rsum,
                'opt': opt,
                'Eiters': model.Eiters,
            }, is_best, filename='checkpoint_epoch_%s.pth.tar'%epoch, prefix=opt.logger_name + '/', best_epoch=best_epoch)
            best_epoch = epoch

        lr_counter += 1
        decay_learning_rate(opt, model.optimizer, opt.lr_decay_rate)
        if not is_best:
            # Early stop occurs if the validation performance does not improve in ten consecutive epochs
            no_impr_counter += 1
            if no_impr_counter > 10:
                print('Early stopping happended.\n')
                break

            # When the validation performance decreased after an epoch,
            # we divide the learning rate by 2 and continue training;
            # but we use each learning rate for at least 3 epochs.
            if lr_counter > 2:
                decay_learning_rate(opt, model.optimizer, 0.5)
                lr_counter = 0
        else:
            no_impr_counter = 0

    fout_val_metric_hist.close()

    print('best performance on validation: {}\n'.format(best_rsum))
    with open(os.path.join(opt.logger_name, 'val_metric.txt'), 'w') as fout:
        fout.write('best performance on validation: ' + str(best_rsum))

    # generate evaluation shell script
    templete = ''.join(open( 'util/TEMPLATE_do_test.sh').readlines())
    striptStr = templete.replace('@@@rootpath@@@', rootpath)
    striptStr = striptStr.replace('@@@testCollection@@@', testCollection)
    striptStr = striptStr.replace('@@@logger_name@@@', opt.logger_name)
    striptStr = striptStr.replace('@@@overwrite@@@', str(opt.overwrite))
    striptStr = striptStr.replace('@@@n_caption@@@', str(opt.n_caption))

    # perform evaluation on test set
    runfile = 'do_test_%s_%s.sh' % (opt.model, testCollection)
    open(runfile, 'w').write(striptStr + '\n')
    os.system('chmod +x %s' % runfile)
    # os.system('./'+runfile)


def train(opt, train_loader, model, epoch):
    # average meters to record the training statistics
    batch_time = AverageMeter()
    data_time = AverageMeter()
    train_logger = LogCollector()

    # switch to train mode
    model.train_start()

    progbar = Progbar(len(train_loader.dataset))
    end = time.time()
    for i, train_data in enumerate(train_loader):

        # measure data loading time
        data_time.update(time.time() - end)

        # make sure train logger is used
        model.logger = train_logger

        # Update the model
        b_size, loss = model.train_emb(*train_data)

        progbar.add(b_size, values=[('loss', loss)])

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # Record logs in tensorboard
        tb_logger.log_value('epoch', epoch, step=model.Eiters)
        tb_logger.log_value('step', i, step=model.Eiters)
        tb_logger.log_value('batch_time', batch_time.val, step=model.Eiters)
        tb_logger.log_value('data_time', data_time.val, step=model.Eiters)
        model.logger.tb_log(tb_logger, step=model.Eiters)


def validate(opt, val_loader, model, measure='cosine'):
    # compute the encoding for all the validation video and captions
    video_embs, cap_embs, video_ids, caption_ids = evaluation.encode_data(model, val_loader, opt.log_step, logging.info)

    # we load data as video-sentence pairs
    # but we only need to forward each video once for evaluation
    # so we get the video set and mask out same videos with feature_mask
    feature_mask = []
    evaluate_videos = set()
    for video_id in video_ids:
        feature_mask.append(video_id not in evaluate_videos)
        evaluate_videos.add(video_id)
    video_embs = video_embs[feature_mask]
    video_ids = [x for idx, x in enumerate(video_ids) if feature_mask[idx] is True]

    c2i_all_errors = evaluation.cal_error(video_embs, cap_embs, measure)
    if opt.val_metric == "recall":

        # video retrieval
        (r1i, r5i, r10i, medri, meanri) = evaluation.t2i(c2i_all_errors, n_caption=opt.n_caption)
        print(" * Text to video:")
        print(" * r_1_5_10: {}".format([round(r1i, 3), round(r5i, 3), round(r10i, 3)]))
        print(" * medr, meanr: {}".format([round(medri, 3), round(meanri, 3)]))
        print(" * "+'-'*10)

        # caption retrieval
        (r1, r5, r10, medr, meanr) = evaluation.i2t(c2i_all_errors, n_caption=opt.n_caption)
        print(" * Video to text:")
        print(" * r_1_5_10: {}".format([round(r1, 3), round(r5, 3), round(r10, 3)]))
        print(" * medr, meanr: {}".format([round(medr, 3), round(meanr, 3)]))
        print(" * "+'-'*10)

        # record metrics in tensorboard
        tb_logger.log_value('r1', r1, step=model.Eiters)
        tb_logger.log_value('r5', r5, step=model.Eiters)
        tb_logger.log_value('r10', r10, step=model.Eiters)
        tb_logger.log_value('medr', medr, step=model.Eiters)
        tb_logger.log_value('meanr', meanr, step=model.Eiters)
        tb_logger.log_value('r1i', r1i, step=model.Eiters)
        tb_logger.log_value('r5i', r5i, step=model.Eiters)
        tb_logger.log_value('r10i', r10i, step=model.Eiters)
        tb_logger.log_value('medri', medri, step=model.Eiters)
        tb_logger.log_value('meanri', meanri, step=model.Eiters)


    elif opt.val_metric == "map":
        i2t_map_score = evaluation.i2t_map(c2i_all_errors, n_caption=opt.n_caption)
        t2i_map_score = evaluation.t2i_map(c2i_all_errors, n_caption=opt.n_caption)
        tb_logger.log_value('i2t_map', i2t_map_score, step=model.Eiters)
        tb_logger.log_value('t2i_map', t2i_map_score, step=model.Eiters)
        print('i2t_map', i2t_map_score)
        print('t2i_map', t2i_map_score)

    currscore = 0
    if opt.val_metric == "recall":
        if opt.direction == 'i2t' or opt.direction == 'all':
            currscore += (r1 + r5 + r10)
        if opt.direction == 't2i' or opt.direction == 'all':
            currscore += (r1i + r5i + r10i)
    elif opt.val_metric == "map":
        if opt.direction == 'i2t' or opt.direction == 'all':
            currscore += i2t_map_score
        if opt.direction == 't2i' or opt.direction == 'all':
            currscore += t2i_map_score

    tb_logger.log_value('rsum', currscore, step=model.Eiters)

    return currscore


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar', prefix='', best_epoch=None):
    """save checkpoint at specific path"""
    torch.save(state, prefix + filename)
    if is_best:
        shutil.copyfile(prefix + filename, prefix + 'model_best.pth.tar')
    if best_epoch is not None:
        os.remove(prefix + 'checkpoint_epoch_%s.pth.tar'%best_epoch)

def decay_learning_rate(opt, optimizer, decay):
    """decay learning rate to the last LR"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr']*decay

def get_learning_rate(optimizer):
    """Return learning rate"""
    lr_list = []
    for param_group in optimizer.param_groups:
        lr_list.append(param_group['lr'])
    return lr_list

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    main()
