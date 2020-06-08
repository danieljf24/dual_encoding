from __future__ import print_function
import pickle
import os
import sys

import torch

import evaluation
from model import get_model
import util.data_provider as data
from util.vocab import Vocabulary
from util.text2vec import get_text_encoder

import logging
import json
import numpy as np

import argparse
from basic.util import read_dict
from basic.constant import ROOT_PATH
from basic.bigfile import BigFile
from basic.common import makedirsforfile, checkToSkip

def parse_args():
    # Hyper Parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('testCollection', type=str, help='test collection')
    parser.add_argument('--rootpath', type=str, default=ROOT_PATH, help='path to datasets. (default: %s)'%ROOT_PATH)
    parser.add_argument('--overwrite', type=int, default=0, choices=[0,1],  help='overwrite existed file. (default: 0)')
    parser.add_argument('--log_step', default=10, type=int, help='Number of steps to print and record the log.')
    parser.add_argument('--batch_size', default=128, type=int, help='Size of a training mini-batch.')
    parser.add_argument('--workers', default=5, type=int, help='Number of data loader workers.')
    parser.add_argument('--logger_name', default='runs', help='Path to save the model and Tensorboard log.')
    parser.add_argument('--checkpoint_name', default='model_best.pth.tar', type=str, help='name of checkpoint (default: model_best.pth.tar)')
    parser.add_argument('--n_caption', type=int, default=20, help='number of captions of each image/video (default: 1)')

    args = parser.parse_args()
    return args

def load_config(config_path):
    variables = {}
    exec(compile(open(config_path, "rb").read(), config_path, 'exec'), variables)
    return variables['config']

def main():
    opt = parse_args()
    print(json.dumps(vars(opt), indent=2))

    rootpath = opt.rootpath
    testCollection = opt.testCollection
    n_caption = opt.n_caption
    resume = os.path.join(opt.logger_name, opt.checkpoint_name)

    if not os.path.exists(resume):
        logging.info(resume + ' not exists.')
        sys.exit(0)


    checkpoint = torch.load(resume)
    start_epoch = checkpoint['epoch']
    best_rsum = checkpoint['best_rsum']
    print("=> loaded checkpoint '{}' (epoch {}, best_rsum {})"
          .format(resume, start_epoch, best_rsum))
    options = checkpoint['opt']
    if not hasattr(options, 'concate'):
        setattr(options, "concate", "full")

    trainCollection = options.trainCollection
    output_dir = resume.replace(trainCollection, testCollection)
    output_dir = output_dir.replace('/%s/' % options.cv_name, '/results/%s/' % trainCollection )
    result_pred_sents = os.path.join(output_dir, 'id.sent.score.txt')
    pred_error_matrix_file = os.path.join(output_dir, 'pred_errors_matrix.pth.tar')
    if checkToSkip(pred_error_matrix_file, opt.overwrite):
        sys.exit(0)
    makedirsforfile(pred_error_matrix_file)

    # data loader prepare
    caption_files = {'test': os.path.join(rootpath, testCollection, 'TextData', '%s.caption.txt'%testCollection)}
    img_feat_path = os.path.join(rootpath, testCollection, 'FeatureData', options.visual_feature)
    visual_feats = {'test': BigFile(img_feat_path)}
    assert options.visual_feat_dim == visual_feats['test'].ndims
    video2frames = {'test': read_dict(os.path.join(rootpath, testCollection, 'FeatureData', options.visual_feature, 'video2frames.txt'))}

    # set bow vocabulary and encoding
    bow_vocab_file = os.path.join(rootpath, options.trainCollection, 'TextData', 'vocabulary', 'bow', options.vocab+'.pkl')
    bow_vocab = pickle.load(open(bow_vocab_file, 'rb'))
    bow2vec = get_text_encoder('bow')(bow_vocab)
    options.bow_vocab_size = len(bow_vocab)

    # set rnn vocabulary 
    rnn_vocab_file = os.path.join(rootpath, options.trainCollection, 'TextData', 'vocabulary', 'rnn', options.vocab+'.pkl')
    rnn_vocab = pickle.load(open(rnn_vocab_file, 'rb'))
    options.vocab_size = len(rnn_vocab)

    # Construct the model
    model = get_model(options.model)(options)
    model.load_state_dict(checkpoint['model'])
    model.Eiters = checkpoint['Eiters']
    model.val_start()
    
    if testCollection.startswith('msvd'):# or testCollection.startswith('msrvtt'):
        # set data loader
        video_ids_list = data.read_video_ids(caption_files['test'])
        vid_data_loader = data.get_vis_data_loader(visual_feats['test'], opt.batch_size, opt.workers, video2frames['test'], video_ids=video_ids_list)
        text_data_loader = data.get_txt_data_loader(caption_files['test'], rnn_vocab, bow2vec, opt.batch_size, opt.workers)
        # mapping
        video_embs, video_ids = evaluation.encode_text_or_vid(model.embed_vis, vid_data_loader)
        cap_embs, caption_ids = evaluation.encode_text_or_vid(model.embed_txt, text_data_loader)
    else:
        # set data loader
        data_loader = data.get_test_data_loaders(
            caption_files, visual_feats, rnn_vocab, bow2vec, opt.batch_size, opt.workers, opt.n_caption, video2frames=video2frames)
        # mapping
        video_embs, cap_embs, video_ids, caption_ids = evaluation.encode_data(model, data_loader['test'], opt.log_step, logging.info)
        # remove duplicate videos
        idx = range(0, video_embs.shape[0], n_caption)
        video_embs = video_embs[idx,:]
        video_ids = video_ids[::opt.n_caption]

    c2i_all_errors = evaluation.cal_error(video_embs, cap_embs, options.measure)
    torch.save({'errors': c2i_all_errors, 'videos': video_ids, 'captions': caption_ids}, pred_error_matrix_file)    
    print("write into: %s" % pred_error_matrix_file)


    if testCollection.startswith('msvd'):# or testCollection.startswith('msrvtt'):
        # caption retrieval
        (r1, r5, r10, medr, meanr, i2t_map_score) = evaluation.i2t_varied(c2i_all_errors, caption_ids, video_ids)
        # video retrieval
        (r1i, r5i, r10i, medri, meanri, t2i_map_score) = evaluation.t2i_varied(c2i_all_errors, caption_ids, video_ids)
    else:
        # caption retrieval
        (r1i, r5i, r10i, medri, meanri) = evaluation.t2i(c2i_all_errors, n_caption=n_caption)
        t2i_map_score = evaluation.t2i_map(c2i_all_errors, n_caption=n_caption)

        # video retrieval
        (r1, r5, r10, medr, meanr) = evaluation.i2t(c2i_all_errors, n_caption=n_caption)
        i2t_map_score = evaluation.i2t_map(c2i_all_errors, n_caption=n_caption)

    print(" * Text to Video:")
    print(" * r_1_5_10, medr, meanr: {}".format([round(r1i, 1), round(r5i, 1), round(r10i, 1), round(medri, 1), round(meanri, 1)]))
    print(" * recall sum: {}".format(round(r1i+r5i+r10i, 1)))
    print(" * mAP: {}".format(round(t2i_map_score, 3)))
    print(" * "+'-'*10)

    # caption retrieval
    print(" * Video to text:")
    print(" * r_1_5_10, medr, meanr: {}".format([round(r1, 1), round(r5, 1), round(r10, 1), round(medr, 1), round(meanr, 1)]))
    print(" * recall sum: {}".format(round(r1+r5+r10, 1)))
    print(" * mAP: {}".format(round(i2t_map_score, 3)))
    print(" * "+'-'*10)



if __name__ == '__main__':
    main()
