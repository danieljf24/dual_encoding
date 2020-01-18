from __future__ import print_function
import pickle
import time
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
from basic.generic_utils import Progbar

def parse_args():
    # Hyper Parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('testCollection', type=str, help='test collection')
    parser.add_argument('--rootpath', type=str, default=ROOT_PATH, help='path to datasets. (default: %s)'%ROOT_PATH)
    parser.add_argument('--overwrite', type=int, default=0, choices=[0,1],  help='overwrite existed file. (default: 0)')
    parser.add_argument('--batch_size', default=128, type=int, help='Size of a training mini-batch.')
    parser.add_argument('--workers', default=5, type=int, help='Number of data loader workers.')
    parser.add_argument('--logger_name', default='runs', help='Path to save the model and Tensorboard log.')
    parser.add_argument('--checkpoint_name', default='model_best.pth.tar', type=str, help='name of checkpoint (default: model_best.pth.tar)')
    parser.add_argument('--query_sets', type=str, default='tv16.avs.txt',  help='test query sets,  tv16.avs.txt,tv17.avs.txt,tv18.avs.txt for TRECVID 16/17/18.')

    args = parser.parse_args()
    return args


def encode_data(encoder, data_loader, return_ids=True):
    """Encode all videos and captions loadable by `data_loader`
    """
    # numpy array to keep all the embeddings
    embeddings = None
    ids = ['']*len(data_loader.dataset)
    pbar = Progbar(len(data_loader.dataset))
    for i, (datas, idxs, data_ids) in enumerate(data_loader):

        # compute the embeddings
        emb = encoder(datas)

        # initialize the numpy arrays given the size of the embeddings
        if embeddings is None:
            embeddings = np.zeros((len(data_loader.dataset), emb.size(1)))

        # preserve the embeddings by copying from gpu and converting to numpy
        embeddings[idxs] = emb.data.cpu().numpy().copy()

        for j, idx in enumerate(idxs):
            ids[idx] = data_ids[j]

        del datas
        pbar.add(len(idxs))

    if return_ids == True:
        return embeddings, ids,
    else:
        return embeddings


def main():
    opt = parse_args()
    print(json.dumps(vars(opt), indent=2))

    rootpath = opt.rootpath
    testCollection = opt.testCollection
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
    model = get_model(options.model)(options)
    model.load_state_dict(checkpoint['model'])
    model.val_start()

    trainCollection = options.trainCollection
    valCollection = options.valCollection

    visual_feat_file = BigFile(os.path.join(rootpath, testCollection, 'FeatureData', options.visual_feature))
    assert options.visual_feat_dim == visual_feat_file.ndims
    video2frame = read_dict(os.path.join(rootpath, testCollection, 'FeatureData', options.visual_feature, 'video2frames.txt'))
    visual_loader = data.get_vis_data_loader(visual_feat_file, opt.batch_size, opt.workers, video2frame)
    vis_embs = None

    # set bow vocabulary and encoding
    bow_vocab_file = os.path.join(rootpath, options.trainCollection, 'TextData', 'vocabulary', 'bow', options.vocab+'.pkl')
    bow_vocab = pickle.load(open(bow_vocab_file, 'rb'))
    bow2vec = get_text_encoder('bow')(bow_vocab)
    options.bow_vocab_size = len(bow_vocab)

    # set rnn vocabulary 
    rnn_vocab_file = os.path.join(rootpath, options.trainCollection, 'TextData', 'vocabulary', 'rnn', options.vocab+'.pkl')
    rnn_vocab = pickle.load(open(rnn_vocab_file, 'rb'))
    options.vocab_size = len(rnn_vocab)

    output_dir = resume.replace(trainCollection, testCollection)
    for query_set in opt.query_sets.strip().split(','):
        output_dir_tmp = output_dir.replace(valCollection, '%s/%s/%s' % (query_set, trainCollection, valCollection))
        output_dir_tmp = output_dir_tmp.replace('/%s/' % options.cv_name, '/results/')
        pred_result_file = os.path.join(output_dir_tmp, 'id.sent.score.txt')
        print(pred_result_file)
        if checkToSkip(pred_result_file, opt.overwrite):
            continue
        try:
            makedirsforfile(pred_result_file)
        except Exception as e:
            print(e)

        # data loader prepare
        query_file = os.path.join(rootpath, testCollection, 'TextData', query_set)

        # set data loader
        query_loader = data.get_txt_data_loader(query_file, rnn_vocab, bow2vec, opt.batch_size, opt.workers)

        if vis_embs is None:
            start = time.time()
            vis_embs, vis_ids = encode_data(model.embed_vis, visual_loader)
            print("encode image time: %.3f s" % (time.time()-start))

        start = time.time()
        query_embs, query_ids = encode_data(model.embed_txt, query_loader)
        print("encode text time: %.3f s" % (time.time()-start))

        start = time.time()
        t2i_matrix = query_embs.dot(vis_embs.T)
        inds = np.argsort(t2i_matrix, axis=1)
        print("compute similarity time: %.3f s" % (time.time()-start))

        with open(pred_result_file, 'w') as fout:
            for index in range(inds.shape[0]):
                ind = inds[index][::-1]
                fout.write(query_ids[index]+' '+' '.join([vis_ids[i]+' %s'%t2i_matrix[index][i]
                    for i in ind])+'\n')

        if testCollection == 'iacc.3':
            templete = ''.join(open( 'tv-avs-eval/TEMPLATE_do_eval.sh').readlines())
            striptStr = templete.replace('@@@rootpath@@@', rootpath)
            striptStr = striptStr.replace('@@@testCollection@@@', testCollection)
            striptStr = striptStr.replace('@@@topic_set@@@', query_set.split('.')[0])
            striptStr = striptStr.replace('@@@overwrite@@@', str(opt.overwrite))
            striptStr = striptStr.replace('@@@score_file@@@', pred_result_file)

            runfile = 'do_eval_%s.sh' % testCollection
            open(os.path.join('tv-avs-eval', runfile), 'w').write(striptStr + '\n')
            os.system('cd tv-avs-eval; chmod +x %s; bash %s; cd -' % (runfile, runfile))
            


if __name__ == '__main__':
    main()
