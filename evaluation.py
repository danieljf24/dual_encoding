from __future__ import print_function
import os
import pickle

import numpy
import time
import numpy as np
from scipy.spatial import distance
import torch
from torch.autograd import Variable
from basic.metric import getScorer
from basic.util import AverageMeter, LogCollector

def l2norm(X):
    """L2-normalize columns of X
    """
    norm = np.linalg.norm(X, axis=1, keepdims=True)
    return 1.0 * X / norm

def cal_error(videos, captions, measure='cosine'):
    if measure == 'cosine':
        captions = l2norm(captions)
        videos = l2norm(videos)
        errors = -1*numpy.dot(captions, videos.T)
    elif measure == 'euclidean':
        errors = distance.cdist(captions, videos, 'euclidean')
    return errors




def encode_data(model, data_loader, log_step=10, logging=print, return_ids=True):
    """Encode all videos and captions loadable by `data_loader`
    """
    batch_time = AverageMeter()
    val_logger = LogCollector()

    # switch to evaluate mode
    model.val_start()

    end = time.time()

    # numpy array to keep all the embeddings
    video_embs = None
    cap_embs = None
    video_ids = ['']*len(data_loader.dataset)
    caption_ids = ['']*len(data_loader.dataset)
    for i, (videos, captions, idxs, cap_ids, vid_ids) in enumerate(data_loader):
        # make sure val logger is used
        model.logger = val_logger

        # compute the embeddings
        vid_emb, cap_emb = model.forward_emb(videos, captions, True)

        # initialize the numpy arrays given the size of the embeddings
        if video_embs is None:
            video_embs = np.zeros((len(data_loader.dataset), vid_emb.size(1)))
            cap_embs = np.zeros((len(data_loader.dataset), cap_emb.size(1)))

        # preserve the embeddings by copying from gpu and converting to numpy
        video_embs[idxs] = vid_emb.data.cpu().numpy().copy()
        cap_embs[idxs] = cap_emb.data.cpu().numpy().copy()

        for j, idx in enumerate(idxs):
            caption_ids[idx] = cap_ids[j]
            video_ids[idx] = vid_ids[j]

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % log_step == 0:
            logging('Test: [{0:2d}/{1:2d}]\t'
                    '{e_log}\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    .format(
                        i, len(data_loader), batch_time=batch_time,
                        e_log=str(model.logger)))
        del videos, captions

    if return_ids == True:
        return video_embs, cap_embs, video_ids, caption_ids
    else:
        return video_embs, cap_embs


# recall@k, Med r, Mean r for Text-to-Video Retrieval
def t2i(c2i, vis_details=False, n_caption=5):
    """
    Text->Videos (Text-to-Video Retrieval)
    c2i: (5N, N) matrix of caption to video errors
    vis_details: if true, return a dictionary for ROC visualization purposes
    """
    # print("errors matrix shape: ", c2i.shape)
    assert c2i.shape[0] / c2i.shape[1] == n_caption, c2i.shape
    ranks = np.zeros(c2i.shape[0])

    for i in range(len(ranks)):
        d_i = c2i[i]
        inds = np.argsort(d_i)

        rank = np.where(inds == i/n_caption)[0][0]
        ranks[i] = rank

    # Compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1

    return map(float, [r1, r5, r10, medr, meanr])



# recall@k, Med r, Mean r for Video-to-Text Retrieval
def i2t(c2i, n_caption=5):
    """
    Videos->Text (Video-to-Text Retrieval)
    c2i: (5N, N) matrix of caption to video errors
    """
    #remove duplicate videos
    # print("errors matrix shape: ", c2i.shape)
    assert c2i.shape[0] / c2i.shape[1] == n_caption, c2i.shape
    ranks = np.zeros(c2i.shape[1])

    for i in range(len(ranks)):
        d_i = c2i[:, i]
        inds = np.argsort(d_i)

        rank = np.where(inds/n_caption == i)[0][0]
        ranks[i] = rank

    # Compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1
    return map(float, [r1, r5, r10, medr, meanr])


# mAP for Text-to-Video Retrieval
def t2i_map(c2i, n_caption=5):
    """
    Text->Videos (Text-to-Video Retrieval)
    c2i: (5N, N) matrix of caption to video errors
    """
    # print("errors matrix shape: ", c2i.shape)
    assert c2i.shape[0] / c2i.shape[1] == n_caption, c2i.shape

    scorer = getScorer('AP')
    perf_list = []
    for i in range(c2i.shape[0]):
        d_i = c2i[i, :]
        labels = [0]*len(d_i)
        labels[i/n_caption] = 1

        sorted_labels = [labels[x] for x in np.argsort(d_i)]
        current_score = scorer.score(sorted_labels)
        perf_list.append(current_score)

    return np.mean(perf_list)


# mAP for Video-to-Text Retrieval
def i2t_map(c2i, n_caption=5):
    """
    Videos->Text (Video-to-Text Retrieval)
    c2i: (5N, N) matrix of caption to video errors
    """
    # print("errors matrix shape: ", c2i.shape)
    assert c2i.shape[0] / c2i.shape[1] == n_caption, c2i.shape

    scorer = getScorer('AP')
    perf_list = []
    for i in range(c2i.shape[1]):
        d_i = c2i[:, i]
        labels = [0]*len(d_i)
        labels[i*n_caption:(i+1)*n_caption] = [1]*n_caption

        sorted_labels = [labels[x] for x in np.argsort(d_i)]
        current_score = scorer.score(sorted_labels)
        perf_list.append(current_score)

    return np.mean(perf_list)


def t2i_inv_rank(c2i, n_caption=1):
    """
    Text->Videos (Text-to-Video Retrieval)
    c2i: (5N, N) matrix of caption to video errors
    n_caption: number of captions of each image/video
    """
    assert c2i.shape[0] / c2i.shape[1] == n_caption, c2i.shape
    inv_ranks = np.zeros(c2i.shape[0])

    for i in range(len(inv_ranks)):
        d_i = c2i[i,:]
        inds = np.argsort(d_i)

        rank = np.where(inds == i/n_caption)[0]
        inv_ranks[i] = sum(1.0 / (rank +1 ))

    return np.mean(inv_ranks)



def i2t_inv_rank(c2i, n_caption=1):
    """
    Videos->Text (Video-to-Text Retrieval)
    c2i: (5N, N) matrix of caption to video errors
    n_caption: number of captions of each image/video
    """
    assert c2i.shape[0] / c2i.shape[1] == n_caption, c2i.shape
    inv_ranks = np.zeros(c2i.shape[1])

    for i in range(len(inv_ranks)):
        d_i = c2i[:, i]
        inds = np.argsort(d_i)

        rank = np.where(inds/n_caption == i)[0]
        inv_ranks[i] = sum(1.0 / (rank +1 ))

    return np.mean(inv_ranks)




def i2t_inv_rank_multi(c2i, n_caption=2):
    """
    Text->videos (Image Search)
    c2i: (5N, N) matrix of caption to image errors
    n_caption: number of captions of each image/video
    """
    # print("errors matrix shape: ", c2i.shape)
    assert c2i.shape[0] / c2i.shape[1] == n_caption, c2i.shape
    inv_ranks = np.zeros(c2i.shape[1])

    result = []
    for i in range(n_caption):
        idx = range(i, c2i.shape[0], n_caption)
        sub_c2i = c2i[idx, :]
        score = i2t_inv_rank(sub_c2i, n_caption=1)
        result.append(score)
    return result




# the number of captions are various across videos
def eval_various(label_matrix):
    ranks = np.zeros(label_matrix.shape[0])
    aps = np.zeros(label_matrix.shape[0])

    for index in range(len(ranks)):
        rank = np.where(label_matrix[index]==1)[0] + 1
        ranks[index] = rank[0]

        aps[index] = np.mean([(i+1.)/rank[i] for i in range(len(rank))])

    r1, r5, r10 = [100.0*np.mean([x <= k for x in ranks]) for k in [1, 5, 10]]
    medr = np.floor(np.median(ranks))
    meanr = ranks.mean()
    # mir = (1.0/ranks).mean()
    mAP = aps.mean()

    return (r1, r5, r10, medr, meanr, mAP)


def i2t_various(c2i_all_errors, caption_ids, video_ids):
    inds = np.argsort(-c2i_all_errors, axis=1)
    label_matrix = np.zeros(inds.shape)
    for index in range(inds.shape[0]):
        ind = inds[index][::-1]
        label_matrix[index][np.where(np.array(video_ids)[ind]==caption_ids[index].split('#')[0])[0]]=1
    return eval_various(label_matrix)


def t2i_various(c2i_all_errors, caption_ids, video_ids):
    inds = np.argsort(-c2i_all_errors.T, axis=1)
    label_matrix = np.zeros(inds.shape)
    caption_ids = [txt_id.split('#')[0] for txt_id in caption_ids]
    for index in range(inds.shape[0]):
        ind = inds[index][::-1]
        label_matrix[index][np.where(np.array(caption_ids)[ind]==video_ids[index])[0]]=1
    return eval_various(label_matrix)