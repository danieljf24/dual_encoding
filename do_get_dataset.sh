ROOTPATH=$HOME/VisualSearch
mkdir -p $ROOTPATH && cd $ROOTPATH

# download and extract dataset
wget http://lixirong.net/data/cvpr2019/msrvtt10k-text-and-resnet-152-img1k.tar.gz
tar zxf msrvtt10k-text-and-resnet-152-img1k.tar.gz

# download and extract pre-trained word2vec
wget http://lixirong.net/data/w2vv-tmm2018/word2vec.tar.gz
tar zxf word2vec.tar.gz