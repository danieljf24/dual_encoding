trainCollection=$1
valCollection=$2
testCollection=$3
rootpath=$4
visual_feature=$5
n_caption=$6
concate=$7
overwrite=0

# Generate a vocabulary on the training set
./do_get_vocab.sh $trainCollection

# Generate video frame info
./do_get_frameInfo.sh $trainCollection $visual_feature

# training
gpu=0
CUDA_VISIBLE_DEVICES=$gpu python trainer.py $trainCollection $valCollection $testCollection --overwrite $overwrite \
                                            --max_violation --text_norm --visual_norm --visual_feature $visual_feature --n_caption $n_caption  --concate $concate

# evaluation (Notice that a script file do_test_${testCollection}.sh will be automatically generated when the training process ends.)
./do_test_dual_encoding_${testCollection}.sh