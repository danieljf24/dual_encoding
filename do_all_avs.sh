trainCollection=tgif-msrvtt10k
valCollection=tv2016train
testCollection=iacc.3
rootpath=$HOME/VisualSearch
visual_feature=pyresnext-101_rbps13k,flatten0_output,os
n_caption=2

overwrite=0

direction=all
cost_style=sum

postfix=runs_0

# Generate a vocabulary on the training set
#./do_get_vocab.sh $trainCollection

# Generate video frame info
#./do_get_frameInfo.sh $trainCollection $visual_feature

# training
gpu=2
CUDA_VISIBLE_DEVICES=$gpu python trainer.py $trainCollection $valCollection $testCollection --overwrite $overwrite \
                                            --max_violation --text_norm --visual_norm --visual_feature $visual_feature --n_caption $n_caption --direction $direction --postfix $postfix --cost_style $cost_style

# evaluation (Notice that a script file do_test_${testCollection}.sh will be automatically generated when the training process ends.)
./do_test_dual_encoding_${testCollection}.sh
