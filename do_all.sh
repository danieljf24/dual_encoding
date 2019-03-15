trainCollection=$1
valCollection=$2
testCollection=$3
overwrite=0

# Generate a vocabulary on the training set
./do_get_vocab.sh $trainCollection

# training
gpu=1
CUDA_VISIBLE_DEVICES=$gpu python trainer.py $trainCollection $valCollection $testCollection  --overwrite $overwrite \
                                            --max_violation --text_norm --visual_norm 
                                            
# evaluation (Notice that a script file do_test_${testCollection}.sh will be automatically generated when the training process ends.)
./do_test_dual_encoding_${testCollection}.sh