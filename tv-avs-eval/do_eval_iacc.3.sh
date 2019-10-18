rootpath=/data/home/xcx/VisualSearch2
testCollection=iacc.3
topic_set=tv18
overwrite=1

score_file=/data/home/xcx/VisualSearch/iacc.3/results/tv18.avs.txt/tgif-msrvtt10k/tv2016train/dual_encoding_concate_full_dp_0.2_measure_cosine/vocab_word_vocab_5_word_dim_500_text_rnn_size_512_text_norm_True_kernel_sizes_2-3-4_num_512/visual_feature_pyresnext-101_rbps13k,flatten0_output,os_visual_rnn_size_1024_visual_norm_True_kernel_sizes_2-3-4-5_num_512/mapping_text_0-2048_img_0-2048/loss_func_mrl_margin_0.2_direction_all_max_violation_True_cost_style_sum/optimizer_adam_lr_0.0001_decay_0.99_grad_clip_2.0_val_metric_recall/runs_0/model_best.pth.tar/id.sent.score.txt

bash do_txt2xml.sh $testCollection $score_file $topic_set $overwrite
python trec_eval.py ${score_file}.xml --rootpath $rootpath --collection $testCollection --edition $topic_set --overwrite $overwrite


