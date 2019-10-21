rootpath=$HOME/VisualSearch3
set_style=ImageSets
overwrite=0

featname=pyresnext-101_rbps13k,flatten0_output,os

collection=tgif-msrvtt10k
sub_collections=tgif@msrvtt10k


python util/combine_features.py $collection $featname ${sub_collections} ${set_style} \
    --overwrite $overwrite --rootpath $rootpath


feat_dir=$rootpath/$collection/FeatureData/$featname
feat_file=${feat_dir}/id.feature.txt

if [ -f ${feat_file} ]; then
    python util/txt2bin.py 2048 ${feat_file} 0 ${feat_dir} --overwrite 1
    rm ${feat_file}
fi
