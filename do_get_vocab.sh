collection=$1

threshold=5
overwrite=0
for text_style in bow rnn
do
python util/vocab.py $collection --threshold $threshold --text_style $text_style --overwrite $overwrite 
done