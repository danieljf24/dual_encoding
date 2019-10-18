rootpath=@@@rootpath@@@
testCollection=@@@testCollection@@@
topic_set=@@@topic_set@@@
overwrite=@@@overwrite@@@

score_file=@@@score_file@@@

bash do_txt2xml.sh $testCollection $score_file $topic_set $overwrite
python trec_eval.py ${score_file}.xml --rootpath $rootpath --collection $testCollection --edition $topic_set --overwrite $overwrite

