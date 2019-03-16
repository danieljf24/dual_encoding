rootpath=@@@rootpath@@@
testCollection=@@@testCollection@@@
logger_name=@@@logger_name@@@
n_caption=@@@n_caption@@@
overwrite=@@@overwrite@@@

gpu=0

CUDA_VISIBLE_DEVICES=$gpu python tester.py $testCollection --rootpath $rootpath --overwrite $overwrite --logger_name $logger_name --n_caption $n_caption
