rootpath=$HOME/VisualSearch

etime=1.0

if [ "$#" -lt 3 ]; then
    echo "Usage: $0 testCollection score_file edition"
    exit
fi

test_collection=$1
score_file=$2
edition=$3
overwrite=0

if [ $# -gt 3 ]; then
    overwrite=$4
fi
python txt2xml.py $test_collection $score_file --edition $edition --priority 1 --etime $etime --desc "This run uses the top secret x-component" --rootpath $rootpath --overwrite $overwrite


