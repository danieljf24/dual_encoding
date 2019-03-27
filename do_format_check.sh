trainCollection=$1
valCollection=$2
testCollection=$3
rootpath=$4
feature=$5

for collection in $trainCollection $valCollection $testCollection
do
python util/format_check.py --collection $collection --feature $feature --rootpath $rootpath
done