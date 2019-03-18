import os
import sys
import numpy as np
from basic.constant import ROOT_PATH
from basic.common import checkToSkip, makedirsforfile
from basic.bigfile import BigFile


def read_dict(filepath):
    f = open(filepath,'r')
    a = f.read()
    dict_data = eval(a)
    f.close()
    return dict_data


def write_dict(filepath, dict_data):
    f = open(filepath,'w')
    f.write(str(dict_data))
    f.close()


def process(opt):

    rootpath = opt.rootpath
    collection = opt.collection
    feature = opt.feature
    overwrite = opt.overwrite

    feat_path = os.path.join(rootpath, collection, "FeatureData", feature)
    result_file = os.path.join(feat_path, "video2frames.txt")
    if checkToSkip(result_file, overwrite):
        sys.exit(0)
    makedirsforfile(result_file)

    feat_data = BigFile(feat_path)
    video2fmnos = {}
    int2str = {}
    for frame_id in feat_data.names:
        data = frame_id.strip().split("_")
        #print data
        video_id = '_'.join(data[:-1])
        fm_no = data[-1]
        video2fmnos.setdefault(video_id, []).append(int(fm_no))
        if int(fm_no) not in int2str:
            int2str[int(fm_no)] = fm_no

    video2frames = {}
    for video_id, fmnos in video2fmnos.iteritems():
        for fm_no in sorted(fmnos):
            video2frames.setdefault(video_id, []).append(video_id + "_" + int2str[fm_no])

    write_dict(result_file, video2frames)
    print "write out into: ", result_file



def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]

    from optparse import OptionParser
    parser = OptionParser(usage="""usage: %prog [options]""")
    parser.add_option("--rootpath", default=ROOT_PATH, type="string", help="rootpath (default: %s)" % ROOT_PATH)
    parser.add_option("--collection", default="", type="string", help="collection name")
    parser.add_option("--feature", default="", type="string", help="feature name")
    parser.add_option("--overwrite", default=0, type="int", help="overwrite existing file (default=0)")

    (options, args) = parser.parse_args(argv)
    return process(options)

if __name__ == "__main__":
    sys.exit(main())
