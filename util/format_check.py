import os
import sys
from basic.constant import ROOT_PATH

def process(opt):

    rootpath = opt.rootpath
    collection = opt.collection
    feature = opt.feature
    flag=0

    feat_dir = os.path.join(rootpath, collection, 'FeatureData', feature)
    if not os.path.exists(os.path.join(feat_dir, 'feature.bin')):
        print "file %s is not exits" % os.path.join(feat_dir, 'feature.bin')
        flag=1
    if not os.path.exists(os.path.join(feat_dir, 'id.txt')):
        print "file %s is not exits" % os.path.join(feat_dir, 'id.txt')
        flag=1
    if not os.path.exists(os.path.join(feat_dir, 'shape.txt')):
        print "file %s is not exits" % os.path.join(feat_dir, 'shape.txt')
        flag=1


    textfile = feat_dir = os.path.join(rootpath, collection, 'TextData', '%s.caption.txt' % collection)
    if not os.path.exists(textfile):
        print "file %s is not exits" % textfile
        flag=1

    if flag == 0:
        print "%s format check pass!" % collection


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]

    from optparse import OptionParser
    parser = OptionParser(usage="""usage: %prog [options]""")
    parser.add_option("--rootpath", default=ROOT_PATH, type="string", help="rootpath (default: %s)" % ROOT_PATH)
    parser.add_option("--collection", default="", type="string", help="collection name")
    parser.add_option("--feature", default="", type="string", help="feature name")

    (options, args) = parser.parse_args(argv)
    return process(options)

if __name__ == "__main__":
    sys.exit(main())
