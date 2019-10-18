import os
import sys
import logging


logger = logging.getLogger(__file__)
logging.basicConfig(
    format="[%(asctime)s - %(filename)s:line %(lineno)s] %(message)s",
    datefmt='%d %b %H:%M:%S')
logger.setLevel(logging.INFO)


ROOT_PATH = os.path.join(os.environ['HOME'], 'VisualSearch')
COLLECTION = 'iacc.3'
EDITION = 'tv18'
TRAIN_TYPE = 'D' # used any other training data with any annotation
PCLASS = 'F' # fully automatic
PID = 'RUCMM' # short participant ID
PRIORITY = 1
TOPK = 1000
DESC = 'place holder'
XML_HEAD = '<!DOCTYPE videoAdhocSearchResults SYSTEM "https://www-nlpir.nist.gov/projects/tv2018/dtds/videoAdhocSearchResults.dtd">'
ETIME = 25 # the time in seconds from the time the system is presented with the topic until the searching on that topic ended


def read_topics(topics_file):
    lines = map(str.strip, open(topics_file).readlines())
    qry_list = []
    for line in lines:
        tnum, query = line.split(' ', 1)
        qry_list.append((tnum, query))
    return qry_list


def wrap_topic_result(tNum, elapsedTime, topicResult):
    new_res = ['<videoAdhocSearchTopicResult tNum="%s" elapsedTime="%g">' % (tNum, elapsedTime)]
    for i,shot_id in enumerate(topicResult):
        new_res.append('<item seqNum="%d" shotId="%s" />' % (i+1, shot_id))
    new_res.append('</videoAdhocSearchTopicResult>')
    return new_res


def process(options, collection, input_txt_file):
    rootpath = options.rootpath
    overwrite = options.overwrite
    trtype = options.trtype
    pclass = options.pclass
    pid = options.pid
    priority = options.priority
    edition = options.edition
    desc = options.desc
    etime = options.etime
    topk = options.topk

    output_xml_file = input_txt_file + '.xml'

    if os.path.exists(output_xml_file):
        if overwrite:
            logger.info('%s exists. Overwrite' % output_xml_file)
        else:
            logger.info('%s exists. Use "--overwrite 1" if you want to overwrite' % output_xml_file)
            return

    topics_file = os.path.join(rootpath, collection, 'TextData', '%s.avs.txt'%edition)
    shots_file = os.path.join(rootpath, collection, 'VideoSets', '%s.txt'%collection)

    topics = read_topics(topics_file)
    tnum_set = set([x[0] for x in topics])
    shot_set = set(map(str.strip, open(shots_file).readlines()))
    logger.info('%s -> %d testing topics, %d shots', edition, len(tnum_set), len(shot_set))

    data = map(str.strip, open(input_txt_file).readlines())
    assert(len(data) == len(tnum_set)), "number of topics does not match"

    xml_content = []
    for line in data:
        elems = line.split()
        tNum = elems[0]
        del elems[0]
        
        assert(len(elems)>= (2*topk)) # shall contain at least topk pairs

        logger.debug('processing testing topic %s', tNum)
        prev_score = 1e8
        topic_res = []
        for i in range(0, 2*topk, 2):
            shot_id = elems[i]
            score = float(elems[i+1])
            assert(shot_id in shot_set), "invalid shot id: %s" % shot_id
            assert(score < (prev_score+1e-8)), "shots have not been sorted"
            prev_score = score
            topic_res.append(shot_id)

        xml_content += wrap_topic_result(tNum, etime, topic_res)
        xml_content.append('') # add a new line

    xml_file = [XML_HEAD]
    xml_file.append('') # add a new line
    xml_file.append('<videoAdhocSearchResults>')
    xml_file.append('<videoAdhocSearchRunResult trType="%s" class="%s" pid="%s" priority="%s" desc="%s">' % (trtype, pclass, pid, priority, desc))
    xml_file += xml_content
    xml_file.append('') # add a new line
    xml_file.append('</videoAdhocSearchRunResult>')
    xml_file.append('</videoAdhocSearchResults>')

    if not os.path.exists(os.path.split(output_xml_file)[0]):
        os.makedirs(os.path.split(output_xml_file)[0])

    open(output_xml_file, 'w').write('\n'.join(xml_file))
    logger.info('%s -> %s' % (input_txt_file, output_xml_file))


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]

    from optparse import OptionParser
    parser = OptionParser(usage="""usage: %prog [options] collection input_txt_file""")
    parser.add_option('--rootpath', type=str, default=ROOT_PATH, help='path to datasets. (default: %s)'%ROOT_PATH)
    parser.add_option("--overwrite", default=0, type="int", help="overwrite existing file (default: 0)")
    parser.add_option("--trtype", default=TRAIN_TYPE, type="string", help="training type (default: %s)" % TRAIN_TYPE)
    parser.add_option("--edition", default=EDITION, type="string", help="trecvid edition (default: %s)" % EDITION)
    parser.add_option("--pclass", default=PCLASS, type="string", help="processing type (default: %s)" % PCLASS)
    parser.add_option("--pid", default=PID, type="string", help="participant ID (default: %s)" % PID)
    parser.add_option("--desc", default=DESC, type="string", help="description of this run (default: %s)" % DESC)
    parser.add_option("--etime", default=ETIME, type="float", help="elapsed time in seconds (default: %g)" % ETIME)
    parser.add_option("--topk", default=TOPK, type="int", help="number of returned shots per query (default: %d)" % TOPK)
    parser.add_option("--priority", default=PRIORITY, type="int", help="priority (default: %d)" % PRIORITY)
            
    (options, args) = parser.parse_args(argv)
    if len(args) < 1:
        parser.print_help()
        return 1

    return process(options, args[0], args[1])


if __name__ == "__main__":
    sys.exit(main())

