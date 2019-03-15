from __future__ import print_function

import os
import time

# def checkFileExist(filename, overwrite, logger=print):
#     if os.path.exists(filename):
#         if overwrite:
#             logger('%s exists. overwrite'%filename)
#             return 0
#         else:
#             logger('%s exists. skip'%filename)
#             return 1
#     return 0


def makedirsforfile(filename):
    try:
        os.makedirs(os.path.split(filename)[0])
    except:
        pass



def checkToSkip(filename, overwrite):
    if os.path.exists(filename):
        print ("%s exists." % filename),
        if overwrite:
            print ("overwrite")
            return 0
        else:
            print ("skip")
            return 1
    return 0 


def printMessage(message_type, trace, message):
    print ('%s %s [%s] %s' % (time.strftime('%d/%m/%Y %H:%M:%S'), message_type, trace, message))

def printStatus(trace, message):
    printMessage('INFO', trace, message)

def printError(trace, message):
    printMessage('ERROR', trace, message)