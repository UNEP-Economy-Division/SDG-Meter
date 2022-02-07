import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np  # linear algebra
from IPython import get_ipython
import os
import collections
import pandas as pd
from datetime import datetime
import random
import markovify
import numpy as np
import nltk
import re
from nltk.corpus import stopwords
import time
import uuid
from pathlib import Path
import numpy as np
import pandas as pd


def read_file(file_object):
    # The file_content is the.txt you upload
    # File have already been opened during the transmittion, no need to open it again.
    # Only surport .txt
    # You can read it directly
    print('file')
    # print(file_object.read())
    st = str(file_object.read())
    THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.join(THIS_FOLDER, 'text.txt')
    #
    #
    f1 = open(input_file, 'w', encoding="utf-8")
    #f1 = open('E:/Codebase/archive/text.txt', 'wt', encoding="utf-8")
    f1.write(st)
    f1.close()
    time.sleep(5)

    os.system("python Bert.py")

    my_file = os.path.join(THIS_FOLDER, 'output.txt')
    lines = open(my_file)
    array = lines.readlines()
    array2 = []
    for i in array:
        i = i.strip('\n')
        array2.append(i)

    print(array2)
    print(array2[0])
    lines.close()

    data = [
        {
            "date": "SDG1",
            "value": float(array2[0])
        }, {
            "date": "SDG2",

            "value": float(array2[1])
        }, {
            "date": "SDG3",

            "value": float(array2[2])
        }, {
            "date": "SDG4",

            "value": float(array2[3])
        }, {
            "date": "SDG5",

            "value": float(array2[4])
        }, {
            "date": "SDG6",

            "value": float(array2[5])
        }, {
            "date": "SDG7",

            "value": float(array2[6])
        }, {
            "date": "SDG8",

            "value": float(array2[7])
        }, {
            "date": "SDG9",

            "value": float(array2[8])
        }, {
            "date": "SDG10",

            "value": float(array2[9])
        }, {
            "date": "SDG11",

            "value": float(array2[10])
        }, {
            "date": "SDG12",

            "value": float(array2[11])
        }, {
            "date": "SDG13",

            "value": float(array2[12])
        }, {
            "date": "SDG14",

            "value": float(array2[13])
        }, {
            "date": "SDG15",

            "value": float(array2[14])
        }, {
            "date": "SDG16",

            "value": float(array2[15])
        }, {
            "date": "SDG17",

            "value": float(array2[16])
        }
    ]

    #

    return data
