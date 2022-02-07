# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# ### 1. Load the packages

# %%
from pandas.core.frame import DataFrame
from fast_bert.prediction import BertClassificationPredictor
import logging
import time
from transformers import BertTokenizer
from pathlib import Path
import torch
from box import Box
import pandas as pd
import collections
import os
from tqdm import tqdm, trange
import sys
import random
import numpy as np
import apex
from sklearn.model_selection import train_test_split
import datetime
from fast_bert.modeling import BertForMultiLabelSequenceClassification
from fast_bert.data_cls import BertDataBunch, InputExample, InputFeatures, MultiLabelTextProcessor, convert_examples_to_features
from fast_bert.learner_cls import BertLearner
from fast_bert.metrics import accuracy_multilabel, accuracy_thresh, fbeta, roc_auc


def get_SDG(text):
    # ### 2. Display bar of the time required for code training.

    from fast_bert.prediction import BertClassificationPredictor

    predictor = BertClassificationPredictor(
                    model_path='archive/finetuned_models/model_out',
                    label_path='archive/labels/',
                    multi_label=True,
                    model_type='bert',
                    do_lower_case=False)

    # Single prediction

    # ### 10. Read the text from text.txt


#    lines1 = open(THIS_FOLDER+'/text.txt', encoding="utf-8")
    #lines1 = open('E:/Codebase/archive/text.txt', encoding="utf-8")
    texts123 = text
    texts321 = [texts123]


    # ### 11. make a prediction

    single_prediction = predictor.predict(texts123)
    #single_prediction = predictor.predict("Economic and require the production of goods and services that improve the quality of life. Sustainable and require minimizing the resources and toxic materials used, and the waste and pollutants generated, throughout the entire production and consumption process.")
    # lines1.close()

    # %% [markdown]
    # ### 11. Change the output format and save the result to output.txt

    # %%
    data = DataFrame(single_prediction)
    data.columns = ['SDG', 'number']
    array2 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    for index, row in data.iterrows():
        if row['SDG'] == 'SDG1':
            array2[0] = (float(row['number']))
        elif row['SDG'] == 'SDG2':

            array2[1] = (float(row['number']))
        elif row['SDG'] == 'SDG3':

            array2[2] = (float(row['number']))

        elif row['SDG'] == 'SDG4':

            array2[3] = (float(row['number']))
        elif row['SDG'] == 'SDG5':

            array2[4] = (float(row['number']))
        elif row['SDG'] == 'SDG6':

            array2[5] = (float(row['number']))
        elif row['SDG'] == 'SDG7':

            array2[6] = (float(row['number']))
        elif row['SDG'] == 'SDG8':

            array2[7] = (float(row['number']))
        elif row['SDG'] == 'SDG9':

            array2[8] = (float(row['number']))
        elif row['SDG'] == 'SDG10':

            array2[9] = (float(row['number']))
        elif row['SDG'] == 'SDG11':

            array2[10] = (float(row['number']))
        elif row['SDG'] == 'SDG12':

            array2[11] = (float(row['number']))
        elif row['SDG'] == 'SDG13':

            array2[12] = (float(row['number']))
        elif row['SDG'] == 'SDG14':

            array2[13] = (float(row['number']))
        elif row['SDG'] == 'SDG15':

            array2[14] = (float(row['number']))
        elif row['SDG'] == 'SDG16':

            array2[15] = (float(row['number']))
        else:
            array2[16] = (float(row['number']))

    
    
    # dets = np.array([[array2[0]], [array2[1]], [array2[2]], [array2[3]], [array2[4]], [array2[5]], [array2[6]], [array2[7]], [
                    # array2[8]], [array2[9]], [array2[10]], [array2[11]], [array2[12]], [array2[13]], [array2[14]], [array2[15]], [array2[16]]])


    return data.to_json(orient="split")

# output_file = os.path.join(THIS_FOLDER, 'output.txt')
# f = open(output_file, 'wt')
# np.savetxt(f, dets, fmt='%f', delimiter=',')
# f.close()


if __name__ == '__main__':
    get_SDG("Hello")