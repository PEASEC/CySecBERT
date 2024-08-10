# Parameters
# =============================================================================
FILE_PATH = '/work/projects/project01762/CyBERT/cybert/input/NER/tagged_all.csv'
SPLIT_SEED =  42
SPLIT_RATIO = 80
CREATE_DEV_SET = False

NER_TAG = 'AC'

CSV_SEPARATOR = 'Space(s)'
# =============================================================================


# Code

import csv
import json
import os
from pathlib import Path
from numpy.random import randn
import pandas as pd
import numpy as np
import sklearn
from numpy.random import RandomState
import numpy as np
from sklearn.model_selection import train_test_split

O_TAG = 'O'


def main():
    df = read_csv()

    json_list = create_json_per_cve(df)

    split_dict = split_data(json_list)

    data_to_json(split_dict)


def read_csv():
    global csv_filename, req_tag
    csv_filename = FILE_PATH
    req_tag = NER_TAG
    csv_sep = '\s+' if CSV_SEPARATOR.startswith('Space') else CSV_SEPARATOR
              #'\t' if CSV_SEPARATOR == 'tab' else \
              

    if(csv_filename[-3:] != 'csv'):
        raise ValueError("Not a CSV file!")

    df = pd.read_csv(
    csv_filename,
    sep=csv_sep,
    dtype=str,
    header=None,
    skip_blank_lines=True,
    na_filter=True
    )

    return df


def create_json_per_cve(df):
    json_list = []

    groupedby_cve = df.groupby(by=2).groups
    for cve in groupedby_cve.values():

        entities = []
        ner = []
        for cve_loc in cve:
          # workaround to filter out NaN values,
          # since pandas.DataFrame.dropna seems to have a bug on Google Colab
            if (df.loc[[cve_loc]][0].notna().all() and
                df.loc[[cve_loc]][1].notna().all()):
                entities.append(df.loc[[cve_loc]][0].values[0])
                ner.append(tags_to_tag(df.loc[[cve_loc]][1].values[0]))

        json_list.append({
            'words' : entities,
            'ner': ner
             }
            )
    return json_list
    

def split_data(json_list):
    ret_dict = dict()
    seed = SPLIT_SEED
    ratio = int(SPLIT_RATIO) / 100

    train, dev_test = train_test_split(json_list, train_size=ratio, random_state=seed)
    ret_dict = {'train': train,
                'test': dev_test,
                'dev': None}
    if CREATE_DEV_SET:
        dev, test = train_test_split(json_list, train_size=0.5, random_state=seed)
        ret_dict['dev'] = dev
        ret_dict['test'] = test

    return ret_dict


def data_to_json(split_dict):

    json_filename_prefix = Path(csv_filename[:-4] + '_' + req_tag)
    json_filename_train = json_filename_prefix / 'train.json'
    json_filename_test = json_filename_prefix / 'test.json'
    
    os.makedirs(os.path.dirname(json_filename_train), exist_ok=True)
    with open(json_filename_train, 'w') as json_file:
        for e in split_dict['train']:
            json.dump(e, json_file)
            json_file.write('\n')
        print("Train dataset saved in : " + str(json_filename_train))

    with open(json_filename_test, 'w') as json_file:
        for e in split_dict['test']:
            json.dump(e, json_file)
            json_file.write('\n')
        print("Test dataset saved in : " + str(json_filename_test))

    if CREATE_DEV_SET:
        json_filename_dev = json_filename_prefix / 'dev.json'
        with open(json_filename_dev, 'w') as json_file:
            for e in split_dict['dev']:
                json.dump(e, json_file)
                json_file.write('\n')
        print("Dev dataset saved in : " + str(json_filename_dev))


def tags_to_tag(tags):
    return req_tag if req_tag in str(tags) else O_TAG


        
main()
