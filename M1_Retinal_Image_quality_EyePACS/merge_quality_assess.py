import numpy as np
import pandas as pd
import shutil
import os

AUTOMORPH_DATA = os.getenv('AUTOMORPH_DATA','..')

def merge_quality_assessment(df=f'{AUTOMORPH_DATA}/Results/M1/results_ensemble.csv', save:bool=False):

    if isinstance(df, str):
        df = pd.read_csv(df)

    if save:
        if not os.path.exists(f'{AUTOMORPH_DATA}/Results/M1/Good_quality/'):
            os.makedirs(f'{AUTOMORPH_DATA}/Results/M1/Good_quality/')
        if not os.path.exists(f'{AUTOMORPH_DATA}/Results/M1/Bad_quality/'):
            os.makedirs(f'{AUTOMORPH_DATA}/Results/M1/Bad_quality/')
    else:
        img_quality = {
            "good": [],
            "bad": []
        }

    Eyepacs_pre = df['Prediction']
    Eyepacs_bad_mean = df['softmax_bad']
    Eyepacs_usable_sd = df['usable_sd']
    name_list = df['Name']

    Eye_good = 0
    Eye_bad = 0

    for i in range(len(name_list)):
        if Eyepacs_pre[i]==0:
            Eye_good+=1
            if save: 
                shutil.copy(name_list[i], f'{AUTOMORPH_DATA}/Results/M1/Good_quality/')
            else:
                img_quality["good"].append(name_list[i]) # TODO: Do we want to return the name or the image

        elif (Eyepacs_pre[i]==1) and (Eyepacs_bad_mean[i]<0.25):
            Eye_good+=1
            if save:
                shutil.copy(name_list[i], f'{AUTOMORPH_DATA}/Results/M1/Good_quality/')
            else:
                img_quality["good"].append(name_list[i])
        else:
            Eye_bad+=1
            if save:
                shutil.copy(name_list[i], f'{AUTOMORPH_DATA}/Results/M1/Bad_quality/')
            else:
                img_quality["bad"].append(name_list[i])

    print('Gradable cases by EyePACS_QA is {} '.format(Eye_good))
    print('Ungradable cases by EyePACS_QA is {} '.format(Eye_bad))
    if not save: return img_quality