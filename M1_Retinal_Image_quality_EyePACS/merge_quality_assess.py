import numpy as np
import pandas as pd
import shutil
import os

def merge_quality_assessment(df='results_ensemble.csv', save_path=None):

    if isinstance(df, str):
        df = pd.read_csv(df)

    if save_path:
        if not os.path.exists(os.path.join(save_path, 'Good_quality/')):
            os.makedirs(os.path.join(save_path, 'Good_quality/'))
        if not os.path.exists(os.path.join(save_path, 'Bad_quality/')):
            os.makedirs(os.path.join(save_path, 'Bad_quality/'))
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
            if save_path: 
                shutil.copy(name_list[i], os.path.join(save_path, 'Good_quality/'))
            img_quality["good"].append(name_list[i]) # TODO: Do we want to return the name or the image

        elif (Eyepacs_pre[i]==1) and (Eyepacs_bad_mean[i]<0.25):
            Eye_good+=1
            if save_path:
                shutil.copy(name_list[i], os.path.join(save_path, 'Good_quality/'))
            img_quality["good"].append(name_list[i])
        else:
            Eye_bad+=1
            if save_path:
                shutil.copy(name_list[i], os.path.join(save_path, 'Bad_quality/'))
            img_quality["bad"].append(name_list[i])

    #print('Gradable cases by EyePACS_QA is {} '.format(Eye_good))
    #print('Ungradable cases by EyePACS_QA is {} '.format(Eye_bad))
    return img_quality