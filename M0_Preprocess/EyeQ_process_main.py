from . import fundus_prep as prep
import os
import pandas as pd
from PIL import ImageFile
import shutil
ImageFile.LOAD_TRUNCATED_IMAGES = True

AUTOMORPH_DATA = os.getenv('AUTOMORPH_DATA','..')

def process(image_list, save_path, resolution_list = None):
    
    radius_list = []
    centre_list_w = []
    centre_list_h = []
    name_list = []
    list_resolution = []
    scale_resolution = []
    save = save_path is not None
    
    if resolution_list is None:
        if os.path.exists(f'{AUTOMORPH_DATA}/resolution_information.csv'):
            resolution_list = pd.read_csv(f'{AUTOMORPH_DATA}/resolution_information.csv')
    elif type(resolution_list) == str:
        resolution_list = pd.read_csv(resolution_list)
    img_list = []
    for index, image_path in enumerate(image_list):
        if isinstance(image_path, str):
            resolution_ = resolution_list['res'][resolution_list['fundus']==os.path.basename(image_path)].values[0] if resolution_list is not None else 0.008
            list_resolution.append(resolution_)

            dst_img = f'{save_path}' + os.path.basename(image_path).split('.')[0] + '.png'
            if save and os.path.exists(dst_img):
                img = prep.imread(dst_img)
                img_list.append(img)
            else:
                name_list.append(os.path.basename(image_path).split('.')[0] + '.png')
                img = prep.imread(image_path)
                r_img, borders, mask, r_img, radius_list,centre_list_w, centre_list_h = prep.process_without_gb(img,img,radius_list,centre_list_w, centre_list_h)
                img_list.append(r_img)
                if save: prep.imwrite(dst_img, r_img)

        else:
            resolution_ = resolution_list['res'][index] if resolution_list is not None else 0.008
            list_resolution.append(resolution_)

            name_list.append(str(index))
            img = image_path
            r_img, borders, mask, r_img, radius_list,centre_list_w, centre_list_h = prep.process_without_gb(img,img,radius_list,centre_list_w, centre_list_h)
            img_list.append(r_img)
            if save: prep.imwrite(f'{save_path}' + str(index) + '.png', r_img)

    scale_list = [a*2/912 for a in radius_list]
    scale_resolution = [a*b*1000 for a,b in zip(list_resolution,scale_list)]
    Data4stage2 = pd.DataFrame({'Name':name_list, 'centre_w':centre_list_w, 'centre_h':centre_list_h, 'radius':radius_list, 'Scale':scale_list, 'Scale_resolution':scale_resolution})
    if save: Data4stage2.to_csv(f'{save_path}/crop_info.csv', index = None, encoding='utf8', mode='w')

    return img_list, Data4stage2

if __name__ == "__main__":
    if os.path.exists(f'{AUTOMORPH_DATA}/images/.ipynb_checkpoints'):
        shutil.rmtree(f'{AUTOMORPH_DATA}/images/.ipynb_checkpoints')
    image_list = sorted(os.listdir(f'{AUTOMORPH_DATA}/images'))
    save_path = f'{AUTOMORPH_DATA}/Results/M0/images/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    process(image_list, save_path)

        




