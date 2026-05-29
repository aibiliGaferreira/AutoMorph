from . import fundus_prep as prep
import os
import pandas as pd
from PIL import ImageFile
import shutil
ImageFile.LOAD_TRUNCATED_IMAGES = True

def process(images, save_path, resolution_list = None):
    
    radius_list = []
    centre_list_w = []
    centre_list_h = []
    name_list = []
    list_resolution = []
    scale_resolution = []
    save = save_path is not None

    if isinstance(images, str) and os.path.isfile(images):
        images = [images]

    if hasattr(images, "ndim") and images.ndim == 3:
        images = [images]
    
    if resolution_list is None:
        if os.path.exists(f'data/resolution_information.csv'):
            resolution_list = pd.read_csv(f'data/resolution_information.csv')
    elif type(resolution_list) == str:
        resolution_list = pd.read_csv(resolution_list)
    
    img_list = []
    for index, image_path in enumerate(images):
        if isinstance(image_path, str):
            resolution_ = resolution_list['res'][resolution_list['fundus']==os.path.basename(image_path)].values[0] if resolution_list is not None else 0.008
            list_resolution.append(resolution_)
            img = prep.imread(image_path)
            name_list.append(os.path.basename(image_path).split('.')[0] + '.png')
            r_img, borders, mask, r_img, radius_list,centre_list_w, centre_list_h = prep.process_without_gb(img,img,radius_list,centre_list_w, centre_list_h)

            if save:
                dst_img = os.path.join(save_path, os.path.basename(image_path).split('.')[0] + '.png')
                if os.path.exists(dst_img):
                    img = prep.imread(dst_img)
                    img_list.append(img)
                else:
                    img_list.append(r_img)
                    prep.imwrite(dst_img, r_img)
            else:
                img_list.append(r_img)

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
    if os.path.exists(f'data/images/.ipynb_checkpoints'):
        shutil.rmtree(f'data/images/.ipynb_checkpoints')
    image_list = sorted(os.listdir(f'data/images'))
    save_path = f'data/output/Results/M0/images/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    process(image_list, save_path)

        




