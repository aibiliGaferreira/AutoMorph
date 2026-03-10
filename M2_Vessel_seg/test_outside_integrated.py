
import argparse
import logging
import os
import torch
import numpy as np
from tqdm import tqdm
from .model import Segmenter
from .dataset import SEDataset_out
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from PIL import Image
from .utils import Define_image_size
import torchvision
from skimage.morphology import skeletonize,remove_small_objects
from skimage import io
from .FD_cal import fractal_dimension,vessel_density
import shutil

AUTOMORPH_DATA = os.getenv('AUTOMORPH_DATA','..')

def filter_frag(images):
    if isinstance(images, str):
        data_path = images
        if os.path.isdir(data_path + 'resize_binary/.ipynb_checkpoints'):
            shutil.rmtree(data_path + 'resize_binary/.ipynb_checkpoints')

        image_list=os.listdir(data_path + 'resize_binary')
    else:
        image_list = images["mask_bin_small_list"]

    
    FD_cal=[]
    name_list=[]
    VD_cal=[]
    width_cal=[]

    binary_process_list = []
    binary_skeleton_list = []

    for i in image_list:
        if isinstance(images, str):
            img=io.imread(data_path + 'resize_binary/' + i, as_gray=True).astype(np.int64)
        else: 
            img = i.squeeze()
        img2=img>0
        img2 = remove_small_objects(img2, 30, connectivity=5)
        
        if isinstance(images, str):
            if not os.path.isdir(data_path + 'binary_process/'): os.makedirs(data_path + 'binary_process/') 
            io.imsave(data_path + 'binary_process/' + i , 255*(img2.astype('uint8')),check_contrast=False)
        else:
            binary_process_list.append(img2)

        skeleton = skeletonize(img2)
        if isinstance(images, str):
            if not os.path.isdir(data_path + 'binary_skeleton/'): os.makedirs(data_path + 'binary_skeleton/') 
            io.imsave(data_path + 'binary_skeleton/' + i, 255*(skeleton.astype('uint8')),check_contrast=False)
        else:
            binary_skeleton_list.append(skeleton)
        
        FD_boxcounting = fractal_dimension(img2)
        VD = vessel_density(img2)
        width = np.sum(img2)/np.sum(skeleton)
        FD_cal.append(FD_boxcounting)
        name_list.append(i)
        VD_cal.append(VD)
        width_cal.append(width)
    
    return {
        "binary_process_list": binary_process_list,
        "binary_skeleton_list": binary_skeleton_list,
        "FD_cal": FD_cal,
        "VD_cal": VD_cal,
        "width_cal": width_cal
    }


def segment_fundus(data_path, nets, loader, device, save=False):
    n_val = len(loader) 
    i = 0
    
    if save:
        seg_results_small_path='./outside_test/segs/'
        seg_results_raw_path='./outside_test/segs/'
        seg_results_small_path = data_path + 'resize/'
        seg_results_small_binary_path = data_path + 'resize_binary/'
        seg_results_raw_path = data_path + 'raw/'
        seg_results_raw_binary_path = data_path + 'raw_binary/'
        seg_uncertainty_small_path = data_path + 'resize_uncertainty/'
        seg_uncertainty_raw_path = data_path + 'raw_uncertainty/'
    
        if not os.path.isdir(seg_results_small_path): os.makedirs(seg_results_small_path)
        if not os.path.isdir(seg_results_raw_path): os.makedirs(seg_results_raw_path)
        if not os.path.isdir(seg_results_small_binary_path): os.makedirs(seg_results_small_binary_path)
        if not os.path.isdir(seg_results_raw_binary_path): os.makedirs(seg_results_raw_binary_path)
        if not os.path.isdir(seg_uncertainty_small_path): os.makedirs(seg_uncertainty_small_path)
        if not os.path.isdir(seg_uncertainty_raw_path): os.makedirs(seg_uncertainty_raw_path)
        
    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        uncertainty_img_small_list = []
        uncertainty_tensor_list = []
        mask_pred_small_list = []
        mask_pred_list = []
        mask_bin_small_list = []
        mask_bin_list = []

        for batch in loader:
            imgs = batch['image']
            ori_width=batch['width']
            ori_height=batch['height']
            img_name = batch['name']
            
            imgs = imgs.to(device=device, dtype=torch.float32)

            mask_preds = []
            for net in nets:
                with torch.no_grad():
                    mask = net(imgs)
                
                mask_pred_sigmoid = torch.sigmoid(mask)
                mask_preds.append(mask_pred_sigmoid)
            
            mask_pred_sigmoid = torch.mean(torch.stack(mask_preds), dim=0)
            
            uncertainty_map = torch.sqrt(torch.mean(torch.stack([(mask_pred_sigmoid - mp) ** 2 for mp in mask_preds]), dim=0))
            
            n_image = mask_pred_sigmoid.shape[0]
            
            for i in range(n_image):    
                n_img_name = img_name[i]
                n_ori_width = ori_width[i]
                n_ori_height = ori_height[i]

                # Uncertainty map
                uncertainty_img_small = torch.unsqueeze(uncertainty_map[i,...], 0)
                if save: save_image(uncertainty_img_small, seg_uncertainty_small_path+n_img_name+'.png')
                else: uncertainty_img_small_list.append(uncertainty_img_small.cpu().numpy())
                uncertainty_img = Image.fromarray(uncertainty_img_small.squeeze().cpu().numpy()).resize((n_ori_width,n_ori_height)).convert('L') 
                uncertainty_tensor = torchvision.transforms.ToTensor()(uncertainty_img)
                if save: save_image(uncertainty_tensor, seg_uncertainty_raw_path+n_img_name+'.png')
                else: uncertainty_tensor_list.append(uncertainty_tensor.cpu().numpy())

                # Segmentation map
                mask_pred_img_small = torch.unsqueeze(mask_pred_sigmoid[i,...], 0)
                if save: save_image(mask_pred_img_small, seg_results_small_path+n_img_name+'.png')
                else: mask_pred_small_list.append(mask_pred_img_small.cpu().numpy())
                mask_pred_img = Image.fromarray(mask_pred_img_small.squeeze().cpu().numpy()).resize((n_ori_width,n_ori_height)).convert('L') 
                mask_pred_tensor = torchvision.transforms.ToTensor()(mask_pred_img)
                if save: save_image(mask_pred_tensor, seg_results_raw_path+n_img_name+'.png')
                else: mask_pred_list.append(mask_pred_tensor.cpu().numpy())

                # Binary
                mask_pred_resize_bin=torch.zeros(mask_pred_img_small.shape)
                mask_pred_resize_bin[mask_pred_img_small>=0.5]=1
                if save: save_image(mask_pred_resize_bin, seg_results_small_binary_path+n_img_name+'.png')
                else: mask_bin_small_list.append(mask_pred_resize_bin.cpu().numpy())
                mask_pred_numpy_bin=torch.zeros(mask_pred_tensor.shape)
                mask_pred_numpy_bin[mask_pred_tensor>=0.5]=1
                if save: save_image(mask_pred_numpy_bin, seg_results_raw_binary_path+n_img_name+'.png')
                else: mask_bin_list.append(mask_pred_numpy_bin.cpu().numpy())

            pbar.update(imgs.shape[0])

    if not save:
        return {
            "uncertainty_img_small_list": uncertainty_img_small_list,
            "uncertainty_tensor_list": uncertainty_tensor_list,
            "mask_pred_small_list": mask_pred_small_list,
            "mask_pred_list": mask_pred_list,
            "mask_bin_small_list": mask_bin_small_list,
            "mask_bin_list": mask_bin_list
        }
    else:
        return data_path


def test_net(imgs=f'{AUTOMORPH_DATA}/Results/M1/Good_quality/', batch_size=8, device="cpu", dataset_train="ALL-SIX", image_size=(912,912), job_name="20210630_uniform_thres40_ALL-SIX", threshold=40, save=False):
    
    data_path = f'{AUTOMORPH_DATA}/Results/M2/binary_vessel/' # TODO: Do we want this a configurable parameter?
    FD_list = []
    Name_list = []
    VD_list = []
    
    dataset_data = SEDataset_out(imgs, image_size, threshold)
    test_loader = DataLoader(dataset_data, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False, drop_last=False)
    
    nets = []
    for i in range(10):
        dir_checkpoint = os.path.abspath(os.path.join(os.path.dirname(__file__), f"Saved_model/train_on_{dataset_train}/{job_name}_savebest_randomseed_{24+2*i}/"))
        net = Segmenter(input_channels=3, n_filters = 32, n_classes=1, bilinear=False)
        net.load_state_dict(torch.load(os.path.join(dir_checkpoint, 'G_best_F1_epoch.pth'), map_location=device))
        net.eval()
        net.to(device=device)
        nets.append(net)
        
    images = segment_fundus(data_path, nets, test_loader, device, save=save)
    
    analysis = filter_frag(images)
    images["binary_process_list"] = analysis["binary_process_list"]
    images["binary_skeleton_list"] = analysis["binary_skeleton_list"]
    
    if save and not os.path.exists(f'{AUTOMORPH_DATA}/Results/M3/'):
        os.makedirs(f'{AUTOMORPH_DATA}/Results/M3/')

    if not save:
        return {
            "images": images,
            "FD_list": analysis["FD_cal"],
            "Name_list": Name_list,
            "VD_list": analysis["VD_cal"],
            "width_cal": analysis["width_cal"]
        }
                            
    #Data4stage2 = pd.DataFrame({'Image_id':Name_list, 'FD_boxC':FD_list, 'Vessel_Density':VD_list})
    #Data4stage2.to_csv('../Results/M3/Binary_Features_Measurement.csv', index = None, encoding='utf8')
        
        

def get_args():
    
    parser = argparse.ArgumentParser(description='Utilize the symmetric equilibrium segmentation net',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ########################### Training setting ##########################
    
    parser.add_argument('--epochs', type=int, default=1, help='Number of epochs', dest='epochs')
    parser.add_argument('--batchsize', type=int, default=6, help='Batch size', dest='batchsize')
    parser.add_argument('--learning_rate', type=float, default=2e-4, help='Learning rate', dest='lr')
    parser.add_argument('--load', type=str, default=False, help='Load trained model from .pth file', dest='load')
    parser.add_argument('--discriminator', type=str, default='unet', help='type of discriminator', dest='dis')
    parser.add_argument('--jn', type=str, default='unet', help='type of discriminator', dest='jn')
    parser.add_argument('--worker_num', type=int, default=2, help='type of discriminator', dest='worker')
    parser.add_argument('--save_model', type=str, default='regular', help='type of discriminator', dest='save')
    parser.add_argument('--train_test_mode', type=str, default='trainandtest', help='train and test, or directly test', dest='ttmode') 
    parser.add_argument('--pre_threshold', type=float, default=0.0, help='threshold in standalisation', dest='pthreshold')   

    ########################### Training data ###########################

    parser.add_argument('--dataset', type=str, help='training dataset name', dest='dataset')
    parser.add_argument('--seed_num', type=int, default=42, help='Validation split seed', dest='seed')
    parser.add_argument('--dataset_test', type=str, help='test dataset name', dest='dataset_test')    
    parser.add_argument('--validation_ratio', type=float, default=10.0, help='Percent of the data that is used as validation 0-100', dest='val')
    parser.add_argument('--uniform', type=str, default='False', help='whether to uniform the image size', dest='uniform')
    parser.add_argument('--out_test', type=str, default='False', help='whether to uniform the image size', dest='data_path')
    
    ####################### Loss weights ################################
    
    parser.add_argument('--alpha', type=float, default=0.08, help='Loss weight of Adversarial Loss', dest='alpha')
    parser.add_argument('--beta', type=float, default=1.1, help='Loss weight of segmentation cross entropy', dest='beta')
    parser.add_argument('--gamma', type=float, default=0.5, help='Loss weight of segmentation mean square error', dest='gamma')

    return parser.parse_args()


if __name__ == '__main__':
    
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()
    # Check if CUDA is available
    if torch.cuda.is_available():
        logging.info("CUDA is available. Using CUDA...")
        device = torch.device("cuda:0")
    elif torch.backends.mps.is_available():  # Check if MPS is available (for macOS)
        logging.info("MPS is available. Using MPS...")
        device = torch.device("mps")
    else:
        logging.info("Neither CUDA nor MPS is available. Using CPU...")
        device = torch.device("cpu")

    logging.info(f'Using device {device}')


    image_size = Define_image_size(args.uniform, args.dataset)
    lr = args.lr
    
    test_net(data_path=args.data_path,
             batch_size=args.batchsize,
             device=device,
             dataset_train=args.dataset,
             dataset_test=args.dataset_test, 
             image_size=image_size, 
             job_name=args.jn,
             threshold=args.pthreshold,
             checkpoint_mode=args.save,
             mask_or=True, 
             train_or=False)
 