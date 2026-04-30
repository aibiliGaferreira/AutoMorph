#!/usr/bin/env python3

# Retipy - Retinal Image Processing on Python
# Copyright (C) 2017  Alejandro Valdes
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import argparse
import glob
import os
import shutil
import pandas as pd

from .retipy import configuration, retina, tortuosity_measures

def getFeatures(
        vessel_process,
        artery_process,
        vein_process,
        vessel_skeleton,
        artery_skeleton,
        vein_skeleton,
        resolution_df,
        optic_disc_df,
        save_path,
        pixels_per_window=15,
        sampling_size=6,
        r_2_threshold=0.96,
        names_list=None
    ):

    if isinstance(artery_skeleton, str) and os.path.isdir(os.path.join(artery_skeleton, ".ipynb_checkpoints")):
        shutil.rmtree(os.path.join(artery_skeleton, ".ipynb_checkpoints"))
    if isinstance(vessel_skeleton, str) and os.path.isdir(os.path.join(vessel_skeleton, ".ipynb_checkpoints")):
        shutil.rmtree(os.path.join(vessel_skeleton, ".ipynb_checkpoints"))
    if isinstance(vein_skeleton, str) and os.path.isdir(os.path.join(vein_skeleton, ".ipynb_checkpoints")):
        shutil.rmtree(os.path.join(vein_skeleton, ".ipynb_checkpoints"))

    if save_path is not None and not os.path.exists(f'{save_path}/Results/M3/Disc_centred/Width/'):
        os.makedirs(f'{save_path}/Results/M3/Disc_centred/Width/')

    binary_FD_binary,binary_VD_binary,binary_Average_width,binary_t2_list,binary_t4_list,binary_t5_list = [],[],[],[],[],[]
    artery_FD_binary,artery_VD_binary,artery_Average_width,artery_t2_list,artery_t4_list,artery_t5_list = [],[],[],[],[],[]
    vein_FD_binary,vein_VD_binary,vein_Average_width,vein_t2_list,vein_t4_list,vein_t5_list = [],[],[],[],[],[]
    
    name_binary_list = []
    name_artery_list = []
    name_vein_list = []

    if isinstance(vessel_skeleton, str):
        vessel_skeleton = sorted(glob.glob(os.path.join(vessel_skeleton, "*.png")))
    if isinstance(artery_skeleton, str):
        artery_skeleton = sorted(glob.glob(os.path.join(artery_skeleton, "*.png")))
    if isinstance(vein_skeleton, str):
        vein_skeleton = sorted(glob.glob(os.path.join(vein_skeleton, "*.png")))

    for index, filename in enumerate(vessel_skeleton):    
        #try:
            segmentedImage = retina.Retina(
                None,
                filename,
                store_path=vessel_process,
                index=index,
                resolution=resolution_df)
            window_sizes = [912]
            window = retina.Window(segmentedImage, window_sizes[-1], min_pixels=pixels_per_window)

            FD_binary,VD_binary,Average_width,t2,t4,td = tortuosity_measures.evaluate_window(
                window,
                pixels_per_window,
                sampling_size,
                r_2_threshold,
                store_path=vessel_process
            )
            
            binary_t2_list.append(t2)
            binary_t4_list.append(t4)
            binary_t5_list.append(td)
            binary_FD_binary.append(FD_binary)
            binary_VD_binary.append(VD_binary)
            binary_Average_width.append(Average_width)
            name_binary_list.append(filename.split('/')[-1] if isinstance(filename, str) else os.path.basename(names_list[index]) if names_list is not None else index)
        
        #except:
        #    binary_t2_list.append(-1)
        #    binary_t4_list.append(-1)
        #    binary_t5_list.append(-1)
        #    binary_FD_binary.append(-1)
        #    binary_VD_binary.append(-1)
        #    binary_Average_width.append(-1)
        #    name_binary_list.append(filename.split('/')[-1] if isinstance(filename, str) else os.path.basename(names_list[index]) if names_list is not None else index)

    for index, filename in enumerate(artery_skeleton):
        #try:
            segmentedImage = retina.Retina(
                None,
                filename,
                store_path=artery_process,
                index=index,
                resolution=resolution_df)
            window_sizes = [912]
            window = retina.Window(segmentedImage, window_sizes[-1], min_pixels=pixels_per_window)

            FD_binary,VD_binary,Average_width,t2,t4,td = tortuosity_measures.evaluate_window(
                window,
                pixels_per_window,
                sampling_size,
                r_2_threshold,
                store_path=artery_process
            )
        
            artery_t2_list.append(t2)
            artery_t4_list.append(t4)
            artery_t5_list.append(td)
            artery_FD_binary.append(FD_binary)
            artery_VD_binary.append(VD_binary)
            artery_Average_width.append(Average_width)
            name_artery_list.append(filename.split('/')[-1] if isinstance(filename, str) else os.path.basename(names_list[index]) if names_list is not None else index)
    
        #except:
        #    artery_t2_list.append(-1)
        #    artery_t4_list.append(-1)
        #    artery_t5_list.append(-1)
        #    artery_FD_binary.append(-1)
        #    artery_VD_binary.append(-1)
        #    artery_Average_width.append(-1)  
        #    name_artery_list.append(filename.split('/')[-1] if isinstance(filename, str) else os.path.basename(names_list[index]) if names_list is not None else index)  

    for index, filename in enumerate(vein_skeleton):
        #try:
            segmentedImage = retina.Retina(
                None,
                filename,
                store_path=vein_process,
                index=index,
                resolution=resolution_df)
            window_sizes = [912]
            window = retina.Window(segmentedImage, window_sizes[-1], min_pixels=pixels_per_window)
        
            FD_binary,VD_binary,Average_width,t2,t4,td = tortuosity_measures.evaluate_window(
                window,
                pixels_per_window,
                sampling_size,
                r_2_threshold,
                store_path=vein_process
            )
            
            vein_t2_list.append(t2)
            vein_t4_list.append(t4)
            vein_t5_list.append(td)
            vein_FD_binary.append(FD_binary)
            vein_VD_binary.append(VD_binary)
            vein_Average_width.append(Average_width)
            name_vein_list.append(filename.split('/')[-1] if isinstance(filename, str) else os.path.basename(names_list[index]) if names_list is not None else index)
    
        #except:
        #    vein_t2_list.append(-1)
        #    vein_t4_list.append(-1)
        #    vein_t5_list.append(-1)
        #    vein_FD_binary.append(-1)
        #    vein_VD_binary.append(-1)
        #    vein_Average_width.append(-1)
        #    name_vein_list.append(filename.split('/')[-1] if isinstance(filename, str) else os.path.basename(names_list[index]) if names_list is not None else index)

    if isinstance(optic_disc_df, str):
        Disc_file = pd.read_csv(optic_disc_df).astype({"Name": "object"})
    else:
        Disc_file = optic_disc_df
    
    if "Name" not in Disc_file.columns:
        Disc_file["Name"] = pd.Series(dtype="object")
    else:
        Disc_file["Name"] = Disc_file["Name"].astype("object")

    Data4stage2_binary = pd.DataFrame({
        "Name": name_binary_list,
        "Fractal_dimension": binary_FD_binary,
        "Vessel_density": binary_VD_binary,
        "Average_width": binary_Average_width,
        "Distance_tortuosity": binary_t2_list,
        "Squared_curvature_tortuosity": binary_t4_list,
        "Tortuosity_density": binary_t5_list,
    }).astype({"Name": "object"})

    Data4stage2_artery = pd.DataFrame({
        "Name": name_artery_list,
        "Artery_Fractal_dimension": artery_FD_binary,
        "Artery_Vessel_density": artery_VD_binary,
        "Artery_Average_width": artery_Average_width,
        "Artery_Distance_tortuosity": artery_t2_list,
        "Artery_Squared_curvature_tortuosity": artery_t4_list,
        "Artery_Tortuosity_density": artery_t5_list,
    }).astype({"Name": "object"})

    Data4stage2_vein = pd.DataFrame({
        "Name": name_vein_list,
        "Vein_Fractal_dimension": vein_FD_binary,
        "Vein_Vessel_density": vein_VD_binary,
        "Vein_Average_width": vein_Average_width,
        "Vein_Distance_tortuosity": vein_t2_list,
        "Vein_Squared_curvature_tortuosity": vein_t4_list,
        "Vein_Tortuosity_density": vein_t5_list,
    }).astype({"Name": "object"})

    Disc_file_binary = pd.merge(Disc_file, Data4stage2_binary, how="outer", on=["Name"])
    artery_vein = pd.merge(Data4stage2_artery, Data4stage2_vein, how="outer", on=["Name"])
    Data4stage2 = pd.merge(Disc_file_binary, artery_vein, how="outer", on=["Name"])

    if save_path is not None:
        Data4stage2.to_csv(f'{save_path}/Disc_Measurement.csv', index = None, encoding='utf8')
    else: return Data4stage2

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-c",
        "--configuration",
        help="the configuration file location",
        default="resources/retipy.config")
    args = parser.parse_args()
    return args