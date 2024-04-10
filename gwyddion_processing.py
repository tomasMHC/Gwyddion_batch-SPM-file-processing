
import numpy as np
import matplotlib.pyplot as plt
from topostats.plottingfuncs import Images
from topostats.theme import Colormap
from topostats.io import find_files, read_yaml, write_yaml, LoadScans
from topostats.plottingfuncs import Images
from datetime import datetime
from topostats.filters import Filters
import tkinter as tk
import tkinter.filedialog as fd
import pandas as pd
import os
from matplotlib.colors import LightSource

BASE_DIR = str(r"C:\Users\zbnj4y\OneDrive - onsemi\_TP\Programming\AFM_topostats\TopoStats")
DATA_DIR = str(r"C:\Users\zbnj4y\OneDrive - onsemi\_TP\Programming\AFM_topostats\Data")
FILE_EXT = ".spm"

THRLD_MULTIPLIER = 25

spms_input=[]
rms_data=[]
file_name=[]
mat_mean=[]
ra_data=[]

class timed_out(LoadScans):
    def timed_out_func(im_files):
        conf=read_yaml(BASE_DIR + "\my_onfig.yaml")
        a=LoadScans(im_files, **conf["loading"])
        a.get_data()
        return a

def stamp(str):
    current_time = datetime.now()
    time_stamp = current_time.timestamp()
    date_time = datetime.fromtimestamp(time_stamp)
    str_date_time = date_time.strftime("%d%m%Y_%H%M%S")
    return(str+"_"+str_date_time)

def nm_or_ang(roughness):
    if roughness<1:
        roughness=roughness*10
        a = f"{np.round((roughness),3)} A"
    else: a= f"{np.round((roughness),3)} nm"
    return a

def roughness_rms(image: np.ndarray) -> float:
    return np.sqrt(np.nanmean(np.square(image)))

def roughness_ra(image: np.ndarray) -> float:
    mean_height = np.mean(image)
    ra_roughness = np.mean(np.abs(image - mean_height))
    return ra_roughness

root = tk.Tk()
root.withdraw()
spms_input = fd.askdirectory(parent=root, title='Select spms directory')

if len(spms_input)>5:
    print(f"Input: {spms_input}")
    # Search for *.spm files one directory level up from the current notebooks
    image_files = find_files(base_dir=spms_input, file_ext=FILE_EXT)
    config = read_yaml(BASE_DIR + "\my_onfig.yaml")
    loading_config = config["loading"]
    filter_config = config["filter"]
    filter_config.pop("run")
    grain_config = config["grains"]
    grain_config.pop("run")
    grainstats_config = config["grainstats"]
    grainstats_config.pop("run")
    plotting_config = config["plotting"]
    plotting_config.pop("run")

    all_scan_data=timed_out.timed_out_func(image_files)

    for i in image_files:
        config = read_yaml(BASE_DIR + "\my_onfig.yaml")
        loading_config = config["loading"]
        filter_config = config["filter"]
        filter_config.pop("run")
        grain_config = config["grains"]
        grain_config.pop("run")
        grainstats_config = config["grainstats"]
        grainstats_config.pop("run")
        plotting_config = config["plotting"]
        plotting_config.pop("run")
        filer=(str(i).split("\\")[-1])
        filer=filer.rsplit(".",1)[0]
        print(filer)
        filtered_image = Filters(
            image=all_scan_data.img_dict[str(filer)]["image_original"],
            filename=all_scan_data.img_dict[str(filer)]["img_path"],
            pixel_to_nm_scaling=all_scan_data.img_dict[str(filer)]["pixel_to_nm_scaling"],
            row_alignment_quantile=filter_config["row_alignment_quantile"],
            threshold_method=filter_config["threshold_method"],
            otsu_threshold_multiplier=filter_config["otsu_threshold_multiplier"],
            threshold_std_dev=filter_config["threshold_std_dev"],
            threshold_absolute=filter_config["threshold_absolute"],
            gaussian_size=filter_config["gaussian_size"],
            gaussian_mode=filter_config["gaussian_mode"],
            remove_scars=filter_config["remove_scars"],
        )
        filtered_image.filter_image()
        pix_to_nm=all_scan_data.img_dict[str(filer)]["pixel_to_nm_scaling"]
        
        spm_array=np.array(filtered_image.images["masked_nonlinear_polynomial_removal"])
        orig_array=spm_array
        v_mean=(np.nanmedian(spm_array))
        
        spm_array[spm_array > v_mean*THRLD_MULTIPLIER] = v_mean
        spm_array_zero=spm_array-np.nanmin(spm_array)
        rms = roughness_rms(spm_array)
        roughnessRms=("RMS: "+nm_or_ang(rms))
        ra=roughness_ra(spm_array)
        Ra_roughness=("Ra: "+nm_or_ang(ra))


        # # 3D plot
        # tdplot=filtered_image.images["gaussian_filtered"]
        # tdplot_mean=(np.nanmean(tdplot))
        # tdplot[tdplot > tdplot_mean*THRLD_MULTIPLIER] = tdplot_mean
        # coef=int(spm_array.shape[0]*pix_to_nm/1000)
        
        # x=np.arange(0,spm_array.shape[0]*pix_to_nm/1000,1)
        # y=np.arange(0,spm_array.shape[1]*pix_to_nm/1000,1)
        # x=np.linspace(0,coef,int(spm_array.shape[0]))
        # y=np.linspace(0,coef,int(spm_array.shape[1]))
        # Y, X = np.meshgrid(x, y)

        # fig = plt.figure()
        # fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        # # light_source = np.array([1, 1, 1])
        # ls=LightSource(270,45)
        # rgb = ls.shade(tdplot, cmap=Colormap("afmhot").get_cmap(), vert_exag=0.1, blend_mode='overlay')
        # ax.plot_surface(X, Y, tdplot, cmap=Colormap("afmhot").get_cmap(),rstride=2,cstride=2,shade=False,antialiased=False,facecolors=rgb,linewidth=0) 
        # ax.set_zlim((np.nanmin(tdplot)-np.nanmin(tdplot)*0.1), (np.nanmax(tdplot)+np.nanmax(tdplot)*0.5))
        # plt.savefig("C:/Users/zbnj4y/OneDrive - onsemi/_TP/Programming/AFM_topostats/img_out"+"/"+filer+".png",format="png",dpi=600)
        # plt.close()

        rms_data.append(rms*10)
        file_name.append(filer)
        mat_mean.append(v_mean)
        ra_data.append(ra*10)

        current_cmap = plotting_config.pop("cmap")
        current_zrange = plotting_config.pop("zrange")
        fig1, ax = Images(
            data=spm_array_zero,
            filename="img_"+filer,
            output_dir=r"C:\Users\zbnj4y\OneDrive - onsemi\_TP\Programming\AFM_topostats\img_out",
            pixel_to_nm_scaling=pix_to_nm,
            title=filer,
            cmap="nanoscope",
            # zrange=[0, np.nanmean(spm_array)+0.2],
            text1=Ra_roughness,
            text2=roughnessRms,
            save=True,
            **plotting_config,
        ).save_figure()
        plt.close()

        # Restore the value for cmap to the dictionary.
        plotting_config["cmap"] = current_cmap
        plotting_config["zrange"] = current_zrange
        fig1

    d={"Name":file_name,"RMS(A)": rms_data,"Ra(A)": ra_data}
    df=pd.DataFrame(data=d)
    CSV_DIR=DATA_DIR+"\csv_data"
    if not os.path.exists(CSV_DIR):
        os.makedirs(CSV_DIR)
    df.to_csv(CSV_DIR+"\\"+stamp("data")+".csv",sep=',', index=False, encoding='utf-8')
    print(df)