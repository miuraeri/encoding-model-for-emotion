import numpy as np
import os

def min_max_normalization(a, axis=None):
    a_min = a.min(axis=axis, keepdims=True)
    a_max = a.max(axis=axis, keepdims=True)
    return (a - a_min) / (a_max - a_min)

sub_list = [57,58,61,62,63,64,65,67,68,69,70,72,73,74,75,76,77,78,79,81,82,83,84,86,87,88,89,91,92,94,95,96,97,98,99,100,101,103,104,105,106,108,109,110,113,114,115]

for sub_id in sub_list:
    if sub_id < 100:
        subject = "sub_EN0"+str(sub_id)
        section = 9
    else:
        subject = "sub_EN"+str(sub_id)
        section = 9
    
    for i in range(3):
        for j in range(9):
            brain_data_path = "/home/miura/brain/pycortex_project/resource/littlePrince/"+subject+"/echo_"+str(i+1)+"_cortex/"
            brain_cortex = np.load(brain_data_path+"run-"+str(j+1)+".npy")
            brain_cortex = min_max_normalization(brain_cortex)
            new_dir_path = brain_data_path+"normalization"
            if not os.path.isdir(new_dir_path):
                os.mkdir(new_dir_path)
            np.save(new_dir_path + "/run-" + str(j+1) + ".npy", brain_cortex)
