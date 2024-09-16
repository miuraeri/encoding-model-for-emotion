import mne
import os
import cortex
import numpy as np

freesurfer_subject_dir = "/home/miura/freesurfer/subjects"
# roi_list = ["inferiortemporal","lateraloccipital","lateralorbitofrontal","medialorbitofrontal","middletemporal","parsorbitalis","rostralmiddlefrontal","frontalpole"]
roi_list = ["bankssts","caudalmiddlefrontal","cuneus","entorhinal","fusiform","inferiorparietal","inferiortemporal",
            "isthmuscingulate","lateraloccipital","lateralorbitofrontal","lingual","medialorbitofrontal",
            "middletemporal","parahippocampal","paracentral","parsopercularis","parsorbitalis","parstriangularis",
            "pericalcarine","postcentral","posteriorcingulate","precentral","precuneus","rostralanteriorcingulate",
            "rostralmiddlefrontal","superiorfrontal","superiorparietal","superiortemporal","supramarginal",
            "frontalpole","temporalpole","transversetemporal","insula"]

sub_id_list = [18,22,23,24,28,30,31,35,36,37,38,39,41,42,43,44,45,46,47,48,49,50,51,52,53,
               57,58,61,62,63,64,65,67,68,69,70,72,73,74,75,76,77,78,79,81,82,83,84,86,87,
               88,89,91,92,93,94,95,96,97,98,99,100,101,103,104,105,106,108,109,110,113,114,115]

target_subject = "sub-22"
for sub_id in sub_id_list:
    if sub_id <= 53:
        subject = "sub-"+str(sub_id)
        model_path = "/home/miura/brain/model/with_concat/Ridge/encodingmodel_output_"+subject+"/"
        section = 5
    elif sub_id < 100:
        subject = "sub_EN0"+str(sub_id)
        model_path = "/home/miura/brain/src/encoding/LittlePrince/model/with_concat/Ridge/encodingmodel_output_"+subject+"/"
        section = 9
    else:
        subject = "sub_EN"+str(sub_id)
        model_path = "/home/miura/brain/src/encoding/LittlePrince/model/with_concat/Ridge/encodingmodel_output_"+subject+"/"
        section = 9
    print(subject,":start")
    for i in range(section):
        save_path = model_path+"run_"+str(i+1)+"/PCA_ROI/each_roi_matrix/"
        os.makedirs(save_path, exist_ok=True)
        mean_each_emotion = np.load(model_path+"run_"+str(i+1)+"/PCA_ROI/matrix/"+subject+"_section_"+str(i+1)+".npy")

        for roi in roi_list:
            for j in range(mean_each_emotion.shape[0]): 
                ver = cortex.Vertex(mean_each_emotion[j], subject=subject)
                new_ver = ver.map(target_subject, fs_subj=subject)

                MT_exvivo_lh_voxels= mne.read_label(os.path.join(freesurfer_subject_dir, target_subject, 'label', "lh."+roi+".label")).vertices 
                MT_exvivo_rh_voxels= mne.read_label(os.path.join(freesurfer_subject_dir, target_subject, 'label', "rh."+roi+".label")).vertices
                mtl_data= np.ones(len(MT_exvivo_lh_voxels)) # get data from your project 
                mtr_data= np.ones(len(MT_exvivo_rh_voxels)) # get data from your project 
                surfs = [cortex.polyutils.Surface(*d) for d in cortex.db.get_surf(target_subject, "fiducial")] 
                brain_cortex_l= np.zeros(surfs[0].pts.shape[0])
                brain_cortex_r= np.zeros(surfs[1].pts.shape[0]) 
                brain_cortex_l[MT_exvivo_lh_voxels] = mtl_data 
                brain_cortex_r[MT_exvivo_rh_voxels] = mtr_data 
                brain_cortex = np.concatenate([brain_cortex_l, brain_cortex_r]).astype("bool")
                mask = new_ver.data
                mask[~brain_cortex]=0
                if j == 0:
                    new_ver_stack = mask
                else:
                    new_ver_stack = np.vstack([new_ver_stack, mask])
            np.save(save_path+roi,new_ver_stack)
    print(subject,":end")
