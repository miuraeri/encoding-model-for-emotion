import os
import cortex
import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np

# freesurferに登録されている被験者リスト
sub_id_list = [18,22,23,24,28,30,31,35,36,37,38,39,41,42,43,44,45,46,47,48,49,50,51,52,53,
               57,58,61,62,63,64,65,67,68,69,70,72,73,74,75,76,77,78,79,81,82,83,84,86,87,
               88,89,91,92,93,94,95,96,97,98,99,100,101,103,104,105,106,108,109,110,113,114,115]

freesurfer_subject_dir = "/home/miura/freesurfer/subjects" # <- $SUBJECTS_DIR
pycortex_dir = cortex.options.config.get("basic", "filestore") 
save_dir = "/home/miura/brain/pycortex_project/"
xfm = "fullhead2"

def traverse_files(directory, conditional):
    file_list = []
    for foldername, subfolders, filenames in os.walk(directory):
        for filename in filenames:
            if conditional in filename and ".json" not in filename:
                file_list.append(os.path.join(foldername, filename))
    return file_list

for sub_id in sub_id_list:
    if sub_id <= 53:
        subject = "sub-"+str(sub_id)
        dataset = "alice"
    elif sub_id < 100:
        subject = "sub_EN0"+str(sub_id)
        dataset = "lpp"
    else:
        subject = "sub_EN"+str(sub_id)
        dataset = "lpp"
    print(subject,":start")

    # pycortexへの被験者登録
    pycortex_subject_dir = os.path.join(pycortex_dir, subject) 
    curvature_png_file = os.path.join(pycortex_subject_dir, "curvature.png") 
    cortex.freesurfer.import_subj(subject, subject, freesurfer_subject_dir=freesurfer_subject_dir)
    # flat.giiの作成
    cortex.freesurfer.import_flat(subject, "full", cx_subject=subject, freesurfer_subject_dir=freesurfer_subject_dir)
    # pycortexのtransformsディレクトリ配下にfullheadを作成
    cortex.align.automatic(subject, "fullhead", '/home/miura/brain/' + dataset + "/" + subject + '/anat/'+ subject +'_T1w.nii.gz', use_fs_bbr=True)
    # boldのデータからfullhead2を作成
    if sub_id < 39:
        cortex.align.automatic(subject, xfm, '/home/miura/brain/' + dataset + "/" + subject + '/func/' + subject + '_task-alice_bold.nii.gz')
    elif sub_id <= 53:
        cortex.align.automatic(subject, xfm, '/home/miura/brain/' + dataset + "/" + subject + '/func/' + subject + '_task-alice_echo-2_bold.nii.gz')
    else:
        # この処理がうまくいかない被験者は「run-」以下の数字を変える
        cortex.align.automatic(subject, xfm, '/home/miura/brain/' + dataset + "/" + subject + '/func/'+ subject +'_task-lppEN_run-14_echo_1_bold.nii.gz')

    # 大脳皮質データを4D (xyzt) のfMRIデータから抽出 (2D行列へ)
    path_to_traverse = '/home/miura/brain/' + dataset + "/" + subject + '/func/'
    vol_to_ver_map = cortex.get_mapper(subject, xfm, 'line_nearest')

    if dataset == "alice":
        for file_name in sorted(traverse_files(path_to_traverse, subject)):
            print(file_name)
            img = nib.load(file_name)  # the data you want to plot
            volarray = img.get_fdata()  # e.g. (72,71,89) or (72,71,89,194)
            volarray = volarray.transpose(2, 1, 0, 3)  # 4D shape transpose
            brain_cortex = np.empty(len(vol_to_ver_map(cortex.Volume(volarray[:, :, :, 1], subject, xfm)).data))
        
            for i in range(volarray.shape[3]):
                ver = vol_to_ver_map(cortex.Volume(volarray[:, :, :, i], subject, xfm))
                brain_cortex = np.vstack([brain_cortex, ver.data])
            brain_cortex = np.delete(brain_cortex, 0, 0)
            np.save(save_dir+ "resource/Alice/" + subject + "/" + subject + "_cortex_line_nearest", brain_cortex)
    else:
        run = [1,2,3,4,5,6,7,8,9]
        for m in range(3):
            echo = "echo_" + str(m+1)
            os.makedirs(save_dir+ "resource/littlePrince/" + subject + "/" + echo + "_cortex", exist_ok=True)
            print(subject, echo +" starting")
            for j,file_name in enumerate(sorted(traverse_files(path_to_traverse, echo))):
                print(file_name)
                img = nib.load(file_name)  # the data you want to plot
                volarray = img.get_fdata()  # e.g. (72,71,89) or (72,71,89,194)
                volarray = volarray.transpose(2, 1, 0, 3)  # 4D shape transpose
                brain_cortex = np.empty(len(vol_to_ver_map(cortex.Volume(volarray[:, :, :, 1], subject, xfm)).data))
    
                for i in range(volarray.shape[3]):
                    ver = vol_to_ver_map(cortex.Volume(volarray[:, :, :, i], subject, xfm))
                    brain_cortex = np.vstack([brain_cortex, ver.data])
                brain_cortex = np.delete(brain_cortex, 0, 0)
                np.save(save_dir+ "resource/littlePrince/" + subject + "/" + echo + "_cortex/run-"+str(run[j]), brain_cortex)
    print(subject,":end")
    
