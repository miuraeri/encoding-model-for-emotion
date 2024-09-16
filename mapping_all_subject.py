import argparse
import numpy as np
import cortex

# コマンドライン引数を設定
parser = argparse.ArgumentParser()
parser.add_argument("-r", "--ridge_type", type=str,
                    choices=["Ridge","KernelRidge"],
                    default="Ridge",
                    help="input ridge type")
parser.add_argument("-m", "--mode", type=str,
                    choices=["without_concat","with_concat", "run_test"],
                    default="with_concat",
                    help="input feature used")
args = parser.parse_args()

ridge_type = args.ridge_type
mode = args.mode

# パスの設定
target_subject = "sub-22"
sub_id_list = [18,22,23,24,28,30,31,35,36,37,38,39,41,42,43,44,45,46,47,48,49,50,51,52,53,
               57,58,61,62,63,64,65,67,68,69,70,72,73,74,75,76,77,78,79,81,82,83,84,86,87,
               88,89,91,92,93,94,95,96,97,98,99,100,101,103,104,105,106,108,109,110,113,114,115]

for sub_id in sub_id_list:
    if sub_id <= 53:
        subject = "sub-"+str(sub_id)
        model_path = "/home/miura/brain/model/with_concat/Ridge/encodingmodel_output_"+subject+"/"
        out_data_dir = "/home/miura/brain/src/all_subject_mean/data/Alice/"
        section = 5
    elif sub_id < 100:
        subject = "sub_EN0"+str(sub_id)
        model_path = "/home/miura/brain/src/encoding/LittlePrince/model/with_concat/Ridge/encodingmodel_output_"+subject+"/"
        out_data_dir = "/home/miura/brain/src/all_subject_mean/data/LittlePrince/"
        section = 9
    else:
        subject = "sub_EN"+str(sub_id)
        model_path = "/home/miura/brain/src/encoding/LittlePrince/model/with_concat/Ridge/encodingmodel_output_"+subject+"/"
        out_data_dir = "/home/miura/brain/src/all_subject_mean/data/LittlePrince/"
        section = 9

    print(subject+":start")

    for run in range(section):
        p_rejected_language = np.load(model_path+"run_"+str(run+1)+"/feature_based_extract/rejected_voxel_language.npy")
        p_rejected = np.load(model_path+"run_"+str(run+1)+"/feature_based_extract/rejected_voxel.npy")
        pred_test_emotion = np.load(model_path+"run_"+str(run+1)+"/feature_based_extract/pred_test_emotion.npy")

        pred_test_emotion = np.mean(pred_test_emotion,axis=0)
        pred_test_emotion[~p_rejected_language] = 0
        pred_test_emotion[~p_rejected] = 0
        pred_test_emotion[pred_test_emotion<0] = 0

        ver_each_run = cortex.Vertex(pred_test_emotion, subject=subject)
        new_ver_each_run = ver_each_run.map(target_subject, fs_subj=subject)

        np.save(out_data_dir+"run_"+str(run+1)+"/"+subject, new_ver_each_run.data)

        if run==0:
            pred_test_emotion_mean = pred_test_emotion
        else:
            pred_test_emotion_mean += pred_test_emotion

    pred_test_emotion_mean = pred_test_emotion_mean / section

    ver = cortex.Vertex(pred_test_emotion_mean, subject=subject)
    new_ver = ver.map(target_subject, fs_subj=subject)

    np.save(out_data_dir+"mean/"+subject, new_ver.data)    

    print(subject+":end")
