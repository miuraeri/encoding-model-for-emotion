import random
import argparse
from sklearn.model_selection import KFold
# 元のコードはsklearnのRidgeを使用しているが、ここをhimalayaへ変更
from himalaya.ridge import Ridge
from himalaya.kernel_ridge import KernelRidge
from himalaya.backend import set_backend
import matplotlib.pyplot as plt
from statsmodels.stats.multitest import fdrcorrection as fdr
import matplotlib as mpl
import numpy as np
from scipy import stats
import os
# 実行時間計測
# discord へ開始、終了通知を送信
import pickle

# GPU を使用するよう変更する
backend = set_backend("torch_cuda")

# コマンドライン引数を設定
parser = argparse.ArgumentParser()
parser.add_argument("-s", "--sub_id", type=int, required=True,
                    choices=[18,22,23,24,28,30,31,35,36,37,38,39,41,42,43,44,45,46,47,48,49,50,51,52,53,
                             57,58,61,62,63,64,65,67,68,69,70,72,73,74,75,76,77,78,79,81,82,83,84,86,87,
                             88,89,91,92,93,94,95,96,97,98,99,100,101,103,104,105,106,108,109,110,113,114,115],
                    help="input subject id")
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

# 関数の定義
'''
    ペアデータを作る
    
    <<parameters>>
    feature: ペアデータにするデータ
    width: 時間幅
    delay: 遅延
    
    <<return>>
    paired_data: ペアデータになったもの
'''
def make_pair_data(feature, width, delay):
    for i in range(width):
        a = np.roll(feature, delay+i, axis=0)
        if i==0:
            paired_data = a
        else:
            paired_data = np.concatenate((paired_data, a), axis=1)
    return paired_data


'''
    cvに使用するインデックスを決める
    
    <<parameters>>
    samplenum: サンプル数
    chunklen: 何個のデータを連続した塊と見るか
    n_cv: 何foldか
    
    <<return>>
    cv_inds: 辞書。n_cv回分のidが入っている
'''
def get_cv_inds(samplenum, chunklen, n_cv):
    # 塊を作ってシャッフル
    allinds = range(samplenum)
    indchunks = list(zip(*[iter(allinds)]*chunklen))
    random.shuffle(indchunks)
    inds = np.array(indchunks).flatten()
    
    cv_inds = []
    kf = KFold(n_splits=n_cv, shuffle=False)
    for trn_ind, val_ind in kf.split(inds):
        cv_inds.append({'trn_ind': inds[trn_ind], 'val_ind': inds[val_ind]})
    return cv_inds

'''
    alphaのグラフを可視化する
    
    <<parameters>>
    alphas: (a, ) アルファ達
    corrs: (a, ) それぞれのアルファの平均の相関係数
    save: (bool) Trueならばグラフを保存する
'''

def alpha_graph(alphas, corrs, b_alpha, n_cv, fig_name=""):
    fig = plt.figure()
    mcc = corrs[np.where(alphas==b_alpha)][0]
    plt.title('average corr coef in training(CV:{})\nbest alpha: {:.2e}, mean corr coef: {:.3f}'.format(n_cv, b_alpha, mcc))
    plt.xlabel('alpha')
    plt.ylabel('average correlation coefficient')
    plt.plot(alphas, corrs, marker='o')
    plt.plot(b_alpha, mcc, color='red', marker='o')
    plt.xscale('log')
    plt.minorticks_on()
    plt.grid(axis='x')
    plt.grid(which='both', axis='y', ls='--')
    plt.savefig(fig_name)
    plt.show()
'''
    pを求める
'''
def compute_p(corrs, n):
    t = np.dot(corrs, np.sqrt(n - 2)) / np.sqrt(1 - np.power(corrs, 2))
    p = (1 - stats.t.cdf(np.absolute(t), n - 2)) * 2
    return p

'''
    相関係数のヒストグラムを作成する
    
    <<parameters>>
    corr: (b, ) 相関係数
    range_min: (float) グラフのメモリの最小
    range_max: (float) グラフのメモリの最大
'''
def corr_graph(corr, range_min=-1.0, range_max=1.0, fig_name=""):

    fig = plt.figure()
    mean = sum(corr)/len(corr)
    plt.title('histogram of corr coef in test\nmean corr:{:.3g}'.format(mean))
    plt.xlabel('correlation coefficient')
    plt.ylabel('number of voxels')
    plt.grid()
    plt.hist(corr, range=(range_min, range_max), bins=40)
    plt.savefig(fig_name)

# パスの設定
sub_id = args.sub_id
if sub_id <= 53:
    subject = "sub-"+str(sub_id)
    section = 5
elif sub_id < 100:
    subject = "sub_EN0"+str(sub_id)
    section = 9
else:
    subject = "sub_EN"+str(sub_id)
    section = 9

if sub_id <= 53:
    brain_path = '/home/sync/brain/pycortex_project/resource/Alice/sub-' + str(sub_id) + "/normalization/split_5data_line_trilinear/"
    brain_files= []
    for i in range(5):
    brain_files.append("sub-" + str(sub_id) + "_" + str(i+1) + ".npy")

    feature_path = '/home/sync/brain/Alice/features/' # 特徴量の .npy を作成したら設定
    if mode == "without_concat":
        feature_file = 'emotion_features_365.npy'
    elif mode == "with_concat":
        feature_file = 'concat_features_365.npy'
    else:
        feature_file = 'emotion_features_365.npy'

    out_data_dir = '/home/sync/brain/model/'+ mode + "/" + ridge_type +'/encodingmodel_output_sub-' + str(sub_id)
    os.mkdir(out_data_dir)

    # 特徴量の読み込み
    cut_timing = [47,162,241,305]
    # cut_timing = [43,77,115,155,191,229,260,297,334]
    with open(feature_path + feature_file, 'rb') as f:
        feature = np.load(f)

    for i in range(len(brain_files)):
        for j in range(len(brain_files)):
            with open(brain_path + brain_files[j], 'rb') as f:
                data = np.load(f)
            if i!=0 and j==0: # リッジ回帰が1回目でなくて最初のファイルを読み込むとき
                brain_trn = data # 最初のファイルは訓練用として使う
            elif i==0 and j==1: # リッジ回帰が1回目で2つ目のファイルを読み込むとき
                brain_trn = data # 最初のファイルはテスト用になるので、2つ目のファイルを訓練用として登録
            elif j==i:
                brain_test = data
            else:
                brain_trn = np.vstack([brain_trn, data])

    # load_imge と形式を合わせる方法    
    print('brain_trn shape: ', brain_trn.shape) # (データ数・時間,脳活動データの次元・大脳皮質) 配列を取り出せれば形式はなんでも
    print('brain_test shape: ', brain_test.shape)

    n_voxels = brain_trn.shape[1] # ボクセル数
    print('n_voxels: ', n_voxels)
    
    # 特徴量
    if i==0:
        feature_trn = feature[cut_timing[i]:]
        feature_test = feature[:cut_timing[i]]
    elif i == len(brain_files) - 1:
        feature_trn = feature[:cut_timing[i-1]]
        feature_test = feature[cut_timing[i-1]:]
    else:
        feature_trn = feature[:cut_timing[i-1]]
        feature_trn = np.vstack([feature_trn,feature[cut_timing[i]:]])
        feature_test = feature[cut_timing[i-1]:cut_timing[i]]

    print('feature_trn shape: ', feature_trn.shape) # (データ数、特徴量の次元)
    print('feature_test shape: ', feature_test.shape)
    
else:
    brain_path = "/home/sync/brain/pycortex_project/resource/littlePrince/" + subject + "/echo_1_cortex/normalization/"
    brain_files= []
    feature_files= []
    
    feature_path = '/home/sync/brain/src/encoding/LittlePrince/features/data/' # 特徴量の .npy を作成したら設定
    if mode == "without_concat":
        feature_file_path = '/home/sync/brain/src/encoding/LittlePrince/features/data/features_emotion/'
    elif mode == "with_concat":
        feature_file_path = '/home/sync/brain/src/encoding/LittlePrince/features/data/features_concated/'
    else:
        feature_file_path = '/home/sync/brain/src/encoding/LittlePrince/features/data/features_bert-cased/'
    
    for i in range(9):
        brain_files.append("run-" + str(i+1) + ".npy")
        feature_files.append("features_section_" + str(i+1) + ".npy")
    
    out_data_dir = '/home/sync/brain/src/encoding/LittlePrince/model/'+ mode + "/" + ridge_type +'/encodingmodel_output_' + subject
    os.makedirs(out_data_dir, exist_ok=True)
    
    # 脳活動データ・特徴量の読み込み
    for i in range(9):
        for j in range(9):
            with open(brain_path + brain_files[j], 'rb') as f:
                brain_data = np.load(f)
            with open(feature_file_path + feature_files[j], 'rb') as f:
                feature_data = np.load(f)
            if i!=0 and j==0: # リッジ回帰が1回目でなくて最初のファイルを読み込むとき
                brain_trn = brain_data # 最初のファイルは訓練用として使う
                feature_trn = feature_data
            elif i==0 and j==1: # リッジ回帰が1回目で2つ目のファイルを読み込むとき
                brain_trn = brain_data # 最初のファイルはテスト用になるので、2つ目のファイルを訓練用として登録
                feature_trn = feature_data
            elif j==i:
                brain_test = brain_data
                feature_test = feature_data
            else:
                brain_trn = np.vstack([brain_trn, brain_data])
                feature_trn = np.vstack([feature_trn, feature_data])
    
        # load_imge と形式を合わせる方法    
        print('brain_trn shape: ', brain_trn.shape) # (データ数・時間,脳活動データの次元・大脳皮質) 配列を取り出せれば形式はなんでも
        print('brain_test shape: ', brain_test.shape)
    
        n_voxels = brain_trn.shape[1] # ボクセル数
        print('n_voxels: ', n_voxels)
    
        print('feature_trn shape: ', feature_trn.shape) # (データ数、特徴量の次元)
        print('feature_test shape: ', feature_test.shape)

    width = 4
    delay = 2

    feature_trn_paired = make_pair_data(feature_trn, width, delay)
    feature_test_paired = make_pair_data(feature_test, width, delay)

    print('feature_trn_paired shape:', feature_trn_paired.shape) # (データ数、　次元*4)
    print('feature_test_paired shape:', feature_test_paired.shape)

    n_samples = feature_trn_paired.shape[0] # データ数
    chunklen = 25 # TR=2のデータなので、25くっつけたら50秒分くっつけたことになる
    n_cv = section # 何fold cross-validationをするか

    cv_inds = get_cv_inds(n_samples, chunklen, n_cv)

    alphas = np.logspace(-5, 10, 15) # ridge回帰の正則化項
    alphas = alphas.astype("float32")
    n_alphas = len(alphas) # 試す正則化項の数

    cv_corr = np.zeros((n_voxels, n_alphas), dtype=np.float32) # 交差検証の結果
    flag=True

    # リッジ回帰+CV
    for k in range(n_cv):
        print('Cross validation {}/{}'.format(k+1, n_cv))
        
        # 訓練データ、検証データを選ぶ
        trn_ind = cv_inds[k]['trn_ind']
        val_ind = cv_inds[k]['val_ind']
        
        X_trn = feature_trn_paired[trn_ind]
        y_trn = brain_trn[trn_ind]
        X_val = feature_trn_paired[val_ind]
        y_val = brain_trn[val_ind]

        # メモリと計算速度を上げるにはfloat32にした方がいいらしい。
        if ridge_type == "KernelRidge":
            X_trn = X_trn.astype("float32")
            y_trn = y_trn.astype("float32")
            X_val = X_val.astype("float32")
            y_val = y_val.astype("float32")
        
        
        corr_lst = [] # (n_alphas, n_voxels)
        for a in alphas: # 全てのalphaでリッジ回帰をしてみる
            # リッジ回帰
            if ridge_type == "Ridge":
                clf = Ridge(alpha=a)
            elif ridge_type == "KernelRidge":
                clf = KernelRidge(alpha=a)
            clf.fit(X_trn, y_trn)
            pred = clf.predict(X_val)
        
            # ボクセルごとに相関を求める
            corr = [] # (n_voxels, )
            for l in range(y_val.shape[1]):
                # r, _ = pearsonr(y_val.T[l,:], pred.T[l,:]) # 相関とp値が求められる関数。np.corrcoefとかでも相関は求められる。
                r = np.corrcoef(y_val.T[l,:], pred.T[l,:])[0,1]
                corr.append(r)
                
            corr_lst.append(corr)
            
        if flag:
            print('corr shape:', np.array(corr).shape)
            print('corr_lst shape:', np.array(corr_lst).shape)
            flag=False
                
        cv_corr += np.transpose(np.array(corr_lst)) # corr_lstを転置して足していく（転置はしてもしなくても良いですが...）
        
    cv_corr = cv_corr / n_cv # 足したので、平均をとります
    print('cv_corr shape:', np.array(cv_corr).shape)

    mean_corr = np.nanmean(cv_corr, axis=0) # alphaごとに平均
    print('mean_corr shape:', mean_corr.shape)

    best_ind = np.argmax(mean_corr) # 平均の相関係数が最も高くなるalphaの位置
    best_alpha = alphas[best_ind] # 平均の相関係数が最も高くなるalpha。これをモデルの正則化項として採用します
    print('index: {}, best_alpha: {}'.format(best_ind, best_alpha))

    # 採用したalphaで学習
    # どのalphaを使ったか、とか重みとかは保存しておくと良いかもしれません
    if ridge_type == "Ridge":
        rg = Ridge(alpha=best_alpha)
    elif ridge_type == "KernelRidge":
        rg = KernelRidge(alpha=best_alpha)
    
    rg.fit(feature_trn_paired, brain_trn)

    # 予測
    # この予測そのものも保存すると良いかも
    pred_test = rg.predict(feature_test_paired)
    print('pred_test shape:', pred_test.shape)

    # 相関&P値
    corr_test = [] # (n_voxels, )

    # ボクセルごとに相関を求める
    for m in range(y_val.shape[1]): 
        # r, _ = pearsonr(brain_test.T[m,:], pred_test.T[m,:])
        r = np.corrcoef(brain_test.T[m,:], pred_test.T[m,:])[0,1]
        corr_test.append(r)
        
    corr_test = np.array(corr_test)
    p_test = compute_p(corr_test, corr_test.shape[0]) # p値を求める。これはライブラリとかを使っても求められると思います

    print('corr_test shape:', np.array(corr_test).shape)
    print('p_test shape:', np.array(p_test).shape)

    # FDR補正 0.05
    p_rejected, _ = fdr(p_test)
    p_rejected = np.asarray(p_rejected)
    print('p_rejected shape:', p_rejected.shape)
    print('棄却されたボクセル数:', np.count_nonzero(p_rejected))

    # 棄却されたところのみ。可視化のために<0は0にしてしまっている
    corr_rejected = np.array(corr_test)
    corr_rejected[~p_rejected] = 0
    corr_rejected[corr_rejected<0] = 0
    

    # データの保存
    # 保存用のディレクトリを作成
    new_save_path = out_data_dir + '/run_' + str(i+1)
    os.mkdir(new_save_path)

    # モデル
    with open(new_save_path+'model.pkl','wb') as f:
        pickle.dump(rg, f)

    # best alpha
    with open(new_save_path+"/best_alpha.txt", mode='w') as f:
        f.write(str(best_alpha))

    # 予測脳活動
    np.save(new_save_path+'/pred_test', pred_test)

    # 相関係数
    np.save(new_save_path+'/corr_test', corr_test)

    # 相関係数 補正済み
    np.save(new_save_path+'/corr_rejected', corr_rejected)

    # 棄却されたボクセル
    np.save(new_save_path+'/rejected_voxel', p_rejected)

    # 
    np.save(new_save_path+'/cv_corr', cv_corr)

    # 相関係数のヒストグラムをかく
    # これがマイナスが多すぎたり0が多すぎたりしたらモデルがうまくできてないかもと疑った方が良さそう
    # 棄却前、棄却後両方とも相関を保存しておくと良いかと思います
    corr_graph(corr_test, fig_name=new_save_path+"/corr_test_graph.png")
    corr_graph(corr_rejected, fig_name=new_save_path+"/corr_rejected_graph.png")        

    # グラフの保存
    # alphaと、その平均の相関係数のグラフを書いてみる
    # これが上に凸な感じになっているかは確認した方が良いです（上がり（下り）続けそうな感じ形の場合、もっとalphaを大きく（小さく）した方が良い可能性があります）
    # 後は、平均の相関が極端に小さいとかだったらモデルがうまくできてないかもしれないと疑った方が良いかも...
    alpha_graph(alphas, mean_corr, best_alpha, n_cv, fig_name=new_save_path+"/alpha_graph.png")
