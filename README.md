# encoding-model-for-emotion
## データの前処理
### textgridファイル(単語+読まれた時間のデータ)の処理
### 脳活動データの前処理
1. fMRIデータをVertex形式(2次元配列)へ変換: `preprocessing_brain_volume.py`
2. データの正規化: `normalization.py`
## 多段階ファインチューニング
1段階目: [GoEmotions-pytorch](https://github.com/monologg/GoEmotions-pytorch) を参照。コード実行時に`--taxonomy original`を指定。  
2段階目: 上記のconfigディレクトリに`alice.json`を追加。`--taxonomy alice`を指定。
## 特徴量の作成
言語特徴量: `sentence_to_bert_feature.py`  
感情特徴量: `sentence_to_emotion_feature.py`
## 符号化モデルの作成
`build_encoding.py` : 符号化モデルの構築およびパラメータ・モデルの保存
## 感情のみに関する予測脳活動の抽出
### 各被験者の予測脳活動

### 全被験者・男女別の平均予測脳活動

## 関心領域別の各感情の反応強度調査
