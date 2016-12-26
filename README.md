# Requirement

* chainer version 1.16.0
* windows version 10, 64bit
* CUDA version 7.5
* Python version 3.5.2 (Anaconda 4.1.1)

# cnn_model.py

* __init__ : cgpリストからCNNの定義
* __call__ : __init__で定義したCNNのforward処理の定義

# cnn_train.py

* __init__ : 引数で与えられたデータセットのロード
* __call__ : cnn_model.pyを呼ぶことで与えられたcgpリストからCNNの構築，学習を行う
