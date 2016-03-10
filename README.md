Deep Learning for Java
======================

## Overview

Deep Learningについていろいろ学習したのでそれを実装+API化

mavenを導入しているので注意です。それとIntelliJを使ってプログラミングしています(特に関係はないかも…)。

slf4j + logbackによってログの書き出しを行っています。logsディレクトリにログが出力されます。

大部分は車輪の再発明です。

### Package:

  - imageprocessing: 画像処理関連パッケージ。ここに重みの可視化も追加する予定。

  - nn: ニューラルネット関連パッケージ。
  
        - layers: ニューラルネットのlayer関連。HiddenLayerやAutoEncoder、LogisticRegressionなどもここにある。
        
        - multilayer: 多層ネット関連。MultiLayerPerceptronやStackedAutoEncoderなど。
        
        - nlp: ニューラルネットを用いた言語モデル関連。NNLMやRNNLMなどがここにある。そのほかにもNLP関連のクラスがある。
        
        - examples: exampleコード。
    
    

### Reference:

  - 深層学習 Deep Learning (MLP 機械学習プロフェッショナルシリーズ), 岡谷貴之

  - http://deeplearning4j.org

  - http://yusugomori.com/projects/deep-learning/
  
  その他いろいろ