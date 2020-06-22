CTR MODEL ACHIEVE:  
  [1]Interactive Model  
    1.[FM] Fast Context-aware Recommendations with Factorization Machines (UKON 2011).pdf  
    2.[PNN] Product-based Neural Networks for User Response Prediction (SJTU 2016).pdf  
    3.[Deep Crossing] Deep Crossing - Web-Scale Modeling without Manually Crafted Combinatorial Features (Microsoft 2016).pdf  
    4.[Wide & Deep] Wide & Deep Learning for Recommender Systems (Google 2016).pdf  
    5.[DeepFM] A Factorization-Machine based Neural Network for CTR Prediction (HIT-Huawei 2017).pdf  
    6.[DCN] Deep & Cross Network for Ad Click Predictions (Stanford 2017).pdf  
    7.[NFM] Neural Factorization Machines for Sparse Predictive Analytics (NUS 2017).pdf  
    8.[xDeepFM] xDeepFM - Combining Explicit and Implicit Feature Interactions for Recommender Systems (USTC 2018).pdf  
    9.[AFM] Attentional Factorization Machines - Learning the Weight of Feature Interactions via Attention Networks (ZJU 2017).pdf  
    10.[AutoInt] AutoInt Automatic Feature Interaction Learning via Self-Attentive Neural Networks(CIKM 2019).pdf  
    ...Later Building...  
      
  [2]Behavior Model  
    1.[DIN] Deep Interest Network for Click-Through Rate Prediction (Alibaba 2018).pdf  
    2.[DIEN] Deep Interest Evolution Network for Click-Through Rate Prediction (Alibaba 2019).pdf  
    3.[DSIN]Deep Session Interest Network for Click-Through Rate Predicti(Alibaba 2019).pdf  
    4.[SeqFM]Sequence-Aware Factorization Machines(2019).pdf  
    ...Later Building...  
    
  [3]目前计划:
    1.[MIMN]Practice on Long Sequential User Behavior Modeling for Click-Through Rate Prediction[2019].pdf
    <building>2.[DSTN]Deep Spatio-Temporal Neural Networks for Click-Through Rate Prediction[2019].pdf
    <building>3.[DTSF]Deep Time-Stream Framework for Click-Through Rate Prediction by Tracking Interest Evolution[2020].pdf

    p.s  
    1.DIEN,paper中的控制更新门并没有实际实现,实际上因为keras里面的
    我只弄了standardLstm,但CudnnLstm改动起来有点麻烦，实际上这里是是直接使
    用weight*hidden_state模型  
  
部分模型小结...  
    https://zhuanlan.zhihu.com/c_1145034612807028736  
  
p.s并不是复现,现在在家没机器,逻辑上应该问题不大,用的部分采样数据,测试模型连通,有问题的话欢迎交流.
