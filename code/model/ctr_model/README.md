![image](https://img.shields.io/badge/achieve_build-14-orange)


CTR MODEL ACHIEVE:  
>[1]Interactive Model  
>>1.[[FM] Fast Context-aware Recommendations with Factorization Machines (UKON 2011).pdf](https://github.com/TIXhjq/CTR_Function/blob/master/paper/interactive/%5BFM%5D%20Fast%20Context-aware%20Recommendations%20with%20Factorization%20Machines%20(UKON%202011).pdf)  
>>2.[[PNN] Product-based Neural Networks for User Response Prediction (SJTU 2016).pdf](https://github.com/TIXhjq/CTR_Function/blob/master/paper/interactive/%5BPNN%5D%20Product-based%20Neural%20Networks%20for%20User%20Response%20Prediction%20(SJTU%202016).pdf)  
>>3.[[Deep Crossing] Deep Crossing - Web-Scale Modeling without Manually Crafted Combinatorial Features (Microsoft 2016).pdf](https://github.com/TIXhjq/CTR_Function/blob/master/paper/interactive/%5BDeep%20Crossing%5D%20Deep%20Crossing%20-%20Web-Scale%20Modeling%20without%20Manually%20Crafted%20Combinatorial%20Features%20(Microsoft%202016).pdf)  
>>4.[[Wide & Deep] Wide & Deep Learning for Recommender Systems (Google 2016).pdf](https://github.com/TIXhjq/CTR_Function/blob/master/paper/interactive/%5BWide%20%26%20Deep%5D%20Wide%20%26%20Deep%20Learning%20for%20Recommender%20Systems%20(Google%202016).pdf)  
>>5.[[DeepFM] A Factorization-Machine based Neural Network for CTR Prediction (HIT-Huawei 2017).pdf](https://github.com/TIXhjq/CTR_Function/blob/master/paper/interactive/%5BDeepFM%5D%20A%20Factorization-Machine%20based%20Neural%20Network%20for%20CTR%20Prediction%20(HIT-Huawei%202017).pdf)  
>>6.[[DCN] Deep & Cross Network for Ad Click Predictions (Stanford 2017).pdf](https://github.com/TIXhjq/CTR_Function/blob/master/paper/interactive/%5BDCN%5D%20Deep%20%26%20Cross%20Network%20for%20Ad%20Click%20Predictions%20(Stanford%202017).pdf)  
>>7.[[NFM] Neural Factorization Machines for Sparse Predictive Analytics (NUS 2017).pdf](https://github.com/TIXhjq/CTR_Function/blob/master/paper/interactive/%5BNFM%5D%20Neural%20Factorization%20Machines%20for%20Sparse%20Predictive%20Analytics%20(NUS%202017).pdf)  
>>8.[[xDeepFM] xDeepFM - Combining Explicit and Implicit Feature Interactions for Recommender Systems (USTC 2018).pdf](https://github.com/TIXhjq/CTR_Function/blob/master/paper/interactive/%5BxDeepFM%5D%20xDeepFM%20-%20Combining%20Explicit%20and%20Implicit%20Feature%20Interactions%20for%20Recommender%20Systems%20(USTC%202018).pdf)  
>>9.[[AFM] Attentional Factorization Machines - Learning the Weight of Feature Interactions via Attention Networks (ZJU 2017).pdf](https://github.com/TIXhjq/CTR_Function/blob/master/paper/interactive/%5BAFM%5D%20Attentional%20Factorization%20Machines%20-%20Learning%20the%20Weight%20of%20Feature%20Interactions%20via%20Attention%20Networks%20(ZJU%202017).pdf)  
>>10.[[AutoInt] AutoInt Automatic Feature Interaction Learning via Self-Attentive Neural Networks(CIKM 2019).pdf](https://github.com/TIXhjq/CTR_Function/blob/master/paper/interactive/%5BAutoInt%5D%20AutoInt%20Automatic%20Feature%20Interaction%20Learning%20via%20Self-Attentive%20Neural%20Networks(CIKM%202019).pdf)  
>>...Later Building...  
      
>[2]Behavior Model  
>>1.[[DIN] Deep Interest Network for Click-Through Rate Prediction (Alibaba 2018).pdf](https://github.com/TIXhjq/CTR_Function/blob/master/paper/behavior/%5BDIN%5D%20Deep%20Interest%20Network%20for%20Click-Through%20Rate%20Prediction%20(Alibaba%202018).pdf)  
>>2.[[DIEN] Deep Interest Evolution Network for Click-Through Rate Prediction (Alibaba 2019).pdf](https://github.com/TIXhjq/CTR_Function/blob/master/paper/behavior/%5BDIEN%5D%20Deep%20Interest%20Evolution%20Network%20for%20Click-Through%20Rate%20Prediction%20(Alibaba%202019).pdf)  
>>3.[[DSIN]Deep Session Interest Network for Click-Through Rate Predicti(Alibaba 2019).pdf](https://github.com/TIXhjq/CTR_Function/blob/master/paper/behavior/%5BDSIN%5DDeep%20Session%20Interest%20Network%20for%20Click-Through%20Rate%20Predicti%5B2019%5D.pdf)  
>>4.[[SeqFM]Sequence-Aware Factorization Machines(2019).pdf](https://github.com/TIXhjq/CTR_Function/blob/master/paper/behavior/%5BSeqFM%5DSequence-Aware%20Factorization%20Machines(2019).pdf)  
>>...Later Building...  
    
>[3]目前计划:  
>>1.[[MIMN]Practice on Long Sequential User Behavior Modeling for Click-Through Rate Prediction[2019].pdf](https://github.com/TIXhjq/CTR_Function/blob/master/paper/behavior/%5BMIMN%5DPractice%20on%20Long%20Sequential%20User%20Behavior%20Modeling%20for%20Click-Through%20Rate%20Prediction%5B2019%5D.pdf)  
>><building>2.[[DSTN]Deep Spatio-Temporal Neural Networks for Click-Through Rate Prediction[2019].pdf](https://github.com/TIXhjq/CTR_Function/blob/master/paper/behavior/%5BDSTN%5DDeep%20Spatio-Temporal%20Neural%20Networks%20for%20Click-Through%20Rate%20Prediction%5B2019%5D.pdf)  
>><building>3.[[DTSF]Deep Time-Stream Framework for Click-Through Rate Prediction by Tracking Interest Evolution[2020].pdf](https://github.com/TIXhjq/CTR_Function/blob/master/paper/Next%20Read/%5BDTSF%5DDeep%20Time-Stream%20Framework%20for%20Click-Through%20Rate%20Prediction%20by%20Tracking%20Interest%20Evolution%5B2020%5D.pdf)  
>>......  

    p.s  
    1.DIEN,paper中的控制更新门并没有实际实现,实际上因为keras里面的
    我只弄了standardLstm,但CudnnLstm改动起来有点麻烦，实际上这里是是直接使
    用weight*hidden_state    
    
p.s并不是复现,现在在家没机器,逻辑上应该问题不大,用的部分采样数据,测试模型连通,有问题的话欢迎交流.
