CTR MODEL ACHIEVE:
  [1]Interactive Model
    1.FM
    2.PNN
    3.DeepCross
    4.Wide_Deep
    5.DeepFM
    6.DCN
    7.NFM
    8.XDeepFM
    9.AFM
    10.AutoInt
    ...Later Building...
    
  [2]Behavior Model
    1.DIN
    2.DIEN
    3.DSIN
    4.SeqFM
    ...Later Building...

    p.s
        1.DIEN,paper中的控制更新门并没有实际实现,实际上因为keras里面的
        我只弄了standardLstm,但CudnnLstm改动起来有点麻烦，实际上这里是是直接使
        用weight*hidden_state模型

部分模型小结...
    https://zhuanlan.zhihu.com/c_1145034612807028736

p.s并不是复现,现在在家没机器,逻辑上应该问题不大,用的部分采样数据,测试模型连通,有问题的话欢迎交流.
