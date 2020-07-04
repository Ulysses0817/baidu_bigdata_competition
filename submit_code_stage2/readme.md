###########系统环境
centos 6
python 3.6
pytorch 0.4
pretrainedmodels 0.7.4

###########主要结构描述
├── readme.md                   // help
├── models                      // 模型文件
│   ├── model.py               // img模型
│   ├── resnet4cifar.py        // visit模型
│   └── resnet110.th           // 预训练模型权重
├── data                        // 数据存放
├── Final_EDA.ipynb             // 数据EDA
├── Organize Visit.ipynb        // visit特征提取
├── lgb.ipynb                   // lgb模型主程序
├── config_dist.py              // 模型配置参数设置
├── main_dist.py                // DL模型主程序
├── data.py                     // 数据读取
├── Predict_Part.ipynb          // TTA预测
├── stacking.ipynb              // 模型融合
├── HazeRemoval.py              // 图像去雾算法
└── utils.py