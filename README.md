# anime_cls
对于二次元图像数据集 [GochiUsa_Faces](https://www.kaggle.com/datasets/rignak/gochiusa-faces) 进行分类

## 数据集准备
* 原始数据集图片尺寸不一，已经使用 `python` 编写脚本 `wk.py` 将所有图片缩小为 `64 x 64 x 3`
* 运行 `download.bat` 即可获得修饰后的数据集

## 运行

仓库代码是为了辅助论文数据分析而写，所以可能不太会呈现出完整项目的模式

* 运行 `main.py` 即可

## 代码文件说明

```
-main.py : 项目主入口，执行即可
-api
	--dl_models.py : 神经网络方法
	--ml_models.py : 传统学习方法
	--Trained : 神经网络预训练参数文件
-datasets_
	--data : 处理过后的数据集
	--data_utils.py : 处理数据集相关的函数
	--wk.py : 用于将数据处理为 32 x 32 的脚本
	--download.bat 下载数据集的脚本
-img : 存储论文中的一些图片
-report : 生成报告的 tex 文件
-result_fig : 存储实验过程需要的一些图片
-requirements.txt : python 依赖库
