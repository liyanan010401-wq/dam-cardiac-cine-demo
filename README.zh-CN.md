# DAM 合成演示

`data/sample.npz` 含5组固定种子生成的几何体和噪声，三个用于训练、一个保留开发、一个用于测试。每组四个相位，空间大小为 4×12×12。图像、标签、ED点、仿射和划分随文件保存，生成规则见 [数据卡](data/DATA_CARD.md) 和 [生成器](src/data.py)。该数据用于检验代码路径。

## 复现

在本目录运行：

```text
python -m pip install -r requirements.txt
python check_read_interface.py
python run_demo.py --output outputs
python -m unittest discover -s tests -v
```

运行 `check_read_interface.py` 可验证层距 8 mm、法向位移 2 mm 对应的 1/4 层插值，并按正式算子从合成的往返距离和支持比例求出空间权重。在完全支持下，零残差的权重为 0.73106，2.35482 mm 临界处为 0.5；无支持时权重为零。脚本另验部分支持、阈值两侧，以及有效键保留、仅指针和整个记忆库为空的回退。

`run_demo.py` 在 CPU 上实际训练一个简化的配准、记忆与解码网络，按相位前向传播并输出 ES 预测。该网络沿用存档的连续指数可靠性近似，采用小型编码器与图像条件平移场。它用于复现流程中的可微读写与梯度传递。正式模型的 SAM/SACB 权重、患者影像及正式临床预测由研究方管理。运行后，指标、预测和轻量权重保存在 `outputs/`。

`python -m src.data` 可重新生成合成输入。训练脚本默认单线程 CPU；文件存在时复用配套的合成数据。


## 数据入口

[合成示例](data/sample.npz) · [ACDC 官方数据集](https://www.creatis.insa-lyon.fr/Challenge/acdc/databases.html) · [M&Ms 原版挑战](https://www.ub.edu/mnms/)
