# 论文机制与合成实现

|论文操作|可运行入口|演示范围|
|---|---|---|
|毫米位移映射到原生源网格；一次三线性读取|src/core.py 的 sampling_grid、read、compose；check_read_interface.py|演示规则网格和给定仿射；2 mm/8 mm 例子可逐项核算。|
|共同 ED 硬种子|hard_seed、hard_labels|合成 ED 标注只用于生成点；轻量点初始化器生成硬种子。|
|源记忆只写一次，整相位提交|DAM.forward、encode_memory|ED 与最近历史相位复用，训练保留软写的时间梯度。|
|有支持的空间键、投影值缩放、指针和空键回退|weighted_attention；check_read_interface.py|由给定距离与支持比例计算 g，再经过 sigmoid、硬阈值并停止梯度；有效键和指针共同归一化。|
|图像驱动配准及三阶段训练|TinyRegistration、run_demo.py|用小型平移估计器与小解码器训练；归档演示使用连续指数近似，配置 tau_mm=1.5 与论文正式尺度2 mm分别记录。|
|因果推理与边界检查|tests/test_core.py；outputs/|11项测试含相位顺序、组合方向、空注意力、物理轴、LNCC退化窗口及时间梯度。正式实现 D81/D82 的运行记录另列在论文，两个测试集合各有来源。|

[合成示例生成器](src/data.py)
