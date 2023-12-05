# nlpa
自然语言处理分析-LSTM实现新闻分类任务
### 1. 构建词表

**`build_vocab(file_path, tokenizer, max_size, min_freq)`**

- **功能**：根据提供的文件路径和分词器构建词表。
- **步骤**：
    - 从文件中读取数据。
    - 根据所提供的分词器（词级别或字符级别）进行分词。
    - 根据词频过滤并限制词表大小。
    - 包含未知词（UNK）和填充词（PAD）。

### 2. 数据准备

**`build_dataset(config, use_word)`**

- **功能**：准备训练、开发和测试数据集。
- **步骤**：
    - 根据用户选择进行词级别或字符级别的分词。
    - 加载并预处理训练、开发和测试数据。

### 3. 数据集迭代

**`DatasetIterater(batches, batch_size, device)`**

- **功能**：数据集迭代器类。
- **作用**：生成可迭代的批次数据供模型训练使用。

### 4. 词嵌入提取

- **功能**：提取预训练词向量。
- **步骤**：
    - 加载预训练词向量文件或进行随机初始化。
    - 基于词表创建并保存词嵌入（保存为.npz格式）。

### 5. 训练与评估

- **功能**：训练模型并进行评估。
- **步骤**：
    - 根据提供的参数选择特定的模型（TextCNN、TextRNN等）。
    - 训练循环：使用交叉熵损失训练模型，定期输出训练和验证集效果。
    - 在验证集上保存效果最好的模型检查点。
    - 在测试数据上评估训练好的模型，输出指标如准确率、损失、精确度、召回率、F1值和混淆矩阵。

### 6. 模型架构与训练配置

- **包含**：模型架构及训练配置、模型初始化和配置（如TextCNN、TextRNN）、优化器配置（Adam）、学习率调度（ExponentialLR）、权重初始化（init_network 函数）。
- **工具**：使用TensorboardX进行可视化。

### TensorBoard结果图
![image](https://github.com/laneirn/LSTM-class/assets/88194633/54db7edb-f769-438c-abf6-c1d17466fd62)

![image](https://github.com/laneirn/LSTM-class/assets/88194633/65d44160-41ca-4631-ad32-4fa95982bbf5)

![image](https://github.com/laneirn/LSTM-class/assets/88194633/83006dc2-b594-43df-a2ca-7f42f6a6f98e)

![image](https://github.com/laneirn/LSTM-class/assets/88194633/a472e7d9-3545-413a-9c3d-e92dfdba2a85)
