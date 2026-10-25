# 这个文件只定义 Student 的网络结构，不在这里计算损失或执行训练。
# __init__：准备可学习的层；forward：规定一批文本如何依次经过这些层。
# 学习参数的更新需要训练脚本执行 loss.backward() 和 optimizer.step()。
#
# 整体数据流：
# token 编号 -> Embedding -> 双向 LSTM -> Masked Mean Pooling -> Dropout
#                                                        |-> 分类层 -> logits
#                                                        |-> 特征映射层 -> teacher_hidden
# 两个输出都由 Student 算出，真实 Teacher 的输出在网络外提供监督目标。

import torch             # 提供张量运算，例如求和 torch.sum。
import torch.nn as nn    # 提供 Embedding、LSTM、Linear 等神经网络层。



class StudentBiLSTM(nn.Module):
    # 继承 nn.Module 后，PyTorch 才能统一管理模型参数、设备和训练/评估状态。
    # 例如 model.parameters() 能找到下面定义的 Embedding、LSTM 和 Linear 的参数。

    # 参数含义：
    # vocab_size：词表大小，必须覆盖 tokenizer 可能产生的所有 token 编号。
    # num_classes：新闻类别数量，例如 10，决定每条新闻要输出几个类别分数。
    # teacher_hidden_state：Teacher 句向量的维度，是整数，不是隐藏状态张量本身。
    #     例如 Teacher 的 CLS 向量为 768 维，这里就传 768；以实际 Teacher 为准。
    # embedding_dim：Student 为每个 token 学习的词向量维度，默认 128。
    # hidden_dim：LSTM 单个方向的隐藏维度，默认 256；双向拼接后为 512。
    # num_layers：堆叠的 LSTM 层数，默认 2；与“前向/后向”两个方向是不同概念。
    # dropout：训练时随机丢弃部分特征的概率，默认 0.3，用来缓解过拟合。
    def __init__(self, vocab_size, num_classes, teacher_hidden_state,
                 embedding_dim=128, hidden_dim=256, num_layers=2, dropout=0.3):
        # 初始化父类。必须先执行这一步，再把神经网络层赋给 self.xxx。
        super(StudentBiLSTM, self).__init__()

        # 第一步：创建词嵌入层，把整数编号转换为可以学习的浮点向量。
        # 为什么：token 编号只表示词表位置，编号大小不代表语义距离。
        # Embedding 相当于一张 [vocab_size, embedding_dim] 的可学习查找表。
        # 它是 Student 自己的参数，不会因为使用 Teacher 的 tokenizer 就自动复制 Teacher 词向量。
        self.embedding = nn.Embedding(
            vocab_size,       # 查找表有多少行，即支持多少个 token 编号。
            embedding_dim,    # 每一行有多少个浮点数，即每个 token 用多少维表示。
            padding_idx=0     # 假设 PAD 编号为 0：该行初始为零，Embedding 不为其累积梯度。
                              # 注意：这里的 0 应与所用 tokenizer 的 pad_token_id 一致。
        )

        # 第二步：创建双向 LSTM，让每个位置的表示包含上下文信息。
        # 为什么：Embedding 只按编号查表；同一 token 的初始词向量不会随句子变化。
        # LSTM 会结合序列上下文，使该 token 的输出与它所在的文本有关。
        self.lstm = nn.LSTM(
            embedding_dim,    # 输入特征维度，必须接得上上一层的词向量维度。
            hidden_dim,       # 单个方向输出 hidden_dim 维。
            num_layers,       # 多层堆叠：上一层 LSTM 的输出交给下一层。
            bidirectional=True,  # 同时从左到右、从右到左编码，再拼接两个方向。
            dropout=dropout,   # 多层 LSTM 的层间 dropout；num_layers=1 时这项不生效。
            batch_first=True  # 输入和输出使用 [批大小, 序列长度, 特征维度] 的顺序。
        )

        # 第三步：准备句向量上的 Dropout，稍后在 forward 中调用。
        # 它与 LSTM 内部的“层间 dropout”不同：这里直接作用于池化后的句向量。
        # model.train() 时随机置零部分元素，并缩放保留元素；model.eval() 时不丢弃。
        self.dropout = nn.Dropout(dropout)

        # 第四步：分类头，把一条新闻的句向量转换为各类别的原始分数。
        # 输入是双向拼接结果，所以维度必须是 hidden_dim * 2，而不是 hidden_dim。
        # 例如：512 维句向量 -> 10 个类别分数。
        self.classifier = nn.Linear(hidden_dim * 2, num_classes)

        # 第五步：特征蒸馏头，把 Student 句向量映射到 Teacher 的特征维度。
        # 为什么：Student 可能为 512 维，Teacher 可能为 768 维，不能直接逐元素比较。
        # 这层学习一个映射，例如 512 -> 768，之后才能在外部计算特征损失（如 MSE）。
        # 维度对齐只是让比较成为可能；语义上的接近要靠训练损失来学习。
        self.hidden_to_teacher = nn.Linear(hidden_dim * 2, teacher_hidden_state)

    def forward(self, input_ids, attention_mask):
        # 通常通过 model(input_ids, attention_mask) 进入这个函数。
        # 为方便理解，下面统一用以下字母表示张量形状：
        # B：批大小；L：本批补齐后的序列长度；E：embedding_dim；H：hidden_dim；
        # C：num_classes；T：Teacher 句向量维度。
        #
        # input_ids：[B, L]，整数张量，来自 tokenizer 的 token 编号。
        # attention_mask：[B, L]，1 表示有效位置，0 表示补齐的 PAD。
        # 这里的有效位置也包括 mask=1 的 CLS、SEP 等特殊 token，并非只指普通词。
        # 输入张量需要由调用方放在与模型相同的设备上。

        # 步骤 1：查词向量表。
        # [B, L] -> [B, L, E]：每个整数编号变成一个 E 维向量。
        # 例如 [64, 128] -> [64, 128, 128]，最后一个 128 是词向量维度。
        embedding = self.embedding(input_ids)

        # 步骤 2：用双向 LSTM 编码整批序列。
        # [B, L, E] -> [B, L, 2H]：每个位置拼接前向和后向输出。
        # outputs 保存最后一层 LSTM 在“每个位置”的输出，接下来要对这些位置做池化。
        # 第二个返回值是 (h_n, c_n)，即各层/方向的最终隐藏状态和细胞状态。
        # 这里选择平均池化来表示句子，因此用 _ 忽略第二个返回值。
        outputs, _ = self.lstm(embedding)

        # 步骤 3：调整 mask 的形状，为逐位置屏蔽 PAD 做准备。
        # 为什么需要 mask：同一批句子长短不同，短句会补齐到相同长度。
        # 例如有效 token 是“我 / 喜欢 / 你”，补齐后为：
        # tokens：[我, 喜欢, 你, PAD, PAD]
        # mask： [ 1,   1,  1,   0,   0]
        #
        # unsqueeze(-1)：[B, L] -> [B, L, 1]，只增加一个维度，不改变原来的数值。
        # float()：把 0/1 转成浮点数，以便和 LSTM 的浮点输出相乘。
        # 相乘时，最后一维的 1 会广播到 2H，给该位置的所有特征乘同一个 mask 值。
        mask = attention_mask.unsqueeze(-1).float()

        # 步骤 4：把 PAD 位置的整个输出向量归零。
        # [B, L, 2H] * [B, L, 1] -> [B, L, 2H]。
        # 有效位置乘 1 保留；PAD 位置乘 0 排除，不让它直接参与后面的求和。
        # 注意：零词向量不代表 LSTM 输出也一定为零，因为还有偏置和递归状态。
        # 当前实现的 mask 只作用于池化，LSTM 仍然处理了 PAD；尤其双向 LSTM 中，
        # PAD 仍可能影响有效位置的上下文。严格跳过 PAD 通常还需要打包有效序列。
        masked_output = outputs * mask

        # 步骤 5：沿序列长度这一维求和，把每条新闻的有效位置向量加起来。
        # dim=1 指的是 L 这一维，而不是批次维或特征维。
        # [B, L, 2H] -> [B, 2H]：每条新闻得到一个向量，不会把不同新闻加在一起。
        sum_output = torch.sum(masked_output, dim=1)

        # 步骤 6：统计每条新闻的有效 token 数，准备计算平均值。
        # mask 为 [B, L, 1]，沿 dim=1 求和后为 [B, 1]。
        # 上面的 [1, 1, 1, 0, 0] 求和为 3，分母应为 3，而不是补齐长度 5。
        # clamp(min=1) 保证分母至少为 1，防止极端情况下整条 mask 全零导致除零。
        # 这是数值保护，不代表空文本本身就是合格的训练样本。
        valid_length = mask.sum(dim=1).clamp(min=1)

        # 步骤 7：有效位置向量之和 / 有效位置数量，得到 Masked Mean Pooling。
        # [B, 2H] / [B, 1] -> [B, 2H]：每条新闻的所有特征除以它自己的长度。
        # 为什么取平均：将不同长度的序列变成固定大小的句向量，并避免求和值
        # 仅仅因为句子更长就变大。双向编码产生的上下文信息也包含在这个句向量中。
        masked_mean_output = sum_output / valid_length

        # 步骤 8：对句向量使用 Dropout，形状保持 [B, 2H]。
        # 为什么：训练时防止模型过度依赖少数特征，提高泛化能力。
        # 下面两个输出分支共享同一次 Dropout 后的句向量。
        sentence_hidden = self.dropout(masked_mean_output)

        # 步骤 9：分类分支，[B, 2H] -> [B, C]。
        # logits 是未归一化的类别分数，可以为负数，不是概率。
        # 不在这里做 softmax：CrossEntropyLoss 接收原始 logits；软标签蒸馏则在
        # 外部结合温度计算 softmax/log_softmax。单标签预测通常取 logits.argmax(dim=1)。
        logits = self.classifier(sentence_hidden)

        # 步骤 10：特征蒸馏分支，[B, 2H] -> [B, T]。
        # 变量名 teacher_hidden 容易误解：它实际上是 Student 投影后的特征，
        # 不是这里运行 Teacher 得到的真实特征，也不是 Teacher 的参数。
        # 外部训练代码可以将它与缓存中 Teacher 最后一层的 CLS 特征进行比较，
        # 让 Student 不仅模仿分类答案，也学习 Teacher 对新闻的内部表示。
        teacher_hidden = self.hidden_to_teacher(sentence_hidden)

        # 步骤 11：返回两个张量，由外部训练代码组合不同的损失。
        # logits 可用于真实标签的分类损失，以及与 Teacher logits 比较的软标签损失；
        # teacher_hidden 可用于与真实 Teacher 特征比较的特征蒸馏损失。
        # 这些损失反向传播后，共同更新前面的 Student 网络。
        # 推理时若只需要分类结果，可以忽略第二个返回值。
        #
        # 默认 E=128、H=256、C=10，假设 B=64、L=128、T=768，则形状依次为：
        # [64,128] -> [64,128,128] -> [64,128,512] -> [64,512]
        # 最后返回 logits=[64,10]，teacher_hidden=[64,768]。
        return logits, teacher_hidden
