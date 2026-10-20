# 单条新闻预测入口：加载已训练的模型，将文本转换为输入张量，返回类别与分数。
# 这里只做推理，不训练模型，也不需要重新加载原始预训练目录。
import json
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from config import Config

# 配置提供最优模型目录和运行设备，模型目录中还保存有分词器与 metadata.json。
config = Config()


# 将模型加载与单条预测封装成一个类。
# 实例化一次后可以反复调用 predict_one，避免每预测一条文本都重新加载权重。
class BertNewsClassifierPredictor():
    def __init__(self):
        # 1. 读取训练时保存的元数据（辅助配置，不是新闻原始数据）。
        # with 会在读取完成后自动关闭文件，json.load 将 JSON 转成 Python 字典。
        with open(config.best_model_dir / 'metadata.json', 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        # 类别列表的下标就是类别 ID，其顺序必须与模型训练时保持一致。
        self.class_list = metadata['class_list']

        # 沿用训练时的最大词元长度，超过此长度的输入会被截断。
        # 这里限制的是分词后的词元数量，包含特殊词元，不是简单的中文字符数。
        self.max_len = metadata['max_length']

        # 2. 从最优模型目录加载完整分类模型，Auto 类根据 config.json 选择结构。
        self.model = AutoModelForSequenceClassification.from_pretrained(config.best_model_dir)

        # 必须使用保存时配套的分词器，让输入的词元 ID 与模型词表一致。
        self.tokenizer = AutoTokenizer.from_pretrained(config.best_model_dir)
        # 模型参数和输入张量必须处于同一设备，不能一个在 CPU、另一个在 GPU。
        self.model.to(config.device)
        # 显式切换为评估模式，关闭 Dropout 的随机丢弃行为。
        # eval() 控制模型层的行为，下面的 no_grad() 则负责关闭梯度记录。
        self.model.eval()
        self.device = config.device

    # 接收一条字符串，返回 {'pred': 类别ID, 'class': 类别名称, 'score': 概率值}。
    def predict_one(self, text):
        # 去除字符串首尾的空白，保留中间的文本内容。
        text = text.strip()

        # 3. 分词并转为模型输入：input_ids、attention_mask，以及可能的 token_type_ids。
        # truncation=True：超过 max_length 时截断。
        # padding=False：当前只预测一条文本，不需要为了对齐其他样本而补齐长度。
        # return_tensors='pt'：返回 PyTorch 张量，形状通常为 [1, 当前词元长度]。
        # 这里的 1 是批次大小，即使只有一条文本，仍保留批次维度。
        tokens = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_len,
            padding=False,
            return_tensors='pt'
        )

        # 分词器返回的 BatchEncoding 支持 .to()，会移动其中所有输入张量。
        tokens = tokens.to(self.device)
        # 4. 推理不更新参数，不需要记录反向传播的计算图，可减少内存占用。
        with torch.no_grad():
            # **tokens 将输入字典展开为关键字参数。
            # logits 是各类别的原始分数，形状为 [1, 类别数]。
            logits = self.model(**tokens).logits
            # 沿类别维度做 softmax，把分数变成概率，每条样本的类别概率之和约为 1。
            prob = torch.softmax(logits,dim=1)
            # torch.argmax 只返回最大值的下标，不能拆成 score 和 pred 两个变量。
            # torch.max(..., dim=1) 同时返回最大概率和对应的类别下标。
            # 此时 score 和 pred 的形状均为 [1]，分别对应这条文本的分数与类别 ID。
            score,pred = torch.max(prob, dim=1)
            # item() 将只有一个元素的张量转为普通 Python 数值。
            pred = pred.item()
        # 5. 用类别 ID 查询名称，并返回便于打印或写入 JSON 的普通字典。
        # score 是模型给预测类别的 softmax 概率，不代表现实中一定正确的概率。
        return {
            'pred': pred,
            'class': self.class_list[pred],
            'score': score.item()
        }
# 直接运行本文件时执行示例；被其他模块导入时不会自动加载模型、运行预测。
if __name__ == '__main__':
    predictor = BertNewsClassifierPredictor()
    text = "华为发了新的手机，新手机是很垃圾的芯片"
    result = predictor.predict_one(text)
    print(result)
