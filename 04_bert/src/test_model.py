from transformers import AutoTokenizer, AutoModelForSequenceClassification

from config import Config
from data import load_class_list, build_dataloader
from train import evaluate


def main():
    config = Config()
    class_list = load_class_list()

    # 根据保存目录中的 config.json 自动选择模型结构。
    # 训练保存的是 BERT 分类模型，不能强行使用 ALBERT 类加载。
    model = AutoModelForSequenceClassification.from_pretrained(
        config.best_model_dir, local_files_only=True
    )
    tokenizer = AutoTokenizer.from_pretrained(
        config.best_model_dir, local_files_only=True
    )
    model.to(config.device)

    # 第一个参数是测试数据的文件路径，配置对象通过 config= 单独传入。
    dataloader = build_dataloader(
        config.test_path, tokenizer, shuffle=False, config=config
    )

    # evaluate 需要类别名称来生成分类报告，返回值是按固定顺序排列的元组。
    # 最后一项是报告字符串，这里已经由 evaluate 打印，不再重复打印。
    acc, precision, recall, f1, _ = evaluate(
        model, dataloader, config.device, class_list
    )
    print(f"F1的值：{f1:.4f}")
    print(f"准确率：{acc:.4f}")
    print(f"召回率：{recall:.4f}")
    print(f"精确率：{precision:.4f}")


# 直接运行本文件时才执行测试，其他文件导入时不会自动加载模型、评估数据。
if __name__ == '__main__':
    main()
