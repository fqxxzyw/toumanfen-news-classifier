# 导入自动分类模型
from transformers import AutoModelForSequenceClassification
# 导入自动 Tokenizer
from transformers import AutoTokenizer


def load_model_and_tokenizer(
    model_path,
    num_classes
):
    # 加载 Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_path
    )

    # 加载已经微调好的分类模型
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        num_labels=num_classes
    )

    # 返回 Tokenizer 和模型
    return (
        tokenizer,
        model
    )
