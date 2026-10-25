# 导入 Hugging Face 自动模型
from transformers import AutoModelForSequenceClassification
# 导入 Hugging Face 自动 Tokenizer
from transformers import AutoTokenizer


def load_teacher(
    model_path,
    num_classes
):
    # 从 Teacher 目录加载 Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_path
    )

    # 从 Teacher 目录加载分类模型
    teacher = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        num_labels=num_classes
    )

    # Teacher 在蒸馏时不参与参数更新
    teacher.eval()

    # 返回 Tokenizer 和 Teacher
    return (
        tokenizer,
        teacher
    )