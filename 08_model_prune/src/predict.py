# 导入 PyTorch
import tests
import torch
# 导入配置
from config import config, device
# 导入类别名称
from dataset import load_classes
# 导入模型
from model import load_model_and_tokenizer

def predict(texts):
    tokenizer, model =load_model_and_tokenizer(
        config.pruned_model_path,
        config.num_classes
    )
    # 把模型放到设备
    model = model.to(device)

    # 切换到评估模式
    model.eval()

    # 批量 Tokenizer
    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=config.max_length,
        return_tensors="pt"
    )


    with torch.inference_mode():
        outputs = model(
            input_ids=encoded[
                "input_ids"
            ].to(device),
            attention_mask=encoded[
                "attention_mask"
            ].to(device),
            return_dict=True
        )

        # 计算类别概率
        probs = torch.softmax(
            outputs.logits,
            dim=-1
        )

        preds = torch.argmax(
            probs,
            dim=-1
        )
    # 加载类别名称
    classes = load_classes(
        config.class_path
    )

    # 创建结果
    results = []

    # 遍历每条文本
    for i, test in enumerate(
            texts
    ):
        # 获取类别ID
        pred_id = int(
            preds[i].item()
        )

        # 获取类别名称
        pred_name= (
            classes[pred_id] if pred_id < len(classes) else str(pred_id)
        )

        confidence = float(
            probs[i][pred_id].item()
        )
        results.append({
            "text": test,
            "label_id": pred_id,
            "label_name": pred_name,
            "confidence": confidence
        })
    return results

def mian():
    tests = [
        "球队在决赛中夺取冠军",
        "我最喜欢的电影是《肖申克的救赎》",
        "今天天气晴朗",
        "我最喜欢的季节是夏天",
        "我最喜欢的数字是123"
    ]
    results = predict(tests)
    for result in results:
        print(result)

if __name__ == '__main__':
    mian()