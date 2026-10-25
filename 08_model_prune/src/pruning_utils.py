import torch
import torch.nn.utils.prune as prune

from config import config
from model import load_model_and_tokenizer


def get_query_parameters(model):
    if not hasattr(model,'bert'):
        raise TypeError('当前模型不是bert模型结构')
    parameters_to_prune = []
    for layer in model.bert.encoder.layer:
        query = layer.attention.self.query
        # 声明全局剪枝列表
        parameters_to_prune.append((query,'weight'))
    return parameters_to_prune




def global_prune(model,amount=0.3):
    weigths=get_query_parameters(model)
    prune.global_unstructured(
        weigths,
        pruning_method=prune.L1Unstructured,
        amount=amount,
    )
    return weigths

def calculate_sparsity(weights):
    total=0
    zeros=0
    for module,name in weights:
        weight=getattr(module,name).detach()
        total+=weight.numel()
        zeros+=(weight==0).sum().item()
    sparsity=zeros/max(1,total)
    return sparsity

# 固化剪枝结果
# 查看剪枝的状态，weight_origin
def remove_pruning(weights):
    # 便利所有被剪枝的参数
    for module,name in weights:
        # 判断当前是否还存在weight_org
        if hasattr(module,f"{name}_orig"):
            prune.remove(module,name)

# 查看整个模型所有矩阵权重：0占比多大比例
def cal_model_zero_ratio(model):
    total = 0
    zeros = 0
    for module in model.modules():
        if hasattr(model,'weight'):
            weight = getattr(module, 'weight')
            if not torch.is_tensor(weight):
                continue
            total += weight.numel()
            zeros += (weight == 0).sum().item()
    sparsity = zeros / max(1, total)
    return sparsity
if __name__ == '__main__':
    _,model = load_model_and_tokenizer(config.dense_model_path,10)
    print(get_query_parameters(model))