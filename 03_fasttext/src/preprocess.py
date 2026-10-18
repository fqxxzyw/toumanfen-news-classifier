import jieba
from config import Config

conf = Config()

def load_label(class_path):

    id_to_name={}

    with open(class_path,'r',encoding='utf-8') as f:
        for idx ,line in enumerate(f):
            id_to_name[idx]=line.strip()

    return id_to_name

def preprocess_text(text,mode='word'):
    text = text.strip()
    if mode == 'char':
        tokens = list(text)
    elif mode == 'word':
        tokens = jieba.lcut(text)
    else:
        raise ValueError('Invalid mode. Supported modes: "char","word"')

    tokens = [token for token in tokens if token.strip() != '']

    process_text = ' '.join(tokens)
    return process_text

def build_fasttest_data(input_path,out_path,label_map,mode='word'):
    with open(input_path,'r',encoding='utf-8') as input_path,open(out_path,'a',encoding='utf-8') as out_path:
        for line in input_path:
            line = line.strip()
            if not line:
                continue
            text,label = line.split('\t')
            label_id = int(label)
            class_name = label_map[label_id]
            build_ = f"__label__{class_name}"

if __name__ == '__main__':
    print(load_label(conf.class_path))
    print(preprocess_text('这是一个测试文本。',mode='char'))
    preprocess_text('这是一个测试文本。',mode='word')
