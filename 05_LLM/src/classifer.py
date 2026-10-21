"""新闻分类封装：直接运行本文件会在控制台打印一条示例的结果。"""

import json

from deepseek_client import DeepSeekClient, DeepSeekClientError


def classify_news(news):
    # 先检查类型，再调用 strip，避免列表、None 等对象触发其他类型错误。
    if not isinstance(news, str) or not news.strip():
        raise ValueError('新闻必须是非空字符串')
    news = news.strip()

    client = DeepSeekClient()
    try:
        # classify 接收标题字符串，在内部完成提示词组装、请求、JSON 解析和白名单校验。
        # 返回值已经是 {'category': ..., 'reason': ...}，不能再访问 .choices。
        result = client.classify(news)

        # 将客户端的 category 字段映射为当前封装使用的 class_name。
        # 用量属于整个 API 响应；total_tokens 是本次输入和输出的总 token 数。
        # 服务未提供用量时返回 None（打印为 JSON 的 null），不假定消耗为 0。
        return {
            'class_name': result['category'],
            'reason': result['reason'],
            'total_usage': getattr(client.last_usage, 'total_tokens', None),
        }
    finally:
        # 无论预测成功还是抛出异常，都释放底层 HTTP 客户端连接。
        client.close()


if __name__ == '__main__':
    news = '中国队在总决赛中战胜对手，获得冠军。'
    try:
        result = classify_news(news)
    except (ValueError, DeepSeekClientError) as error:
        raise SystemExit(f'分类失败：{error}') from None
    print(json.dumps(result, ensure_ascii=False, indent=2))
