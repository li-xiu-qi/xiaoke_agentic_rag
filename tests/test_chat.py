from http import client
from typing import Dict, List
from openai import OpenAI
from dotenv import load_dotenv 
import os  

# LOCAL_API_KEY,LOCAL_BASE_URL,LOCAL_TEXT_MODEL,LOCAL_EMBEDDING_MODEL

load_dotenv()  # 加载环境变量
local_api_key =  os.getenv('LOCAL_API_KEY')
local_base_url = os.getenv('LOCAL_BASE_URL')
local_text_model = os.getenv('LOCAL_TEXT_MODEL')
local_embedding_model = os.getenv('LOCAL_EMBEDDING_MODEL')


client = OpenAI(
    api_key=local_api_key,
    base_url=local_base_url,
)

# chat

def chat(messages: List[Dict[str, str]], tools: List[Dict] = None) -> Dict[str, any]:
    """
    与本地模型进行对话
    :param messages: 消息列表
    :param tools: 可选的工具列表
    :return: 模型响应
    """
    kwargs = {
        "model": local_text_model,
        "messages": messages,
        "temperature": 0.7,
    }
    
    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = "auto"  # 让模型自动决定是否使用工具
    
    response = client.chat.completions.create(**kwargs)
    return response


if __name__ == "__main__":
    # 定义示例工具
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_current_weather",
                "description": "获取指定城市的当前天气信息",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "城市名称，如'北京'、'上海'"
                        }
                    },
                    "required": ["location"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "calculate_investment",
                "description": "计算投资回报",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "amount": {
                            "type": "number",
                            "description": "投资金额"
                        },
                        "years": {
                            "type": "number",
                            "description": "投资年限"
                        },
                        "rate": {
                            "type": "number",
                            "description": "年化收益率（小数形式）"
                        }
                    },
                    "required": ["amount", "years", "rate"]
                }
            }
        }
    ]

    # 示例对话
    messages = [
        {"role": "system", "content": "你是一个有用的助手。"},
        {"role": "user", "content": "北京今天天气怎么样？顺便帮我计算一下投资10万元，年化收益率8%，投资3年的收益。"},
    ]
    
    result = chat(messages, tools=tools)
    print("\n=== 模型响应 ===")
    print(result.choices[0].message)
    
    # 如果有工具调用
    if hasattr(result.choices[0].message, 'tool_calls') and result.choices[0].message.tool_calls:
        print("\n=== 工具调用 ===")
        for tool_call in result.choices[0].message.tool_calls:
            print(f"工具名称: {tool_call.function.name}")
            print(f"参数: {tool_call.function.arguments}")
            print("---")


"""
=== 模型响应 ===
ChatCompletionMessage(content='', refusal=None, role='assistant', annotations=None, audio=None, function_call=None, tool_calls=[ChatCompletionMessageToolCall(id='call_e61a3c08-c2b3-4987-8e9d-2f73af321b2d', function=Function(arguments='{"location": "北京"}', name='get_current_weather'), type='function'), ChatCompletionMessageToolCall(id='call_e61a3c08-c2b3-4987-8e9d-2f73af321b2d', function=Function(arguments='{"amount": 100000, "rate": 0.08, "years": 3}', name='calculate_investment'), type='function')])

=== 工具调用 ===
工具名称: get_current_weather
参数: {"location": "北京"}
---
工具名称: calculate_investment
参数: {"amount": 100000, "rate": 0.08, "years": 3}
---
"""