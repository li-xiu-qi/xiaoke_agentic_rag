from http import client
from openai import OpenAI
from dotenv import load_dotenv 
import os
import hashlib
from diskcache import Cache
from typing import List, Dict, Optional, Tuple
import json

# LOCAL_API_KEY,LOCAL_BASE_URL,LOCAL_TEXT_MODEL,LOCAL_EMBEDDING_MODEL

load_dotenv()  # 加载环境变量
local_api_key =  os.getenv('LOCAL_API_KEY')
local_base_url = os.getenv('LOCAL_BASE_URL')
local_text_model = os.getenv('LOCAL_TEXT_MODEL')
local_embedding_model = os.getenv('LOCAL_EMBEDDING_MODEL')

# 创建缓存目录
cache_dir = os.path.join(os.path.dirname(__file__), 'caches')
cache = Cache(cache_dir)

client = OpenAI(
    api_key=local_api_key,
    base_url=local_base_url,
)

def get_cache_key(text: str) -> str:
    """
    为文本生成缓存键
    :param text: 输入文本
    :return: 缓存键
    """
    return hashlib.md5(text.encode('utf-8')).hexdigest()

def batch_get_embeddings(texts: List[str], batch_size: int = 64) -> List[List[float]]:
    """
    批量获取文本的嵌入向量
    :param texts: 文本列表
    :param batch_size: 批处理大小
    :return: 嵌入向量列表
    """
    all_embeddings = []
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        response = client.embeddings.create(
            model=local_embedding_model,
            input=batch_texts
        )
        batch_embeddings = [embedding.embedding for embedding in response.data]
        all_embeddings.extend(batch_embeddings)
    
    return all_embeddings

def get_cached_embeddings(texts: List[str]) -> Tuple[List[Tuple[int, List[float]]], List[Tuple[int, str]]]:
    """
    从缓存中获取embeddings，返回已缓存的结果和未缓存的索引及文本
    :param texts: 文本列表
    :return: (已缓存的(索引,embedding)列表, 未缓存的(索引,文本)列表)
    """
    cached_results = []
    uncached_items = []
    
    for idx, text in enumerate(texts):
        cache_key = get_cache_key(text)
        cached_embedding = cache.get(cache_key)
        if cached_embedding is not None:
            cached_results.append((idx, cached_embedding))
        else:
            uncached_items.append((idx, text, cache_key))
            
    return cached_results, [(idx, text) for idx, text, _ in uncached_items], [key for _, _, key in uncached_items]

def get_text_embedding(texts: List[str]) -> List[List[float]]:
    """
    获取文本的嵌入向量，支持批次处理和缓存，保持输出顺序与输入顺序一致
    :param texts: 文本列表
    :return: 嵌入向量列表
    """
    # 1. 检查缓存并获取未缓存的项
    cached_results, uncached_items, cache_keys = get_cached_embeddings(texts)
    result_embeddings = cached_results.copy()
    
    # 2. 如果有未缓存的项，批量获取它们的embeddings
    if uncached_items:
        uncached_texts = [text for _, text in uncached_items]
        uncached_indices = [idx for idx, _ in uncached_items]
        
        # 获取新的embeddings
        new_embeddings = batch_get_embeddings(uncached_texts)
        
        # 保存到缓存并添加到结果中
        for idx, embedding, cache_key in zip(uncached_indices, new_embeddings, cache_keys):
            cache.set(cache_key, embedding)
            result_embeddings.append((idx, embedding))
    
    # 3. 按原始顺序排序并返回结果
    return [embedding for _, embedding in sorted(result_embeddings, key=lambda x: x[0])]

if __name__ == "__main__":
    # 测试获取文本嵌入向量
    texts = ["Hello, world!", "This is a test."]
    embeddings = get_text_embedding(texts)
    print("Embeddings:", embeddings)
    
    # 测试大批量文本
    large_texts = [f"Text {i}" for i in range(100)]  # 创建100个测试文本
    large_embeddings = get_text_embedding(large_texts)
    print(f"处理了 {len(large_embeddings)} 个文本的嵌入向量")
    
    # 测试缓存效果（第二次请求应该直接从缓存获取，且顺序一致）
    print("\n测试缓存效果...")
    test_texts = ["Hello, world!", "New text", "This is a test."]  # 混合缓存和非缓存的文本
    cached_embeddings = get_text_embedding(test_texts)
    print(f"处理了 {len(cached_embeddings)} 个文本的嵌入向量，保持原始顺序")
    
    
