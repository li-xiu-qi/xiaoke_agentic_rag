import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="milvus_lite")

from pymilvus import MilvusClient
from get_text_embedding import get_text_embedding


class VectorDatabase:
    """向量数据库封装类"""
    
    def __init__(self, db_path: str = "milvus_demo.db"):
        """初始化向量数据库客户端"""
        self.client = MilvusClient(db_path)
    
    def create_collection(self, collection_name: str, dimension: int = 1024, 
                         metric_type: str = "IP", index_type: str = "HNSW",
                         m: int = 16, ef_construction: int = 200, 
                         drop_if_exists: bool = True):
        """创建集合
        
        Args:
            collection_name: 集合名称
            dimension: 向量维度
            metric_type: 度量类型 (IP/L2/COSINE)
            index_type: 索引类型 (HNSW/IVF_FLAT等)
            m: HNSW参数，每个节点的最大连接数
            ef_construction: HNSW参数，构建索引时的搜索范围
            drop_if_exists: 如果存在是否删除重建
        """
        if drop_if_exists and self.client.has_collection(collection_name=collection_name):
            self.client.drop_collection(collection_name=collection_name)
        
        self.client.create_collection(
            collection_name=collection_name,
            dimension=dimension,
            metric_type=metric_type,
            index_type=index_type,
            auto_id=True,  # 启用自动ID生成
            index_params={
                "M": m,
                "efConstruction": ef_construction,
            }
        )
        print(f"创建集合 '{collection_name}' 成功")
    
    def insert_documents(self, collection_name: str, docs: list[str], metadata: list[dict] = None) -> int:
        """插入文档到集合
        
        Args:
            collection_name: 集合名称
            docs: 文档列表
            metadata: 可选的元数据列表，包含额外字段用于过滤
            
        Returns:
            插入的文档数量
        """
        # 获取文档的向量表示
        embeddings = get_text_embedding(docs)
        
        # 构建数据（不包含id，让Milvus自动生成）
        data = []
        for i, (doc, embedding) in enumerate(zip(docs, embeddings)):
            item = {
                "text": doc,
                "vector": embedding,
            }
            # 如果提供了元数据，添加到数据中
            if metadata and i < len(metadata):
                item.update(metadata[i])
            data.append(item)
        
        # 插入数据
        self.client.insert(collection_name=collection_name, data=data)
        print(f"插入了 {len(data)} 条数据到集合 '{collection_name}'")
        return len(data)
    
    def search(self, collection_name: str, query_text: str, limit: int = 3, 
               ef: int = 64, filter: str = None) -> list:
        """搜索相似文档
        
        Args:
            collection_name: 集合名称
            query_text: 查询文本
            limit: 返回结果数量
            ef: HNSW搜索参数，候选数量
            filter: 过滤条件，例如 'color like "red%" and likes > 50'
            
        Returns:
            搜索结果列表
        """
        # 获取查询文本的向量
        query_embedding = get_text_embedding([query_text])[0]
        
        # 执行搜索
        search_params = {
            "ef": ef,
        }
        
        # 构建搜索参数
        search_kwargs = {
            "collection_name": collection_name,
            "data": [query_embedding],
            "limit": limit,
            "output_fields": ["text", "category", "year", "importance"],
            "search_params": search_params
        }
        
        # 如果有过滤条件，添加到搜索参数中
        if filter:
            search_kwargs["filter"] = filter
            
        search_res = self.client.search(**search_kwargs)
        
        # 解析结果
        results = []
        for hits in search_res:
            for hit in hits:
                result = {
                    "text": hit['entity']['text'],
                    "score": hit['distance'],
                    "id": hit['id']
                }
                # 添加元数据字段（如果存在）
                for field in ['category', 'year', 'importance']:
                    if field in hit['entity']:
                        result[field] = hit['entity'][field]
                results.append(result)
        
        return results
    
    def print_search_results(self, results: list):
        """打印搜索结果"""
        print("\n搜索结果:")
        for i, result in enumerate(results, 1):
            print(f"{i}. 相似度: {result['score']:.4f}")
            print(f"   文本: {result['text']}")
            # 打印元数据
            metadata_fields = ['category', 'year', 'importance']
            metadata_info = []
            for field in metadata_fields:
                if field in result:
                    metadata_info.append(f"{field}: {result[field]}")
            if metadata_info:
                print(f"   元数据: {', '.join(metadata_info)}")
            print()


# 使用示例
if __name__ == "__main__":
    # 创建向量数据库实例
    vector_db = VectorDatabase()
    
    # 创建集合
    vector_db.create_collection("demo_collection")
    
    # 准备文档和元数据
    docs = [
        "Artificial intelligence was founded as an academic discipline in 1956.",
        "Alan Turing was the first person to conduct substantial research in AI.",
        "Born in Maida Vale, London, Turing was raised in southern England.",
    ]
    
    # 添加元数据用于过滤演示
    metadata = [
        {"category": "AI", "year": 1956, "importance": 90},
        {"category": "AI", "year": 1950, "importance": 95},
        {"category": "Biography", "year": 1912, "importance": 80},
    ]
    
    # 插入文档和元数据
    vector_db.insert_documents("demo_collection", docs, metadata)
    
    # 搜索
    query_text = "What is artificial intelligence?"
    results = vector_db.search("demo_collection", query_text)
    
    # 打印结果
    vector_db.print_search_results(results)
    
    # 过滤搜索示例
    print("\n--- 过滤搜索示例 ---")
    print("1. 搜索AI相关且重要性大于85的文档:")
    filtered_results = vector_db.search(
        "demo_collection", 
        query_text, 
        filter='category == "AI" and importance > 85'
    )
    vector_db.print_search_results(filtered_results)
    
    print("2. 搜索1950年以后的文档:")
    year_filtered_results = vector_db.search(
        "demo_collection", 
        "research", 
        filter='year > 1950'
    )
    vector_db.print_search_results(year_filtered_results)
    
    print("3. 测试文本字段过滤 - 包含'AI'关键词的文档:")
    text_filtered_results = vector_db.search(
        "demo_collection", 
        "artificial intelligence", 
        filter='text like "%AI%"'
    )
    vector_db.print_search_results(text_filtered_results)
    
    print("4. 测试文本字段过滤 - 包含'Turing'关键词的文档:")
    turing_filtered_results = vector_db.search(
        "demo_collection", 
        "person", 
        filter='text like "%Turing%"'
    )
    vector_db.print_search_results(turing_filtered_results)