from recursive_text_splitter import RecursiveTextSplitter
from knowledge_database import VectorDatabase
from chat import chat

def build_rag_demo():
    # 1. 文本分割
    text = """
    人工智能（Artificial Intelligence，简称AI）是计算机科学的一个分支，它企图了解智能的实质，
    并生产出一种新的能以人类智能相似的方式做出反应的智能机器。该领域的研究包括机器人、
    语言识别、图像识别、自然语言处理和专家系统等。
    人工智能从诞生以来，理论和技术日益成熟，应用领域也不断扩大。可以设想，未来人工智能
    带来的科技产品，将会是人类智慧的"容器"。人工智能可以对人的意识、思维的信息过程的模拟。
    人工智能不是人的智能，但能像人那样思考、也可能超过人的智能。
    机器学习是人工智能的一个重要分支。它是一种通过算法解析数据、从中学习，然后对真实世界
    中的事件做出决策和预测的方法。与传统的为解决特定任务、硬编码的软件程序不同，机器学习
    是用大量的数据来"训练"，通过各种算法从数据中学习如何完成任务。
    """
    splitter = RecursiveTextSplitter(chunk_size=200)
    chunks = splitter.split_text(text)

    # 2. 向量数据库存储
    db = VectorDatabase()
    collection_name = "rag_demo"
    db.create_collection(collection_name, dimension=1024, drop_if_exists=True)
    db.insert_documents(collection_name, chunks)

    # 3. 用户提问
    query = "人工智能有哪些主要研究方向？"
    results = db.search(collection_name, query, limit=3)
    context = "\n".join([r["text"] for r in results])
    print("\n=== 检索到的内容 ===")
    for i, result in enumerate(results, 1):
        print(f"{i}. 相似度: {result['score']:.4f}")
        print(f"   文本: {result['text']}")
        print()
    # 4. 构造prompt并调用本地模型
    messages = [
        {"role": "system", "content": "你是一个有用的AI助手，请结合检索到的内容回答用户问题。重要提醒：请严格基于提供的参考信息回答问题，不要捏造或编造参考信息中不存在的内容。如果参考信息不足以回答问题，请明确说明需要更多信息。"},
        {"role": "user", "content": f"已检索内容：\n{context}\n\n问题：{query}"}
    ]
    response = chat(messages)
    print("\n=== RAG模型回答 ===")
    print(response.choices[0].message.content)

if __name__ == "__main__":
    build_rag_demo()
