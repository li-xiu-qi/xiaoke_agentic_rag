from typing import List, Dict, Optional, Tuple
import json
from recursive_text_splitter import RecursiveTextSplitter
from knowledge_database import VectorDatabase
from chat import chat


class AgenticRAG:
    """
    具有反思能力的智能RAG系统
    能够根据回答质量进行反思，并重新搜索相关信息
    """
    
    def __init__(self, collection_name: str = "agentic_rag", max_iterations: int = 3):
        """
        初始化Agentic RAG系统
        
        Args:
            collection_name: 向量数据库集合名称
            max_iterations: 最大迭代次数
        """
        self.db = VectorDatabase()
        self.collection_name = collection_name
        self.max_iterations = max_iterations
        self.conversation_history = []
        
    def setup_knowledge_base(self, documents: List[str], metadata: List[Dict] = None):
        """
        设置知识库
        
        Args:
            documents: 文档列表
            metadata: 可选的元数据列表
        """
        # 创建集合
        self.db.create_collection(self.collection_name, dimension=1024, drop_if_exists=True)
        
        # 文本分割
        splitter = RecursiveTextSplitter(chunk_size=500)
        all_chunks = []
        all_metadata = []
        
        for i, doc in enumerate(documents):
            chunks = splitter.split_text(doc)
            all_chunks.extend(chunks)
            
            # 为每个chunk添加元数据
            for chunk in chunks:
                chunk_metadata = {"doc_id": i, "chunk_text": chunk}
                if metadata and i < len(metadata):
                    chunk_metadata.update(metadata[i])
                all_metadata.append(chunk_metadata)
        
        # 插入向量数据库
        self.db.insert_documents(self.collection_name, all_chunks, all_metadata)
        print(f"知识库设置完成，共插入 {len(all_chunks)} 个文档块")
    
    def initial_search(self, query: str, limit: int = 5) -> List[Dict]:
        """
        初始搜索
        
        Args:
            query: 用户查询
            limit: 返回结果数量
            
        Returns:
            搜索结果列表
        """
        results = self.db.search(self.collection_name, query, limit=limit)
        print("\n=== 初始搜索 ===")
        print(f"查询: {query}")
        print(f"找到 {len(results)} 个相关文档")
        return results
    
    def generate_initial_answer(self, query: str, search_results: List[Dict]) -> str:
        """
        基于初始搜索结果生成回答
        
        Args:
            query: 用户查询
            search_results: 搜索结果
            
        Returns:
            生成的回答
        """
        context = "\n".join([f"文档{i+1}: {r['text']}" for i, r in enumerate(search_results)])
        
        messages = [
            {"role": "system", "content": "你是一个专业的AI助手。请基于提供的文档内容回答用户问题。重要提醒：请严格基于提供的参考文档回答，不要捏造或编造文档中不存在的信息。如果文档内容不足以完全回答问题，请明确指出需要更多信息的方面，不要进行推测或假设。"},
            {"role": "user", "content": f"基于以下文档内容回答问题：\n\n文档内容：\n{context}\n\n问题：{query}"}
        ]
        
        response = chat(messages)
        answer = response.choices[0].message.content
        print("\n=== 初始回答 ===")
        print(answer)
        return answer
    
    def reflect_on_answer(self, query: str, answer: str, search_results: List[Dict]) -> Dict:
        """
        对回答进行反思和评估
        
        Args:
            query: 原始查询
            answer: 生成的回答
            search_results: 搜索结果
            
        Returns:
            反思结果字典，包含质量评分、问题分析和改进建议
        """
        context_summary = f"共检索到{len(search_results)}个文档片段"
        
        reflection_prompt = f"""
        请对以下问答进行反思和评估：

        用户问题：{query}
        
        检索上下文：{context_summary}
        
        生成的回答：{answer}

        请从以下几个维度进行评估并给出JSON格式的结果：
        1. 回答质量评分 (1-10分)
        2. 回答是否充分解决了问题
        3. 回答是否严格基于检索到的文档内容，没有捏造不存在的信息
        4. 是否需要更多信息
        5. 如果需要改进，应该搜索什么关键词
        6. 问题分析和改进建议

        返回格式：
        {{
            "quality_score": 评分(1-10),
            "is_sufficient": true/false,
            "is_factual": true/false,
            "needs_more_info": true/false,
            "suggested_keywords": ["关键词1", "关键词2"],
            "analysis": "问题分析",
            "improvement_suggestions": "改进建议"
        }}
        """
        
        messages = [
            {"role": "system", "content": "你是一个专业的AI回答质量评估专家。请客观评估回答质量并提供改进建议。特别关注回答是否严格基于提供的检索内容，是否存在捏造或编造信息的情况。"},
            {"role": "user", "content": reflection_prompt}
        ]
        
        response = chat(messages)
        reflection_text = response.choices[0].message.content
        
        try:
            # 尝试解析JSON
            reflection = json.loads(reflection_text)
        except json.JSONDecodeError:
            # 如果JSON解析失败，提供默认结构
            reflection = {
                "quality_score": 5,
                "is_sufficient": False,
                "is_factual": True,
                "needs_more_info": True,
                "suggested_keywords": [query],
                "analysis": "无法解析反思结果",
                "improvement_suggestions": "建议重新搜索"
            }
        
        print("\n=== 反思结果 ===")
        print(f"质量评分: {reflection.get('quality_score', 'N/A')}/10")
        print(f"回答是否充分: {reflection.get('is_sufficient', 'N/A')}")
        print(f"事实准确性: {reflection.get('is_factual', 'N/A')}")
        print(f"需要更多信息: {reflection.get('needs_more_info', 'N/A')}")
        print(f"建议搜索关键词: {reflection.get('suggested_keywords', [])}")
        print(f"分析: {reflection.get('analysis', 'N/A')}")
        
        return reflection
    
    def refined_search(self, original_query: str, reflection: Dict, previous_results: List[Dict]) -> List[Dict]:
        """
        基于反思结果进行精细化搜索
        
        Args:
            original_query: 原始查询
            reflection: 反思结果
            previous_results: 之前的搜索结果
            
        Returns:
            新的搜索结果
        """
        suggested_keywords = reflection.get('suggested_keywords', [])
        
        # 构建新的搜索查询
        if suggested_keywords:
            new_query = f"{original_query} {' '.join(suggested_keywords)}"
        else:
            new_query = original_query
        
        # 执行新搜索
        new_results = self.db.search(self.collection_name, new_query, limit=8)
        
        # 去重：移除与之前结果重复的内容
        previous_texts = {r['text'] for r in previous_results}
        unique_results = [r for r in new_results if r['text'] not in previous_texts]
        
        print("\n=== 精细化搜索 ===")
        print(f"新查询: {new_query}")
        print(f"找到 {len(new_results)} 个结果，其中 {len(unique_results)} 个是新内容")
        
        return unique_results
    
    def generate_improved_answer(self, query: str, all_search_results: List[Dict], iteration: int) -> str:
        """
        基于所有搜索结果生成改进的回答
        
        Args:
            query: 用户查询
            all_search_results: 所有搜索结果
            iteration: 当前迭代次数
            
        Returns:
            改进的回答
        """
        context = "\n".join([f"文档{i+1}: {r['text']}" for i, r in enumerate(all_search_results)])
        
        messages = [
            {"role": "system", "content": f"你是一个专业的AI助手。这是第{iteration}次迭代优化。请基于提供的所有文档内容给出最全面、准确的回答。重要提醒：请严格基于提供的参考文档回答，不要捏造或编造文档中不存在的信息，不要进行无根据的推测或假设。"},
            {"role": "user", "content": f"基于以下所有文档内容回答问题：\n\n文档内容：\n{context}\n\n问题：{query}"}
        ]
        
        response = chat(messages)
        improved_answer = response.choices[0].message.content
        print(f"\n=== 第{iteration}次改进回答 ===")
        print(improved_answer)
        return improved_answer
    
    def query(self, user_query: str) -> Dict:
        """
        执行Agentic RAG查询流程
        
        Args:
            user_query: 用户查询
            
        Returns:
            包含最终答案和处理过程的字典
        """
        print(f"\n{'='*60}")
        print("开始Agentic RAG查询流程")
        print(f"用户问题: {user_query}")
        print(f"{'='*60}")
        
        # 1. 初始搜索和回答
        search_results = self.initial_search(user_query)
        current_answer = self.generate_initial_answer(user_query, search_results)
        
        all_search_results = search_results.copy()
        iteration_history = []
        
        # 2. 迭代改进流程
        for iteration in range(1, self.max_iterations + 1):
            print(f"\n--- 第{iteration}次迭代 ---")
            
            # 反思当前回答
            reflection = self.reflect_on_answer(user_query, current_answer, all_search_results)
            
            # 记录迭代历史
            iteration_info = {
                "iteration": iteration,
                "answer": current_answer,
                "reflection": reflection,
                "search_results_count": len(all_search_results)
            }
            iteration_history.append(iteration_info)
            
            # 检查是否需要继续改进
            quality_score = reflection.get('quality_score', 0)
            is_sufficient = reflection.get('is_sufficient', False)
            
            if quality_score >= 8 and is_sufficient:
                print(f"回答质量达到要求 (评分: {quality_score}/10)，停止迭代")
                break
            
            if not reflection.get('needs_more_info', True):
                print("反思结果显示不需要更多信息，停止迭代")
                break
            
            # 进行精细化搜索
            new_results = self.refined_search(user_query, reflection, all_search_results)
            
            if not new_results:
                print("没有找到新的相关信息，停止迭代")
                break
            
            # 合并搜索结果
            all_search_results.extend(new_results)
            
            # 生成改进的回答
            current_answer = self.generate_improved_answer(user_query, all_search_results, iteration)
        
        # 3. 返回最终结果
        final_result = {
            "query": user_query,
            "final_answer": current_answer,
            "total_search_results": len(all_search_results),
            "iterations": len(iteration_history),
            "iteration_history": iteration_history,
            "all_search_results": all_search_results
        }
        
        print(f"\n{'='*60}")
        print("Agentic RAG查询完成")
        print(f"总迭代次数: {len(iteration_history)}")
        print(f"最终检索文档数: {len(all_search_results)}")
        print(f"{'='*60}")
        
        return final_result


def build_agentic_rag_demo():
    """构建和测试Agentic RAG系统"""
    
    # 创建Agentic RAG实例
    agentic_rag = AgenticRAG(collection_name="agentic_demo", max_iterations=3)
    
    # 准备知识库文档
    documents = [
        """
        人工智能（Artificial Intelligence，简称AI）是计算机科学的一个分支，它企图了解智能的实质，
        并生产出一种新的能以人类智能相似的方式做出反应的智能机器。该领域的研究包括机器人、
        语言识别、图像识别、自然语言处理和专家系统等。人工智能从诞生以来，理论和技术日益成熟，
        应用领域也不断扩大。
        """,
        """
        机器学习是人工智能的一个重要分支。它是一种通过算法解析数据、从中学习，然后对真实世界
        中的事件做出决策和预测的方法。与传统的为解决特定任务、硬编码的软件程序不同，机器学习
        是用大量的数据来"训练"，通过各种算法从数据中学习如何完成任务。常见的机器学习算法包括
        线性回归、决策树、随机森林、支持向量机和神经网络等。
        """,
        """
        深度学习是机器学习的一个子集，它模拟人脑的神经网络结构，通过多层神经网络来学习数据的表示。
        深度学习在图像识别、语音识别、自然语言处理等领域取得了突破性进展。卷积神经网络（CNN）
        主要用于图像处理，循环神经网络（RNN）和长短期记忆网络（LSTM）主要用于序列数据处理，
        而Transformer架构则在自然语言处理领域带来了革命性变化。
        """,
        """
        自然语言处理（Natural Language Processing，简称NLP）是人工智能的一个重要应用领域，
        它研究如何让计算机理解、处理和生成人类语言。NLP的主要任务包括文本分类、情感分析、
        机器翻译、问答系统、文本摘要等。近年来，基于Transformer的大型语言模型如GPT、BERT
        等的出现，大大推动了NLP技术的发展。
        """,
        """
        计算机视觉是人工智能的另一个重要分支，它致力于让计算机能够"看懂"图像和视频。
        计算机视觉的主要任务包括图像分类、目标检测、图像分割、人脸识别等。深度学习技术，
        特别是卷积神经网络的发展，使得计算机视觉在许多任务上达到甚至超越了人类的表现。
        """,
        """
        人工智能的应用领域非常广泛，包括但不限于：医疗诊断、金融风控、自动驾驶、智能推荐、
        语音助手、机器翻译、游戏AI等。随着技术的不断发展，AI正在改变各个行业的工作方式，
        提高效率，创造新的商业模式。同时，AI的发展也带来了一些挑战，如就业影响、隐私保护、
        算法偏见等问题，需要社会各界共同关注和解决。
        """
    ]
    
    # 准备元数据
    metadata = [
        {"category": "AI基础", "topic": "人工智能定义", "difficulty": "初级"},
        {"category": "机器学习", "topic": "机器学习概述", "difficulty": "中级"},
        {"category": "深度学习", "topic": "深度学习技术", "difficulty": "高级"},
        {"category": "NLP", "topic": "自然语言处理", "difficulty": "中级"},
        {"category": "计算机视觉", "topic": "图像处理", "difficulty": "中级"},
        {"category": "AI应用", "topic": "应用场景", "difficulty": "初级"}
    ]
    
    # 设置知识库
    agentic_rag.setup_knowledge_base(documents, metadata)
    
    # 测试查询
    test_queries = [
        "什么是人工智能？它有哪些主要应用领域？",
        "深度学习和机器学习有什么区别？",
        "人工智能在医疗领域有哪些具体应用？"
    ]
    
    # 执行测试
    for i, query in enumerate(test_queries, 1):
        print(f"\n\n🔍 测试查询 {i}: {query}")
        agentic_rag.query(query)


if __name__ == "__main__":
    build_agentic_rag_demo()
