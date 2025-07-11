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
    
    def __init__(self, collection_name: str = "agentic_rag", max_iterations: int = 2):
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
        self.query_history = set()  # 记录已使用的查询语句
        
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
        self.query_history.add(query)  # 记录查询历史
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
            反思结果字典，包含改进建议和搜索策略
        """
        
        reflection_prompt = f"""
        请评估以下回答的完整性，并以JSON格式返回结果：

        用户问题：{query}
        当前回答：{answer}
        已检索文档数：{len(search_results)}

        请评估：
        1. 回答是否完整回答了用户问题
        2. 如果不完整，需要搜索什么信息来补充
        3. 生成具体的语义搜索查询语句（不是关键词）

        请严格按照以下JSON格式返回：
        {{
            "is_complete": true/false,
            "missing_info": "如果不完整，描述缺少什么信息",
            "search_queries": ["具体的搜索查询语句1", "具体的搜索查询语句2"]
        }}

        注意：
        - search_queries应该是完整的问句或描述，不是单个关键词
        - 只有当回答明显不足时才设置is_complete为false
        - 最多生成3个搜索查询
        """
        
        messages = [
            {"role": "system", "content": "你是一个专业的问答质量分析师。请严格按照要求的JSON格式输出，不要添加任何额外的文本。"},
            {"role": "user", "content": reflection_prompt}
        ]
        
        response = chat(messages)
        reflection_text = response.choices[0].message.content.strip()
        
        # 清理可能的markdown代码块标记
        if reflection_text.startswith('```json'):
            reflection_text = reflection_text[7:]  # 移除 ```json
        elif reflection_text.startswith('```'):
            reflection_text = reflection_text[3:]   # 移除 ```
            
        if reflection_text.endswith('```'):
            reflection_text = reflection_text[:-3]  # 移除结尾的 ```
            
        reflection_text = reflection_text.strip()
        
        # 解析JSON响应
        try:
            reflection = json.loads(reflection_text)
            
            # 验证JSON结构
            if not isinstance(reflection.get('is_complete'), bool):
                raise ValueError("is_complete must be boolean")
            
            if not isinstance(reflection.get('search_queries'), list):
                reflection['search_queries'] = []
                
        except (json.JSONDecodeError, Exception) as e:
            print(f"JSON解析错误: {e}")
            print(f"原始响应: {reflection_text}")
            # 提供默认结构
            reflection = {
                "is_complete": False,
                "missing_info": "无法解析反思结果",
                "search_queries": []
            }
        
        print("\n=== 反思结果 ===")
        print(f"回答是否完整: {reflection['is_complete']}")
        print(f"缺少信息: {reflection.get('missing_info', 'N/A')}")
        print(f"建议搜索查询: {reflection['search_queries']}")
        
        return reflection
    
    def refined_search(self, reflection: Dict, previous_results: List[Dict]) -> List[Dict]:
        """
        基于反思结果进行精细化搜索
        
        Args:
            reflection: 反思结果
            previous_results: 之前的搜索结果
            
        Returns:
            新的搜索结果
        """
        search_queries = reflection.get('search_queries', [])
        previous_texts = {r['text'] for r in previous_results}
        all_new_results = []
        
        print("\n=== 精细化搜索 ===")
        
        if not search_queries:
            print("没有建议的搜索查询，跳过精细化搜索")
            return []
        
        # 过滤掉已经使用过的查询
        new_queries = []
        for query in search_queries:
            if query and query not in self.query_history:
                new_queries.append(query)
                self.query_history.add(query)
            else:
                print(f"跳过重复查询: {query}")
        
        if not new_queries:
            print("所有建议的查询都已使用过，跳过搜索")
            return []
        
        # 对每个新查询进行语义搜索
        for query in new_queries:
            print(f"语义搜索: {query}")
            new_results = self.db.search(self.collection_name, query, limit=3)
            
            # 去重：移除与之前结果重复的内容
            unique_results = []
            for result in new_results:
                if result['text'] not in previous_texts:
                    unique_results.append(result)
                    previous_texts.add(result['text'])
            
            all_new_results.extend(unique_results)
            print(f"  找到 {len(unique_results)} 个新文档")
        
        print(f"总共找到 {len(all_new_results)} 个新的相关文档")
        return all_new_results
    
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
        
        # 重置查询历史
        self.query_history.clear()
        
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
            if reflection.get('is_complete', False):
                print("反思结果显示回答已完整，停止迭代")
                break
            
            # 进行精细化搜索
            new_results = self.refined_search(reflection, all_search_results)
            
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
    agentic_rag = AgenticRAG(collection_name="agentic_demo", max_iterations=2)
    
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
        语音助手、机器翻译、游戏AI等。在医疗领域，AI的具体应用包括：医学影像分析、疾病诊断、
        药物发现、个性化治疗方案、手术机器人、健康监测、电子病历分析、基因分析等。
        随着技术的不断发展，AI正在改变各个行业的工作方式，提高效率，创造新的商业模式。
        同时，AI的发展也带来了一些挑战，如就业影响、隐私保护、算法偏见等问题，需要社会各界共同关注和解决。
        """,
        """
        人工智能在医疗健康领域的应用正在快速发展。具体应用包括：
        1. 医学影像分析：利用深度学习技术分析X光、CT、MRI等医学影像，辅助医生诊断肿瘤、骨折等疾病。
        2. 药物研发：通过机器学习算法加速新药发现过程，预测药物分子的有效性和安全性。
        3. 个性化治疗：基于患者的基因信息、病史和生理数据，为患者制定个性化的治疗方案。
        4. 健康监测：利用可穿戴设备和传感器，实时监测患者的生命体征和健康状况。
        5. 手术机器人：协助医生进行精密手术，提高手术精度和安全性。
        6. 电子病历分析：自动提取和分析电子病历中的关键信息，辅助临床决策。
        这些应用大大提高了医疗服务的质量和效率，为患者提供更好的医疗体验。
        """
    ]
    
    # 准备元数据
    metadata = [
        {"category": "AI基础", "topic": "人工智能定义", "difficulty": "初级"},
        {"category": "机器学习", "topic": "机器学习概述", "difficulty": "中级"},
        {"category": "深度学习", "topic": "深度学习技术", "difficulty": "高级"},
        {"category": "NLP", "topic": "自然语言处理", "difficulty": "中级"},
        {"category": "计算机视觉", "topic": "图像处理", "difficulty": "中级"},
        {"category": "AI应用", "topic": "应用场景", "difficulty": "初级"},
        {"category": "医疗AI", "topic": "医疗健康应用", "difficulty": "中级"}
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
