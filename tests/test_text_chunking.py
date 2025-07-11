try:
    import spacy
except ImportError:
    spacy = None
    print("警告: 未安装 spacy，将使用基础分割方法")
    print("安装命令:")
    print("  pip install spacy")
    print("  python -m spacy download zh_core_web_sm")

from typing import List, Optional
import re


class RecursiveTextSplitter:
    """递归文本分割器，使用 spaCy 进行智能分割"""
    
    def __init__(self, 
                 chunk_size: int = 800,
                 chunk_overlap: int = 0,
                 spacy_model_name: str = "zh_core_web_sm",
                 separators: Optional[List[str]] = None):
        """
        初始化递归文本分割器
        
        Args:
            chunk_size: 每个文本块的最大字符数
            chunk_overlap: 文本块之间的重叠字符数 (设为0表示无重叠)
            spacy_model_name: spaCy 模型名称
            separators: 分割符列表，按优先级排序
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # 尝试加载 spaCy 模型
        if spacy is None:
            print("spaCy 未安装，将使用基础分割方法")
            print("安装命令: pip install spacy")
            print("下载中文模型: python -m spacy download zh_core_web_sm")
            self.nlp = None
        else:
            try:
                self.nlp = spacy.load(spacy_model_name)
            except OSError:
                print(f"警告: 无法加载 spaCy 模型 '{spacy_model_name}'，将使用基础分割方法")
                print(f"请先下载模型: python -m spacy download {spacy_model_name}")
                print("或者使用其他可用模型:")
                print("  - zh_core_web_sm (小模型)")
                print("  - zh_core_web_md (中等模型)")
                print("  - zh_core_web_lg (大模型)")
                self.nlp = None
        
        # 默认分割符，按优先级排序
        if separators is None:
            self.separators = [
                "\n\n",  # 段落分割
                "\n",    # 行分割
                "。",    # 中文句号
                "！",    # 中文感叹号
                "？",    # 中文问号
                ";",     # 分号
                ":",     # 冒号
                "，",    # 中文逗号
                ",",     # 英文逗号
                " ",     # 空格
                ""       # 字符级分割
            ]
        else:
            self.separators = separators
    
    def split_text(self, text: str) -> List[str]:
        """
        递归分割文本
        
        Args:
            text: 要分割的文本
            
        Returns:
            分割后的文本块列表
        """
        if not text or not text.strip():
            return []
        
        # 如果文本长度小于等于chunk_size，直接返回
        if len(text) <= self.chunk_size:
            return [text.strip()]
        
        # 尝试使用spaCy进行智能分割
        if self.nlp:
            chunks = self._split_with_spacy(text)
            if chunks:
                return chunks
        
        # 如果spaCy分割失败，使用递归分割符方法
        return self._recursive_split(text, self.separators)
    
    def _split_with_spacy(self, text: str) -> List[str]:
        """使用 spaCy 进行智能分割"""
        try:
            doc = self.nlp(text)
            sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
            
            if not sentences:
                return []
            
            return self._process_sentences(sentences)
            
        except Exception as e:
            print(f"spaCy 分割失败: {e}")
            return []
    
    def _process_sentences(self, sentences: List[str]) -> List[str]:
        """处理句子列表，组合成合适大小的块"""
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if len(sentence) > self.chunk_size:
                chunks = self._handle_long_sentence(chunks, current_chunk, sentence)
                current_chunk = ""
            else:
                current_chunk = self._add_sentence_to_chunk(chunks, current_chunk, sentence)
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return [chunk for chunk in chunks if chunk]
    
    def _handle_long_sentence(self, chunks: List[str], current_chunk: str, sentence: str) -> List[str]:
        """处理超长句子"""
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        long_sentence_chunks = self._recursive_split(sentence, self.separators[1:])
        chunks.extend(long_sentence_chunks)
        return chunks
    
    def _add_sentence_to_chunk(self, chunks: List[str], current_chunk: str, sentence: str) -> str:
        """将句子添加到当前块"""
        if len(current_chunk + sentence) <= self.chunk_size:
            return current_chunk + sentence
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            return sentence
    
    def _recursive_split(self, text: str, separators: List[str]) -> List[str]:
        """
        递归分割文本
        
        Args:
            text: 要分割的文本
            separators: 分割符列表
            
        Returns:
            分割后的文本块列表
        """
        if not separators or separators[0] == "":
            return self._force_split(text)
        
        splits = self._split_by_separator(text, separators[0])
        return self._process_splits(splits, separators[1:])
    
    def _split_by_separator(self, text: str, separator: str) -> List[str]:
        """使用指定分割符分割文本"""
        splits = text.split(separator)
        # 重新组合分割符（除了最后一部分）
        return [split + separator for split in splits[:-1]] + [splits[-1]]
    
    def _process_splits(self, splits: List[str], remaining_separators: List[str]) -> List[str]:
        """处理分割后的文本片段"""
        chunks = []
        current_chunk = ""
        
        for split in splits:
            split = split.strip()
            if not split:
                continue
            
            if len(split) > self.chunk_size:
                chunks = self._handle_oversized_split(chunks, current_chunk, split, remaining_separators)
                current_chunk = ""
            else:
                current_chunk = self._add_split_to_chunk(chunks, current_chunk, split)
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return [chunk for chunk in chunks if chunk]
    
    def _handle_oversized_split(self, chunks: List[str], current_chunk: str, 
                               split: str, remaining_separators: List[str]) -> List[str]:
        """处理超大的分割片段"""
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        sub_chunks = self._recursive_split(split, remaining_separators)
        chunks.extend(sub_chunks)
        return chunks
    
    def _add_split_to_chunk(self, chunks: List[str], current_chunk: str, split: str) -> str:
        """将分割片段添加到当前块"""
        if len(current_chunk + split) <= self.chunk_size:
            return current_chunk + split
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            return split
    
    def _force_split(self, text: str) -> List[str]:
        """强制按字符数分割文本"""
        chunks = []
        for i in range(0, len(text), self.chunk_size):
            chunk = text[i:i + self.chunk_size]
            if chunk.strip():
                chunks.append(chunk.strip())
        return chunks


def test_recursive_text_splitter():
    """测试递归文本分割器"""
    
    # 测试文本
    test_text = """
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
    
    print("=== 递归文本分割测试 ===")
    
    # 创建分割器实例
    splitter = RecursiveTextSplitter(
        chunk_size=200,
        chunk_overlap=0,  # 不使用重叠
        spacy_model_name="zh_core_web_sm"
    )
    
    # 分割文本
    chunks = splitter.split_text(test_text)
    
    print(f"原文长度: {len(test_text)} 字符")
    print(f"分割成 {len(chunks)} 个块:")
    print("-" * 50)
    
    for i, chunk in enumerate(chunks, 1):
        print(f"块 {i} (长度: {len(chunk)}):")
        print(chunk)
        print("-" * 30)
    
    # 测试长文档分割
    print("\n=== 长文档分割测试 ===")
    long_text = test_text * 5  # 重复5次制造长文档
    
    long_splitter = RecursiveTextSplitter(
        chunk_size=500,
        chunk_overlap=0
    )
    
    long_chunks = long_splitter.split_text(long_text)
    print(f"长文档长度: {len(long_text)} 字符")
    print(f"分割成 {len(long_chunks)} 个块")
    
    for i, chunk in enumerate(long_chunks[:3], 1):  # 只显示前3个块
        print(f"块 {i} (长度: {len(chunk)}):")
        print(chunk[:100] + "..." if len(chunk) > 100 else chunk)
        print("-" * 30)


if __name__ == "__main__":
    test_recursive_text_splitter()