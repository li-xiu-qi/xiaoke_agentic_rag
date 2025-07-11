try:
    import spacy
except ImportError:
    spacy = None
    print("警告: 未安装 spacy，将使用基础分割方法")
    print("安装命令:")
    print("  pip install spacy")
    print("  python -m spacy download zh_core_web_sm")

from typing import List, Optional

class SpacyTextSplitter:
    """
    使用 spaCy 进行智能分割的文本分割器。
    """
    def __init__(self, chunk_size: int = 800, spacy_model_name: str = "zh_core_web_sm"):
        self.chunk_size = chunk_size
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

    def split_text(self, text: str) -> List[str]:
        if not text or not text.strip():
            return []
        if len(text) <= self.chunk_size:
            return [text.strip()]
        if self.nlp:
            return self._split_with_spacy(text)
        else:
            return [text.strip()]

    def _split_with_spacy(self, text: str) -> List[str]:
        try:
            doc = self.nlp(text)
            sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
            if not sentences:
                return []
            return self._process_sentences(sentences)
        except Exception as e:
            print(f"spaCy 分割失败: {e}")
            return [text.strip()]

    def _process_sentences(self, sentences: List[str]) -> List[str]:
        chunks = []
        current_chunk = ""
        for sentence in sentences:
            if len(sentence) > self.chunk_size:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = ""
                # 长句直接分割为多个块
                for i in range(0, len(sentence), self.chunk_size):
                    chunk = sentence[i:i + self.chunk_size]
                    if chunk.strip():
                        chunks.append(chunk.strip())
            else:
                if len(current_chunk + sentence) <= self.chunk_size:
                    current_chunk += sentence
                else:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = sentence
        if current_chunk:
            chunks.append(current_chunk.strip())
        return [chunk for chunk in chunks if chunk]


def test_spacy_text_splitter():
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
    print("=== spaCy智能分割测试 ===")
    splitter = SpacyTextSplitter(chunk_size=200)
    chunks = splitter.split_text(test_text)
    print(f"原文长度: {len(test_text)} 字符")
    print(f"分割成 {len(chunks)} 个块:")
    print("-" * 50)
    for i, chunk in enumerate(chunks, 1):
        print(f"块 {i} (长度: {len(chunk)}):")
        print(chunk)
        print("-" * 30)
    print("\n=== 长文档分割测试 ===")
    long_text = test_text * 5
    long_splitter = SpacyTextSplitter(chunk_size=500)
    long_chunks = long_splitter.split_text(long_text)
    print(f"长文档长度: {len(long_text)} 字符")
    print(f"分割成 {len(long_chunks)} 个块")
    for i, chunk in enumerate(long_chunks[:3], 1):
        print(f"块 {i} (长度: {len(chunk)}):")
        print(chunk[:100] + "..." if len(chunk) > 100 else chunk)
        print("-" * 30)

if __name__ == "__main__":
    test_spacy_text_splitter()
