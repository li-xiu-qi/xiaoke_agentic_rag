from typing import List, Optional

class RecursiveTextSplitter:
    """
    纯递归文本分割器，不依赖 spaCy，仅使用分割符递归分割文本。
    """
    def __init__(self, chunk_size: int = 800, separators: Optional[List[str]] = None):
        self.chunk_size = chunk_size
        if separators is None:
            self.separators = [
                "\n\n", "\n", "。", "！", "？", ";", ":", "，", ",", " ", ""
            ]
        else:
            self.separators = separators

    def split_text(self, text: str) -> List[str]:
        if not text or not text.strip():
            return []
        if len(text) <= self.chunk_size:
            return [text.strip()]
        return self._recursive_split(text, self.separators)

    def _recursive_split(self, text: str, separators: List[str]) -> List[str]:
        if not separators or separators[0] == "":
            return self._force_split(text)
        splits = self._split_by_separator(text, separators[0])
        return self._process_splits(splits, separators[1:])

    def _split_by_separator(self, text: str, separator: str) -> List[str]:
        splits = text.split(separator)
        return [split + separator for split in splits[:-1]] + [splits[-1]]

    def _process_splits(self, splits: List[str], remaining_separators: List[str]) -> List[str]:
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

    def _handle_oversized_split(self, chunks: List[str], current_chunk: str, split: str, remaining_separators: List[str]) -> List[str]:
        if current_chunk:
            chunks.append(current_chunk.strip())
        sub_chunks = self._recursive_split(split, remaining_separators)
        chunks.extend(sub_chunks)
        return chunks

    def _add_split_to_chunk(self, chunks: List[str], current_chunk: str, split: str) -> str:
        if len(current_chunk + split) <= self.chunk_size:
            return current_chunk + split
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            return split

    def _force_split(self, text: str) -> List[str]:
        chunks = []
        for i in range(0, len(text), self.chunk_size):
            chunk = text[i:i + self.chunk_size]
            if chunk.strip():
                chunks.append(chunk.strip())
        return chunks


def test_recursive_text_splitter():
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
    print("=== 纯递归文本分割测试 ===")
    splitter = RecursiveTextSplitter(chunk_size=200)
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
    long_splitter = RecursiveTextSplitter(chunk_size=500)
    long_chunks = long_splitter.split_text(long_text)
    print(f"长文档长度: {len(long_text)} 字符")
    print(f"分割成 {len(long_chunks)} 个块")
    for i, chunk in enumerate(long_chunks[:3], 1):
        print(f"块 {i} (长度: {len(chunk)}):")
        print(chunk[:100] + "..." if len(chunk) > 100 else chunk)
        print("-" * 30)

if __name__ == "__main__":
    test_recursive_text_splitter()
