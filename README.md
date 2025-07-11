# 小可智能RAG系统 (XiaoKe Agentic RAG)

一个具有反思能力的智能检索增强生成(RAG)系统，能够根据回答质量进行反思并重新搜索相关信息。

## 项目特性

- **智能反思机制**: 系统能够对生成的回答进行质量评估，并根据评估结果决定是否需要重新搜索
- **递归文本分割**: 使用自定义的递归文本分割器，支持中文文本的智能分割
- **向量数据库**: 基于 Milvus 的向量数据库，支持高效的语义搜索
- **缓存优化**: 内置文本嵌入缓存机制，提高系统响应速度
- **本地模型支持**: 支持本地部署的大语言模型和嵌入模型

## 项目结构

```text
xiaoke_agentic_rag/
├── agentic_rag.py           # 核心智能RAG系统
├── base_rag.py              # 基础RAG示例
├── chat.py                  # 聊天接口模块
├── get_text_embedding.py    # 文本嵌入生成模块
├── knowledge_database.py    # 向量数据库封装
├── recursive_text_splitter.py # 递归文本分割器
├── tests/                   # 测试文件目录
│   ├── test_chat.py
│   ├── test_get_text_embedding.py
│   ├── test_knowledge_database.py
│   └── test_text_chunking.py
└── caches/                  # 缓存目录
```

## 安装依赖

```bash
pip install pymilvus
pip install openai
pip install python-dotenv
pip install diskcache
```

## 环境配置

创建 `.env` 文件并配置以下环境变量（兼容openai sdk格式）：

```env
LOCAL_API_KEY=your_api_key
LOCAL_BASE_URL=your_local_model_base_url
LOCAL_TEXT_MODEL=your_text_model_name
LOCAL_EMBEDDING_MODEL=your_embedding_model_name
```

## 快速开始

### 基础RAG使用

```python
from base_rag import build_rag_demo

# 运行基础RAG演示
build_rag_demo()
```

### 智能RAG使用

```python
from agentic_rag import AgenticRAG

# 初始化智能RAG系统
rag = AgenticRAG(collection_name="my_rag", max_iterations=3)

# 设置知识库
documents = [
    "人工智能是计算机科学的一个分支...",
    "机器学习是人工智能的重要组成部分...",
    # 更多文档
]
rag.setup_knowledge_base(documents)

# 进行智能问答
response = rag.query("什么是人工智能？")
print(response)
```

## 核心模块说明

### AgenticRAG (智能RAG系统)

具有反思能力的RAG系统，主要功能：

- `setup_knowledge_base()`: 设置知识库
- `query()`: 智能问答，支持多轮反思
- `initial_search()`: 初始搜索
- `reflect_on_answer()`: 回答质量反思
- `refined_search()`: 精细化搜索

### VectorDatabase (向量数据库)

基于Milvus的向量数据库封装：

- `create_collection()`: 创建向量集合
- `insert_documents()`: 插入文档
- `search()`: 语义搜索
- `delete_collection()`: 删除集合

### RecursiveTextSplitter (递归文本分割器)

智能文本分割工具：

- 支持中文文本分割
- 递归分割策略
- 可自定义分割符
- 保持文本语义完整性

### 缓存机制

- 文本嵌入缓存，避免重复计算
- 基于diskcache的持久化缓存
- 自动缓存键生成

## 测试

运行测试文件：

```bash
python tests/test_chat.py
python tests/test_get_text_embedding.py
python tests/test_knowledge_database.py
python tests/test_text_chunking.py
```

## 系统要求

- Python 3.8+
- 本地大语言模型服务
- 本地文本嵌入模型服务
注意：本项目基于xinference实现本地化，或者建议使用硅基流动等兼容openai服务。
硅基流动可以去<https://cloud.siliconflow.cn/i/FcjKykMn>获取api key

## 注意事项

1. 确保本地模型服务正常运行
2. 正确配置环境变量
3. 首次运行会创建缓存和数据库文件

## 许可证

本项目采用MIT许可证。

## 贡献

欢迎提交Issue和Pull Request来改进项目！

## 更新日志

### v1.0.0

- 实现基础RAG功能
- 添加智能反思机制
- 支持本地模型部署
- 添加缓存优化
