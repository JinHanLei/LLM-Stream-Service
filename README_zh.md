# LLM Stream Service

![](https://img.shields.io/badge/license-MIT-blue)[![](https://img.shields.io/badge/Engilsh-0000FF)](README.md)[![](https://img.shields.io/badge/中文-FF0000)](README_zh.md)

完全基于Python的大语言模型（LLM）**流式API**和**网页**。

本项目包含：
1. Transformers流式生成：**真正**实现基于transformers的**所有**预训练模型的流式生成；
2. Flask API: 后端流式响应；
3. Gradio APP: 快速、简单的LLM前端界面。

## Quick Start

以Llama3的部署为例：

1. 参考[Llama3 download](https://github.com/meta-llama/llama3?tab=readme-ov-file#download)下载Meta-Llama-3-8B-Instruct模型, 或者[huggingface](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct) / [modelscope](https://modelscope.cn/models/LLM-Research/Meta-Llama-3-8B-Instruct/summary)（国内推荐modelscope）。
2. 参考[Llama3 quick-start](https://github.com/meta-llama/llama3?tab=readme-ov-file#quick-start)安装Llama3的依赖.
3. 克隆本项目并安装所需依赖：

    ```bash
    git clone https://github.com/JinHanLei/LLM-Stream-Service
    pip install flask gradio transformers
    ```


4. 运行Flask服务:

   ```bash
   python llama3_service.py --host 0.0.0.0 --port 8800 --ckpts /Meta-Llama-3-8B-Instruct
   ```

   **注意**

   - 请将命令行中的 `Meta-Llama-3-8B-Instruct/`换成您下载模型文件的路径。

5. 运行Gradio前端界面:

   ```bash
   gradio llama3_app.py
   ```

   **Note**

   - 请将py文件中的`Address`变量换成您的flask服务地址。

# 心路历程

- 项目最初采用的流式输出方案是transformers官方自带的TextIteratorStreamer，然而生成速度还是很慢。调研后发现TextIteratorStreamer实际上是将print-ready text转换为流式结构，也就是需要LLM首先生成整段文本，再进行转换，这不是我想要的，我想要LLM每生成一个token就yield给我。
- 随后我发现了LowinLi的项目（感谢其付诸的努力），真正地实现了预训练模型的流式输出。当我迫不及待地使用到Llama3模型上时，报错了。debug后发现是因为Llama3有两个eos_token，循环过程中生成了负数的id。于是我在该项目的基础上进行了修正，并清理了冗余，使之适配了Llama3，并更容易阅读和理解。
# 鸣谢🙇

- https://github.com/meta-llama/llama3
- https://github.com/TylunasLi/ChatGLM-web-stream-demo
- https://github.com/LowinLi/transformers-stream-generator

