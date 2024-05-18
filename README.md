# LLM Stream Service

![](https://img.shields.io/badge/license-MIT-blue)[![](https://img.shields.io/badge/Engilsh-0000FF)](README.md)[![](https://img.shields.io/badge/中文-FF0000)](README_zh.md)

**Streaming API** and **Web page** for Large Language Models based on Python.

This repository contains:
1. Flask API: **REAL** streaming generation of LLM and streaming response interface.
2. Gradio APP: fast and easy LLM web page.

## Quick Start

Take Llama3 for example: 

1. Follow [Llama3 download](https://github.com/meta-llama/llama3?tab=readme-ov-file#download) to download Meta-Llama-3-8B-Instruct model, or from [huggingface](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct) / [modelscope](https://modelscope.cn/models/LLM-Research/Meta-Llama-3-8B-Instruct/summary).
2. Follow [Llama3 quick-start](https://github.com/meta-llama/llama3?tab=readme-ov-file#quick-start) to install dependencies for Llama3.
3. Clone this repository and install dependencies:

    ```bash
    git clone https://github.com/JinHanLei/LLM-Stream-Service
    pip install flask gradio transformers
    ```


4. Run Flask service:

   ```bash
   python llama3_service.py --host 0.0.0.0 --port 8800 --ckpts /Meta-Llama-3-8B-Instruct
   ```

   **Note**
   - Replace  `Meta-Llama-3-8B-Instruct/` with the path to your checkpoint directory.

5. Run Gradio service:

   ```bash
   gradio llama3_app.py
   ```

   **Note**

   - Replace the `Address` variable in `llama3_app.py` with your service address.

# Journey and Challenges

1. The initial streaming output scheme adopted by the project was the **TextIteratorStreamer** that comes with the official transformers library. However, the generation speed was still very slow. After researching, I found that the TextIteratorStreamer actually converts print-ready text into a streaming structure, meaning that the LLM first needs to generate the entire text block before converting it, which is not what I wanted. I wanted the LLM to yield each token as it is generated.

2. Subsequently, I came across [LowinLi's project](https://github.com/LowinLi/transformers-stream-generator) that truly implemented streaming output for pretrained models. When I eagerly applied it to the Llama3 model, it threw an error. After debugging, I found that Llama3 has two **eos_tokens**, which caused the loop to generate negative ids. Thus, I made modifications based on this project, cleaned up redundancies, adapted it for Llama3, and made it easier to read and understand.

# Thanks 🙇

- https://github.com/meta-llama/llama3
- https://github.com/TylunasLi/ChatGLM-web-stream-demo
- https://github.com/LowinLi/transformers-stream-generator

