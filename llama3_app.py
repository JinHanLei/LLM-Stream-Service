import json
import re
import requests
import gradio as gr

Address = "http://127.0.0.1:8800/"
MAX_HISTORY = 5

def chat(query, history, prompt):
    if not prompt:
        prompt = ""
    if not history:
        history = []
    history = history[-MAX_HISTORY:]

    if query:
        res = requests.post(Address + "chat", json={"query": query, "prompt": prompt, "history": history}).json()
        return res["response"]


def stream_chat(query, history, prompt):
    if not prompt:
        prompt = ""
    if not history:
        history = []
    history = history[-MAX_HISTORY:]
    if query:
        byte_sec = []
        stream = requests.post(Address + "stream", json={"query": query, "prompt": prompt, "history": history}, stream=True)
        for byte_line in stream:
            now_byte = byte_line.decode(errors='ignore')
            byte_sec.append(now_byte)
            now_string = "".join(byte_sec)
            if re.findall(r"\"ByeJin\": (.*?)}", now_string):
                byte_sec = []
                yield json.loads(now_string)["response"]


prompt = gr.Textbox(label="Prompt", value="")
demo = gr.ChatInterface(fn=stream_chat, title="Welcome to Llama3 Chatbot", additional_inputs=prompt).queue()
demo.launch()