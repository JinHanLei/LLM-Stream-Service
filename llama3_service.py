#! /usr/bin/python3
# -*- coding: utf-8 -*-
from gevent import monkey, pywsgi
monkey.patch_all()
from stream_generate import init_stream_support
init_stream_support()
from typing import List, Tuple
from flask import Flask, request, Response
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
from threading import Thread
import argparse
import logging
import json
import sys
import warnings
warnings.filterwarnings("ignore")

def getLogger(name, file_name, use_formatter=True):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    console_handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(asctime)s    %(message)s')
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)
    logger.addHandler(console_handler)
    if file_name:
        handler = logging.FileHandler(file_name, encoding='utf8')
        handler.setLevel(logging.INFO)
        if use_formatter:
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(message)s')
            handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


logger = getLogger('Llama3', 'llama3.log')


class Llama:
    def __init__(self, CKPTS) -> None:
        logger.info("Start initialize model...")
        self.tokenizer = AutoTokenizer.from_pretrained(CKPTS, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(CKPTS, torch_dtype=torch.bfloat16, device_map="auto", )
        self.model.eval()
        logger.info("Model initialization finished.")

    def clear(self) -> None:
        if torch.cuda.is_available():
            for device in self.model.devices:
                with torch.cuda.device(int(device)):
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()

    def build_inputs(self, prompt: str, query: str, history: List[Tuple[str, str]]):
        template = []
        if prompt:
            template += [{"role": "assistant", "content": prompt}]
        if history:
            for q, a in history:
                template += [{"role": "user", "content": q},
                             {"role": "assistant", "content": a}]
        template += [{"role": "user", "content": query}]
        encodings = self.tokenizer.apply_chat_template(template, add_generation_prompt=True, return_tensors="pt").to(
            self.model.device)
        terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
        return encodings, terminators

    def chat(self, prompt: str, query: str, history: List[Tuple[str, str]]):
        encodings, terminators = self.build_inputs(prompt, query, history)
        outputs = self.model.generate_(
            input_ids=encodings,
            max_new_tokens=512,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
        )
        response = self.tokenizer.decode(outputs[0][encodings.shape[-1]:], skip_special_tokens=True)
        history.append((query, response))
        return response, history

    def stream(self, prompt: str, query: str, history: List[Tuple[str, str]]):
        encodings, terminators = self.build_inputs(prompt, query, history)
        generator = self.model.generate_(input_ids=encodings,
                                         max_new_tokens=512,
                                         eos_token_id=terminators,
                                         do_sample=True,
                                         temperature=0.6,
                                         top_p=0.9,
                                         do_stream=True)
        response = ""
        for this_response in generator:
            this_response = this_response.cpu().numpy().tolist()
            response += self.tokenizer.decode(this_response, skip_special_tokens=True)
            yield {"response": response, "history": history + [(query, response)], "ByeJin": False}
        logger.info("Answer - {}".format(response))
        yield {"response": response, "history": history + [(query, response)], "ByeJin": True}

    def stream_with_TextIteratorStreamer(self, prompt: str, query: str, history: List[Tuple[str, str]]):
        encodings, terminators = self.build_inputs(prompt, query, history)

        streamer = TextIteratorStreamer(tokenizer=self.tokenizer, skip_special_tokens=True)
        generate_kwargs = dict(input_ids=encodings,
                               streamer=streamer,
                               max_new_tokens=512,
                               eos_token_id=terminators,
                               do_sample=True,
                               temperature=0.6,
                               top_p=0.9, )

        thread = Thread(target=self.model.generate_, kwargs=generate_kwargs)
        thread.start()

        response = ""
        e_len = len(self.tokenizer.decode(encodings[0], skip_special_tokens=True))
        for this_response in streamer:
            response += this_response
            if len(response) > e_len:
                yield {"response": response[e_len:], "history": history + [(query, response[e_len:])], "ByeJin": False}
        logger.info("Answer - {}".format(response[e_len:]))
        yield {"response": response[e_len:], "history": history + [(query, response[e_len:])], "ByeJin": True}


def start_server(CKPTS, http_address: str, port: int):
    bot = Llama(CKPTS)
    app = Flask(__name__)

    @app.route("/")
    def index():
        return Response(json.dumps({'message': 'started', 'success': True}, ensure_ascii=False),
                        content_type="application/json")

    @app.route("/chat", methods=["GET", "POST"])
    def chat():
        result = {"query": "", "response": "", "success": False}
        try:
            if "application/json" in request.content_type:
                arg_dict = request.get_json()
                query = arg_dict["query"]
                prompt = arg_dict["prompt"]
                history = arg_dict["history"]
                logger.info("Query - {}".format(query))
                if len(history) > 0:
                    logger.info("History - {}".format(history))
                response, history = bot.chat(prompt, query, history)
                logger.info("Answer - {}".format(response))
                result = {"query": query, "response": response,
                          "history": history, "success": True}
        except Exception as e:
            logger.error(f"error: {e}")
        return Response(json.dumps(result, ensure_ascii=False), content_type="application/json")

    @app.route("/stream", methods=["POST"])
    def chat_stream():
        def decorate(generator):
            for item in generator:
                yield json.dumps(item, ensure_ascii=False)

        prompt, query, history = None, None, None
        try:
            if "application/json" in request.content_type:
                arg_dict = request.get_json()
                query = arg_dict["query"]
                prompt = arg_dict["prompt"]
                history = arg_dict["history"]
                logger.info("Prompt - {}".format(prompt))
                logger.info("Query - {}".format(query))
                if len(history) > 0:
                    logger.info("History - {}".format(history))
        except Exception as e:
            logger.error(f"error: {e}")
        return Response(decorate(bot.stream(prompt, query, history)), mimetype='text/event-stream')

    @app.route("/clear", methods=["GET", "POST"])
    def clear():
        try:
            bot.clear()
            return Response(json.dumps({"success": True}, ensure_ascii=False), content_type="application/json")
        except:
            return Response(json.dumps({"success": False}, ensure_ascii=False), content_type="application/json")

    logger.info("starting server...")
    server = pywsgi.WSGIServer((http_address, port), app)
    server.serve_forever()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Stream API Service for Meta-Llama-3')
    parser.add_argument('--ckpts', '-c', help='Model path', default="./ckpts/Meta-Llama-3-8B-Instruct")
    parser.add_argument('--host', '-H', help='host to listen', default='0.0.0.0')
    parser.add_argument('--port', '-P', help='port of this service', default=8800)
    args = parser.parse_args()
    start_server(args.ckpts, args.host, args.port)
