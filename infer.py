import torch
import argparse
import os

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from transformers import StoppingCriteria, StoppingCriteriaList
from transformers import TextStreamer, GenerationConfig

class LocalStoppingCriteria(StoppingCriteria):

    def __init__(self, tokenizer, stop_words = []):
        super().__init__()
        
        stops = [tokenizer(stop_word, return_tensors='pt', add_special_tokens = False)['input_ids'].squeeze() for stop_word in stop_words]
        print('stop_words', stop_words)
        print('stop_words_ids', stops)
        self.stop_words = stop_words
        self.stops = [stop.cuda() for stop in stops]
        self.tokenizer = tokenizer
    def _compare_token(self, input_ids):
        for stop in self.stops:
            if len(stop.size()) != 1:
                continue
            stop_len = len(stop)
            if torch.all((stop == input_ids[0][-stop_len:])).item():
                return True

        return False
    def _compare_decode(self, input_ids):
        input_str = self.tokenizer.decode(input_ids[0])
        for stop_word in self.stop_words:
            if input_str.endswith(stop_word):
                return True
        return False
            
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        if True:
            return self._compare_decode(input_ids)
        else:
            return self._compare_token(input_ids)
    
import os
import json


def main():
    parser = argparse.ArgumentParser("simple argument", add_help=False)

    parser.add_argument("model_name", type=str, help="model_name")
    args = parser.parse_args()
    print(args)

    instruction_prefix = "### instruction: "
    input_prefix = "### input: "
    answer_prefix = "### Response: "
    endoftext = "<|end|>"
    max_new_tokens = 1024

    model = AutoModelForCausalLM.from_pretrained(args.model_name, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    model.eval()

    stop_words = [endoftext, '<s>', '</s>', '\n\n', '###']
    
    stopping_criteria = StoppingCriteriaList([LocalStoppingCriteria(tokenizer=tokenizer, stop_words=stop_words)])

    streamer = TextStreamer(tokenizer)
    
    def mt_generate(x):
        generation_config = GenerationConfig(
            temperature=0.9,
            top_p=0.7,
            top_k=100,
            max_new_tokens=max_new_tokens,
            early_stopping=True,
            do_sample=True,
        )
        q = f"{instruction_prefix}{x}\n\n{answer_prefix}"
        #print(q)
        gened = model.generate(
            **tokenizer(
                q,
                return_tensors='pt',
                return_token_type_ids=False
            ).to('cuda'),
            generation_config=generation_config,
            eos_token_id=model.config.eos_token_id,
            stopping_criteria=stopping_criteria,
            streamer=streamer,
        )
        result_str= tokenizer.decode(gened[0])
        
        start_tag = f"\n\n{answer_prefix}"
        start_index = result_str.find(start_tag)

        if start_index != -1:
            result_str = result_str[start_index + len(start_tag):].strip()
        return result_str

    while True:
        text = input('>')
        if len(text) == 0:
            text = "nlp에 대해서 설명해주세요."
        result = mt_generate(text)
        print('vvvvvvvvvvvvvvvvvvvvvvvvvvvvv')
        print(result)
        print('^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^')
if __name__ == '__main__':
    main()    