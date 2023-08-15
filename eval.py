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

def seed_everything(seed: int):
    import random, os
    import numpy as np
    import torch

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True



def main():
    parser = argparse.ArgumentParser("simple argument", add_help=False)

    parser.add_argument("model_name", type=str, help="model_name")
    parser.add_argument("--arguments", default=None, type=str, help="model_name")
    parser.add_argument("--max_new_tokens", default=1024, type=int, help="batch_size")
    parser.add_argument("--input_data_path", default='./datasets/testset.txt', type=str, help="output_dir")
    parser.add_argument("--output_dir", default='./results/', type=str, help="output_dir")

    args = parser.parse_args()
    print(args)
    #seed_everything(1234)
    
    instruction_prefix = "### instruction: "
    input_prefix = "### input: "
    answer_prefix = "### Response: "
    endoftext = "<|end|>"

    model = AutoModelForCausalLM.from_pretrained(args.model_name, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    model.eval()


    print("model.config.pad_token_id", model.config.pad_token_id)
    print("model.config.bos_token_id", model.config.bos_token_id)
    print("model.config.eos_token_id", model.config.eos_token_id)
    
    stop_words = [endoftext, '<s>', '</s>', '\n\n', '###']
    
    stopping_criteria = StoppingCriteriaList([LocalStoppingCriteria(tokenizer=tokenizer, stop_words=stop_words)])

    streamer = TextStreamer(tokenizer)
    
    def gen(x):
        generation_config = GenerationConfig(
            temperature=0.9,
            top_p=0.7,
            top_k=100,
            max_new_tokens=args.max_new_tokens,
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


    input_data = open(args.input_data_path, mode='r', encoding='utf-8')
    def create_directory_from_filepath(filepath):
        directory = os.path.dirname(filepath)
        os.makedirs(directory, exist_ok=True)

    output_data_path = os.path.join(args.output_dir, "result_")
    output_data_path += args.model_name.replace('/','-')
    output_data_path += '-'
    output_data_path += os.path.basename(args.input_data_path)
    create_directory_from_filepath(args.output_dir)

    print('write', output_data_path)
    output_data = open(output_data_path, mode='w', encoding='utf-8')


    for num, line in enumerate(input_data.readlines()):
        instruction = line
        final_output = gen(instruction)
        new_data = {
            "id": num,
            "instruction": instruction,
            "result": final_output
        }
        print(json.dumps(new_data, indent=2, ensure_ascii=False))
        output_data.write(json.dumps(new_data, ensure_ascii=False) + '\n')
        
if __name__ == '__main__':
    main()    