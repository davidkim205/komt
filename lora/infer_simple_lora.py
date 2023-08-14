import torch

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from transformers import StoppingCriteria, StoppingCriteriaList
from transformers import TextStreamer, GenerationConfig
from peft import PeftModel, PeftConfig

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
        input_str = self.tokenizer.decode(input_ids[0])
        for stop_word in self.stop_words:
            if input_str.endswith(stop_word):
                return True
        return False

#
# config
peft_model_name = 'davidkim205/komt-Llama-2-7b-chat-hf-lora'
model_name = 'davidkim205/komt-Llama-2-7b-chat-hf'
instruction_prefix = "### instruction: "
input_prefix = "### input: "
answer_prefix = "### Response: "
endoftext = "<|end|>"
stop_words = [endoftext, '<s>', '###']
generation_config = GenerationConfig(
    temperature=0.9,
    top_p=0.7,
    top_k=100,
    max_new_tokens=2048,
    early_stopping=True,
    do_sample=True,
)
#
# create model
config = PeftConfig.from_pretrained(peft_model_name)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)
model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_config,
                                             device_map="auto")
model = PeftModel.from_pretrained(model, peft_model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
stopping_criteria = StoppingCriteriaList([LocalStoppingCriteria(tokenizer=tokenizer, stop_words=stop_words)])
streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
model.eval()

#
# generate
prompt = f"### instruction: 자동차 종합(정기)검사 의무기간은 얼마인가요?.\n\n### Response:"
gened = model.generate(
    **tokenizer(
        prompt,
        return_tensors='pt',
        return_token_type_ids=False
    ).to('cuda'),
    generation_config=generation_config,
    eos_token_id=model.config.eos_token_id,
    stopping_criteria=stopping_criteria,
    streamer=streamer
)
output_text = tokenizer.decode(gened[0], skip_special_tokens=True)

print('--------------------')
print(output_text)
