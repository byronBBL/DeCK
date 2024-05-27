import os
import json
import random
from tqdm import tqdm

import torch
from transformers import AutoTokenizer, AutoModel, StoppingCriteria, StoppingCriteriaList, AutoModelForCausalLM

import argparse
from texttable import Texttable
import os
import json
import random
import transformers
from tqdm import tqdm
import argparse
import pandas as pd

from deck import DECK

transformers.logging.set_verbosity(40)

def args_print(args):
    _dict = vars(args)
    t = Texttable() 
    t.add_row(["Parameter", "Value"])
    for k in _dict:
        t.add_row([k, _dict[k]])
    print(t.draw())

parser = argparse.ArgumentParser(description='Arguments For ICE Evaluation')

'''
    arguments
'''
parser.add_argument('--model_name', type=str, default='meta-llama/Llama-2-7b-chat-hf', 
                    help='meta-llama/Llama-2-7b-chat-hf, meta-llama/Llama-2-13b-chat-hf, meta-llama/Meta-Llama-3-8B-Instruct, mistralai/Mistral-7B-Instruct-v0.2')
parser.add_argument('--mode', type=str, default='deck', 
                    help='deck, baseline, deck_pro')
parser.add_argument('--data_path', type=str, default='./MQuAKE')
parser.add_argument("--num-gpus", type=str, default="1")
parser.add_argument("--max_gpu_memory", type=int, default=27)
parser.add_argument("--device", type=str, choices=["cuda", "cpu"], default="cuda")
# parallel mode (split the dataset into multiple parts, inference by separate processes)
parser.add_argument("--max-new-tokens", type=int, default=100)
parser.add_argument("--top_p", type=float, default=0.95)
parser.add_argument("--top_k", type=int, default=0)
parser.add_argument("--temperature", type=float, default=0.9)
parser.add_argument("--repetition_penalty", type=float, default=None)
parser.add_argument("--relative_top", type=float, default=0.01)
parser.add_argument("--batch_size", type=str, default='full', help='1, full')
args = parser.parse_args()

args_print(args)

model_name = args.model_name
num_gpus = args.num_gpus
device = args.device
mode = args.mode


def call_deck(model, edit_prompt, origin_prompt, edit_task_prompt):
    generate_kwargs = dict(max_new_tokens=args.max_new_tokens, top_p=args.top_p, top_k=args.top_k, temperature=args.temperature, mode=mode, repetition_penalty=args.repetition_penalty, relative_top=args.relative_top)
    generated_text = model.generate(edit_prompt, origin_prompt, edit_task_prompt, **generate_kwargs)
    for stop_word in stop:
        if stop_word in generated_text:
            generated_text = generated_text.split(stop_word)[0]
            break

    return generated_text


def mean_pooling(token_embeddings, mask):
    token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.)
    sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
    return sentence_embeddings

def get_sent_embeddings(sents, contriever, tok, BSZ=32):    
    all_embs = []
    for i in tqdm(range(0, len(sents), BSZ)):
        sent_batch = sents[i:i+BSZ]
        inputs = tok(sent_batch, padding=True, truncation=True, return_tensors='pt').to(device)
        with torch.no_grad():
            outputs = contriever(**inputs)
            embeddings = mean_pooling(outputs[0], inputs['attention_mask'])
        all_embs.append(embeddings.cpu())
    all_embs = torch.vstack(all_embs)
    return all_embs

def retrieve_facts(query, fact_embs, contriever, tok, k=3):
    inputs = tok([query], padding=True, truncation=True, return_tensors='pt').to(device)
    with torch.no_grad():
        outputs = contriever(**inputs)
        query_emb = mean_pooling(outputs[0], inputs['attention_mask']).cpu()
    sim = (query_emb @ fact_embs.T)[0]
    knn = sim.topk(k, largest=True)
    return knn.indices

with open(args.data_path + '/datasets/MQuAKE-CF-3k.json', 'r') as f:
    dataset = json.load(f)
    
new_facts = set()
for d in dataset:
    for r in d["requested_rewrite"]:
        new_facts.add(f'{r["prompt"].format(r["subject"])} {r["target_new"]["str"]}')
new_facts = list(new_facts)

contriever = AutoModel.from_pretrained("facebook/contriever-msmarco").to(device)
tokenizer_con = AutoTokenizer.from_pretrained("facebook/contriever-msmarco")

embs = get_sent_embeddings(new_facts, contriever, tokenizer_con)

# read prompts
with open(args.data_path + '/prompts/ice_prompt_cot.txt', 'r') as f:
    edit_task_prompt = f.read()
    
with open(args.data_path + '/prompts/origin_prompt_cot.txt', 'r') as f:
    origin_task_prompt = f.read()
    
stop = ["Question"]
model = DECK(model_name, device, num_gpus, args.max_gpu_memory)
model.set_stop_words(stop)

# Run ICE

cor = 0
tot = 0

for _id, d in enumerate(tqdm(dataset[:1000])):
    tot += 1
    new_fact = ""
    if args.batch_size == '1':
        for r in d["requested_rewrite"]:
            new_fact += f'{r["prompt"].format(r["subject"])} {r["target_new"]["str"]}. '
    for q in d["questions"]:
        ans = ""
        found_ans = False
        if args.batch_size == 'full':
            fact_ids = retrieve_facts(q, embs, contriever, tokenizer_con)
            fact_sent = [new_facts[i] for i in fact_ids]
            new_fact = ". ".join(fact_sent)
        edit_prompt = edit_task_prompt + 'Question: ' + q + '\nEdit Knowledge: ' + new_fact + '\n'
        origin_prompt = origin_task_prompt + 'Question: ' + q + '\n'
        gen = call_deck(model, edit_prompt, origin_prompt, new_fact)
        # print(gen)
        last_sent = gen.strip().split('\n')[-1]
        if last_sent.startswith('Answer: '):
            found_ans = True
            ans = last_sent[len("Answer: "):]
            
        if ans == d["new_answer"] or ans in d["new_answer_alias"]:
            cor += 1
            break

print(f'Multi-hop acc = {cor / tot} ({cor} / {tot})')
