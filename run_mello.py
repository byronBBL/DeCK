import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import json
import random
from tqdm import tqdm

import torch
from transformers import AutoTokenizer, AutoModel, StoppingCriteria, StoppingCriteriaList, AutoModelForCausalLM
import pyvene as pv

from openai import OpenAI
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

parser = argparse.ArgumentParser(description='Arguments For Mello Evaluation')

'''
    arguments
'''
parser.add_argument('--model_name', type=str, default='huggyllama/Llama-2-7b-chat-hf', 
                    help='./Llama-2-7b-chat-hf, ./Llama-2-13b-chat-hf')
parser.add_argument('--mode', type=str, default='baseline', 
                    help='deck, baseline')
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

args = parser.parse_args()
args_print(args)

model_name = args.model_name
num_gpus = args.num_gpus
device = args.device


def call_deck(model, edit_prompt, origin_prompt, mode):
    generate_kwargs = dict(max_new_tokens=args.max_new_tokens, top_p=args.top_p, top_k=args.top_k, temperature=args.temperature, mode=mode, repetition_penalty=args.repetition_penalty, relative_top=args.relative_top)
    generated_text = model.generate(edit_prompt, origin_prompt, **generate_kwargs)
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
    for i in range(0, len(sents), BSZ):
        sent_batch = sents[i:i+BSZ]
        inputs = tok(sent_batch, padding=True, truncation=True, return_tensors='pt').to(device)
        with torch.no_grad():
            outputs = contriever(**inputs)
            embeddings = mean_pooling(outputs[0], inputs['attention_mask'])
        all_embs.append(embeddings.cpu())
    all_embs = torch.vstack(all_embs)
    return all_embs

def retrieve_facts(query, fact_embs, contriever, tok, k=1):
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
origin_facts = set()
for d in dataset:
    for r in d["requested_rewrite"]:
        new_facts.add(f'{r["prompt"].format(r["subject"])} {r["target_new"]["str"]}')
new_facts = list(new_facts)


contriever = AutoModel.from_pretrained(args.data_path + "/contriever-msmarco").to(device)
tokenizer_con = AutoTokenizer.from_pretrained(args.data_path + "/contriever-msmarco")

embs_edit = get_sent_embeddings(new_facts, contriever, tokenizer_con)

# read prompts
with open(args.data_path + '/prompts/MeLLo-prompt.txt', 'r') as f:
    edit_task_prompt = f.read()
    
    
stop = ["Question", "Retrieved fact:", "Intermediate answer:"]

model = DECK(model_name, device, num_gpus, args.max_gpu_memory)
model.set_stop_words(stop)

# Run MELLO

cor = 0
tot = 0
for d in tqdm(dataset[:500]):
    tot += 1
    for q in d["questions"]:
        found_ans = False
        prompt = edit_task_prompt + "\n\nQuestion: " + q
        origin_prompt = ""
        is_edit = False
        for i in range(4):
            if is_edit:
                gen = call_deck(model, prompt, origin_prompt, "deck")
            else:
                gen = call_deck(model, prompt, None, "baseline")
            
            last_sent = gen.strip().split('\n')[-1]
            
            # if final answer is there, get the answer and exit
            if last_sent.startswith('Final answer: '):
                found_ans = True
                ans = last_sent[len("Final answer: "):]
                break
            
            # otherwise, extract the generated subquestion
            if len(gen.strip().split('\n')) < 2:
                break # failed case
            subquestion = gen.strip().split('\n')[-2]
            if not subquestion.startswith('Subquestion: '):
                break # failed case
            subquestion = subquestion[len("Subquestion: "):]
            
            # retrieve an edited fact using the generated subquestion
            fact_ids = retrieve_facts(subquestion, embs_edit, contriever, tokenizer_con)
            fact_sent = new_facts[fact_ids[0]]
            
            # put the retrieved fact at the end of the prompt, the model self-checks if it contradicts
            prompt = prompt + gen + 'Retrieved fact: ' + fact_sent + '.'
            
            if not last_sent.startswith('Generated answer: '):
                break # failed case
            origin_knowledge = last_sent[len("Generated answer: "):]
            origin_prompt = prompt + gen + 'Retrieved fact: ' + origin_knowledge + '.'
            gen = call_deck(model, prompt, None, "baseline")
            if gen == '\nRetrieved fact contradicts to generated answer.\n':
                is_edit = True
                origin_prompt += '\nRetrieved fact does not contradict to generated answer.\nIntermediate answer: '
            elif gen == '\nRetrieved fact does not contradict to generated answer.\n':
                is_edit = False
            else:
                break
            prompt += gen + 'Intermediate answer: '
            
            
        prompt = prompt + gen
        
        if not found_ans:
            continue
        # if the answer is correct
        if ans == d["new_answer"] or ans in d["new_answer_alias"]:
            cor += 1
            break
    print(f'Multi-hop acc = {cor / tot} ({cor} / {tot})')
    print(wa_lst)
