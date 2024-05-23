import argparse
import time
import csv
import tqdm
import os
import json

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer
from transformers.generation.stopping_criteria import StoppingCriteriaList, LLamaQaStoppingCriteria

import argparse
import warnings
import pandas as pd
import numpy as np
from collections import defaultdict
import math

stop_words = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 
'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 
'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 
'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves',
 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are',
 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing',
 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 
'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 
'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 
'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 
'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 
's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y',
 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 
'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn',
 "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 
'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]

class DECK:
    def __init__(self, model_name, device, num_gpus, max_gpu_memory=27):
        self.model_name = model_name
        self.device = device
        self.num_gpus = num_gpus
        self.stopping_criteria = None
        self.max_gpu_memory = max_gpu_memory

        self.model, self.tokenizer = self.load_model(model_name)

    def load_model(self, model_name):
        if self.device == "cuda":
            kwargs = {"torch_dtype": torch.float16, "offload_folder": f"{model_name}/offload"}
            if self.num_gpus == "auto":
                kwargs["device_map"] = "auto"
            else:
                self.num_gpus = int(self.num_gpus)
                if self.num_gpus != 1:
                    kwargs.update({
                        "device_map": "auto",
                        "max_memory": {i: f"{self.max_gpu_memory}GiB" for i in range(self.num_gpus)},
                    })
        elif self.device == "cpu":
            kwargs = {}
        else:
            raise ValueError(f"Invalid device: {self.device}")
        
        tokenizer = AutoTokenizer.from_pretrained(model_name if not 'vicuna' in model_name else 'huggyllama/llama-7b')
        model = AutoModelForCausalLM.from_pretrained(model_name,
            low_cpu_mem_usage=True, **kwargs)

        if self.device == "cuda" and self.num_gpus == 1:
            model.cuda()
        
        return model, tokenizer

    def set_stop_words(self, stop_words):
        self.stop_words = stop_words
        self.stopping_criteria = StoppingCriteriaList()
        list_stop_word_ids = []
        for stop_word in self.stop_words:
            stop_word_ids = self.tokenizer.encode('\n' + stop_word)[3:]
            list_stop_word_ids.append(stop_word_ids)
            print("Added stop word: ", stop_word, 'with the ids', stop_word_ids, flush=True)
        self.stopping_criteria.append(LLamaQaStoppingCriteria(list_stop_word_ids))

    def compute_similarity(self, token1, token2):
        embedding1 = self.model.get_input_embeddings()(torch.tensor([self.tokenizer.convert_tokens_to_ids(token1)], device=self.device))
        embedding2 = self.model.get_input_embeddings()(torch.tensor([self.tokenizer.convert_tokens_to_ids(token2)], device=self.device))
        return torch.cosine_similarity(embedding1, embedding2, dim=-1)

    def get_mask_from_reference(self, reference_prompt, mask_strategy='mean', alpha=0.1, beta=0.1):
        reference_inputs = self.tokenizer(reference_prompt, return_tensors='pt').to(self.device)
        reference_embeddings = self.model.get_input_embeddings()(reference_inputs['input_ids']).to(self.device)
        reference_tokens = self.tokenizer.convert_ids_to_tokens(reference_inputs['input_ids'][0])

        vocab_size = self.model.config.vocab_size

        attention_mask = torch.ones(vocab_size, device=self.device)

        vocab_freq = defaultdict(int)
        for token in reference_tokens:
            if token not in stop_words:
                vocab_freq[token] += 1

        mask_scores_freq = torch.zeros(vocab_size, device=self.device)
        for token, freq in vocab_freq.items():
            token_id = self.tokenizer.convert_tokens_to_ids(token)
            if mask_strategy == 'mean':
                mask_scores_freq[token_id] = reference_embeddings[:, reference_tokens.index(token), :].mean()
            elif mask_strategy == 'max':
                mask_scores_freq[token_id] = reference_embeddings[:, reference_tokens.index(token), :].max()
            elif mask_strategy == 'norm':
                mask_scores_freq[token_id] = torch.norm(reference_embeddings[:, reference_tokens.index(token), :], p=2)
            else:
                raise ValueError(f"Unsupported: {mask_strategy}")

        for token, freq in vocab_freq.items():
            token_id = self.tokenizer.convert_tokens_to_ids(token)
            attention_mask[token_id] = 1.0 + math.log(freq + 1e-12) * alpha

        mask_scores_freq = mask_scores_freq * alpha# + 1.0
        

        mask_scores_sim = torch.zeros(vocab_size, device=self.device)
        for token in vocab_freq.keys():
            token_id = self.tokenizer.convert_tokens_to_ids(token)
            mask_scores_sim[token_id] = max(self.compute_similarity(token, ref_token) for ref_token in reference_tokens)

        attention_mask = attention_mask + mask_scores_freq + beta * mask_scores_sim


        attention_mask = attention_mask.unsqueeze(0)
        return attention_mask
     
    def generate(self, input_text, input_text_student, input_text_ref, max_new_tokens=256, top_p=0.95, top_k=0, temperature=0.8, mode='baseline', verbose=False, remove_stop_words=False, relative_top=0.1, **kwargs):
        with torch.no_grad():
            
            assert input_text is not None, "input_text must be specified"
            input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids.to(self.device)
            max_len = input_ids.shape[-1] + max_new_tokens
            
            if mode == 'baseline':
                outputs = self.model.generate(input_ids, max_length=max_len, num_return_sequences=1,
                                    output_scores=True, return_dict_in_generate=True, deck_decoding=False,
                                    top_p=top_p, top_k=top_k, temperature=temperature, stopping_criteria=self.stopping_criteria, **kwargs)
            elif mode == 'deck':
                assert input_text is not None, "input_text must be specified"
                assert input_text_student is not None, "input_text_student must be specified"
                input_ids_student = self.tokenizer(input_text_student, return_tensors="pt").input_ids.to(self.device)
                max_len_student = input_ids_student.shape[-1] + max_new_tokens
                max_len = max(max_len, max_len_student)
                outputs = self.model.generate(input_ids, input_ids_student, max_length=max_len, num_return_sequences=1,
                                        output_scores=True, return_dict_in_generate=True, deck_decoding=True,
                                        top_p=top_p, top_k=top_k, temperature=temperature, stopping_criteria=self.stopping_criteria, relative_top=relative_top, **kwargs,)

            elif mode == 'deck_pro':
                assert input_text is not None, "input_text must be specified"
                assert input_text_student is not None, "input_text_student must be specified"
                assert input_text_ref is not None, "input_text_ref must be specified"
                input_ids_student = self.tokenizer(input_text_student, return_tensors="pt").input_ids.to(self.device)
                max_len_student = input_ids_student.shape[-1] + max_new_tokens
                max_len = max(max_len, max_len_student)
                ref_mask = self.get_mask_from_reference(input_text_ref, mask_strategy='mean', alpha=0.06, beta=0.01)
                outputs = self.model.generate(input_ids, input_ids_student, max_length=max_len, num_return_sequences=1,
                                        output_scores=True, return_dict_in_generate=True, deck_decoding=True,
                                        top_p=top_p, top_k=top_k, temperature=temperature, stopping_criteria=self.stopping_criteria, relative_top=relative_top, ref_mask=ref_mask, **kwargs,)

            sequences, scores = outputs.sequences, outputs.scores

            gen_sequences = sequences[:, input_ids.shape[-1]:][0, :]
            gen_arr = gen_sequences.cpu().numpy()

            output_str = self.tokenizer.decode(gen_sequences, skip_special_tokens=True)

            if verbose:
                print('MODEL OUTPUT: \n{0}'.format(output_str))

            if remove_stop_words:
                for stop_word in self.stop_words:
                    length_to_remove = len(stop_word)
                    if output_str[-length_to_remove:] == stop_word:
                        output_str = output_str[:-length_to_remove]
                output_str = output_str.strip()

        if self.device:
            torch.cuda.empty_cache()

        return output_str

    def get_relative_top_filter(self, scores: torch.FloatTensor, relative_top: float = 0.1, min_tokens_to_keep: int = 1):
        scores_normalized = scores.log_softmax(dim=-1) 
        sorted_logits, sorted_indices = torch.sort(scores_normalized, descending=True)
        min_thresh = sorted_logits[..., min_tokens_to_keep-1] 
        probs_max = torch.max(scores_normalized, dim=-1).values
        probs_thresh = probs_max + np.log(relative_top)
        probs_thresh = torch.min(min_thresh, probs_thresh)
        probs_thresh = probs_thresh.unsqueeze(-1)
        return scores_normalized < probs_thresh
