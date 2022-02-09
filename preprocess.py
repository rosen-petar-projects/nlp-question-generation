import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import nlp
from transformers import T5Tokenizer, BartTokenizer, HfArgumentParser

import torch

model_type = 't5'
dataset_path = 'data/squad_multitask/'
train_file_name = 'train_data_qg_hl_t5.pt'
valid_file_name = 'valid_data_qg_hl_t5.pt'

class DataProcessor:
    def __init__(self, tokenizer, model_type="t5", max_source_length=512, max_target_length=32):
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.model_type = model_type
        self.hl_token = "<hl>"
        
        if model_type == "t5":
            self.sep_token = "<sep>"
        elif model_type == "bart":
            self.sep_token = "<sep>"
        else:
            self.sep_token = "[SEP]"
  
    def process(self, dataset):
        if self.model_type == "t5":
            dataset = dataset.map(self.add_eos_examples)
        
        dataset = dataset.map(self.add_special_tokens)
        dataset = dataset.map(self.convert_to_features, batched=True)
        
        return dataset
  
    def add_eos_examples(self, example):
        example['source_text'] = example['source_text'] + " </s>"
        example['target_text'] = example['target_text'] + " </s>"
        return example
  
    def add_special_tokens(self, example):
        example['source_text'] = example['source_text'].replace("{hl_token}", self.hl_token)    
        example['target_text'] = example['target_text'].replace("{sep_token}", self.sep_token)
        return example
  
    def convert_to_features(self, example_batch):
        source_encoding = self.tokenizer.batch_encode_plus(
            example_batch['source_text'],
            max_length=self.max_source_length,
            padding='max_length',
            pad_to_max_length=True,
            truncation=True, 
        )
        target_encoding = self.tokenizer.batch_encode_plus(
            example_batch['target_text'],
            max_length=self.max_target_length,
            padding='max_length',
            pad_to_max_length=True,
            truncation=True, 
        )

        encodings = {
            'source_ids': source_encoding['input_ids'], 
            'target_ids': target_encoding['input_ids'],
            'attention_mask': source_encoding['attention_mask'],
        }

        return encodings

if model_type == 't5':
    tokenizer = T5Tokenizer.from_pretrained("t5-base")
else:
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")

tokenizer.add_tokens(['<sep>', '<hl>'])

train_dataset = nlp.load_dataset(dataset_path, name='highlight_qg_format', split=nlp.Split.TRAIN)
valid_dataset = nlp.load_dataset(dataset_path, name='highlight_qg_format', split=nlp.Split.VALIDATION)

train_dataset = train_dataset.filter(lambda x: (x['task'] == 'qg'))
valid_dataset = valid_dataset.filter(lambda x: (x['task'] == 'qg'))

processor = DataProcessor(
    tokenizer,
    model_type=model_type,
    max_source_length=512,
    max_target_length=32
)

train_dataset = processor.process(train_dataset)
valid_dataset = processor.process(valid_dataset)

columns = ["source_ids", "target_ids", "attention_mask"]
train_dataset.set_format(type='torch', columns=columns)
valid_dataset.set_format(type='torch', columns=columns)

train_path = os.path.join("data", train_file_name)
valid_path = os.path.join("data", valid_file_name)

torch.save(train_dataset, train_path)
torch.save(valid_dataset, valid_path)

tokenizer_path = f"{model_type}_qg_tokenizer"

if not os.path.exists(tokenizer_path):
    os.mkdir(tokenizer_path)

tokenizer.save_pretrained(tokenizer_path)