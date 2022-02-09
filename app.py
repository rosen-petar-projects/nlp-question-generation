import itertools
from typing import Optional, Dict, Union

from nltk import sent_tokenize

import torch
from transformers import(
    AutoModelForSeq2SeqLM, 
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)

class QGPipeline:
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        ans_model: PreTrainedModel,
        ans_tokenizer: PreTrainedTokenizer
    ):
        self.model = model
        self.tokenizer = tokenizer

        self.ans_model = ans_model
        self.ans_tokenizer = ans_tokenizer

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

        if self.ans_model is not self.model:
            self.ans_model.to(self.device)

        assert self.model.__class__.__name__ in ["T5ForConditionalGeneration", "BartForConditionalGeneration"]
        
        if "T5ForConditionalGeneration" in self.model.__class__.__name__:
            self.model_type = "t5"
        else:
            self.model_type = "bart"

    def __call__(self, inputs: str):
        inputs = " ".join(inputs.split())
        sents, answers = self.extract_answers(inputs)
        flat_answers = list(itertools.chain(*answers))
        
        if len(flat_answers) == 0:
          return []

        qg_examples = self.prepare_inputs_for_qg_from_answers_hl(inputs, answers)
        
        qg_inputs = [example['source_text'] for example in qg_examples]
        questions = self.generate_questions(qg_inputs)
        output = [{'answer': example['answer'], 'question': que} for example, que in zip(qg_examples, questions)]
        return output
    
    def generate_questions(self, inputs):
        inputs = self.tokenize(inputs, padding=True, truncation=True)
        
        outs = self.model.generate(
            input_ids=inputs['input_ids'].to(self.device), 
            attention_mask=inputs['attention_mask'].to(self.device), 
            max_length=32,
            num_beams=4,
        )
        
        questions = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in outs]
        return questions
    
    def extract_answers(self, context):
        sents, inputs = self.prepare_inputs_for_ans_extraction(context)
        inputs = self.tokenize(inputs, padding=True, truncation=True)

        outs = self.ans_model.generate(
            input_ids=inputs['input_ids'].to(self.device), 
            attention_mask=inputs['attention_mask'].to(self.device), 
            max_length=32,
        )
        
        dec = [self.ans_tokenizer.decode(ids, skip_special_tokens=False) for ids in outs]
        answers = [item.split('<sep>') for item in dec]
        answers = [i[:-1] for i in answers]
        
        return sents, answers
    
    def tokenize(self,
        inputs,
        padding=True,
        truncation=True,
        add_special_tokens=True,
        max_length=512
    ):
        inputs = self.tokenizer.batch_encode_plus(
            inputs, 
            max_length=max_length,
            add_special_tokens=add_special_tokens,
            truncation=truncation,
            padding="max_length" if padding else False,
            pad_to_max_length=padding,
            return_tensors="pt"
        )
        return inputs
    
    def prepare_inputs_for_ans_extraction(self, text):
        sents = sent_tokenize(text)

        inputs = []
        for i in range(len(sents)):
            source_text = "extract answers:"
            for j, sent in enumerate(sents):
                if i == j:
                    sent = "<hl> %s <hl>" % sent
                source_text = "%s %s" % (source_text, sent)
                source_text = source_text.strip()
            
            if self.model_type == "t5":
                source_text = source_text + " </s>"
            inputs.append(source_text)

        return sents, inputs
    
    def prepare_inputs_for_qg_from_answers_hl(self, sents, answers):
        inputs = []
        for i, answer in enumerate(answers):
            if len(answer) == 0: continue
            for answer_text in answer:
                sent = sents[i]
                sents_copy = sents[:]
                
                answer_text = answer_text.strip()
                
                ans_start_idx = sent.index(answer_text)
                
                sent = f"{sent[:ans_start_idx]} <hl> {answer_text} <hl> {sent[ans_start_idx + len(answer_text): ]}"
                sents_copy[i] = sent
                
                source_text = " ".join(sents_copy)
                source_text = f"generate question: {source_text}" 
                if self.model_type == "t5":
                    source_text = source_text + " </s>"
                
                inputs.append({"answer": answer_text, "source_text": source_text})
        
        return inputs

def pipeline(
    model: Optional = None,
    tokenizer: Optional[Union[str, PreTrainedTokenizer]] = None,
    ans_model: Optional = None,
    ans_tokenizer: Optional[Union[str, PreTrainedTokenizer]] = None
):
    # Use default model/config/tokenizer for the task if no model is provided
    if model is None:
        model = "valhalla/t5-small-qg-hl"
    
    # Try to infer tokenizer from model or config name (if provided as str)
    if tokenizer is None:
        tokenizer = model
    
    # Instantiate tokenizer if needed
    if isinstance(tokenizer, (str, tuple)):
        if isinstance(tokenizer, tuple):
            # For tuple we have (tokenizer name, {kwargs})
            tokenizer = AutoTokenizer.from_pretrained(tokenizer[0], **tokenizer[1])
        else:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer)
    
    # Instantiate model if needed
    if isinstance(model, str):
        model = AutoModelForSeq2SeqLM.from_pretrained(model)
    
    if ans_model is None:
        # load default ans model
        ans_model = "valhalla/t5-small-qa-qg-hl"
        ans_tokenizer = AutoTokenizer.from_pretrained(ans_model)
        ans_model = AutoModelForSeq2SeqLM.from_pretrained(ans_model)
    else:
        if ans_tokenizer is None:
            ans_tokenizer = ans_model
        
        # Instantiate tokenizer if needed
        if isinstance(ans_tokenizer, (str, tuple)):
            if isinstance(ans_tokenizer, tuple):
                # For tuple we have (tokenizer name, {kwargs})
                ans_tokenizer = AutoTokenizer.from_pretrained(ans_tokenizer[0], **ans_tokenizer[1])
            else:
                ans_tokenizer = AutoTokenizer.from_pretrained(ans_tokenizer)

        if isinstance(ans_model, str):
            ans_model = AutoModelForSeq2SeqLM.from_pretrained(ans_model)
    
    return QGPipeline(model=model, tokenizer=tokenizer, ans_model=ans_model, ans_tokenizer=ans_tokenizer)