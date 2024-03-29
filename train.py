import json
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union, Callable, Iterable

import numpy as np
from typing import Dict, List, Optional

import torch
from torch import nn

from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    T5Tokenizer,
    BartTokenizer,
    HfArgumentParser,
    DataCollator,
    TrainingArguments,
    Trainer as HFTrainer,
    set_seed
)

from transformers.file_utils import is_apex_available

from utils import freeze_embeds, assert_not_all_frozen, label_smoothed_nll_loss

if is_apex_available():
    from apex import amp

MODEL_TYPE_TO_TOKENIZER = {
    "t5": T5Tokenizer,
    "bart": BartTokenizer,
}

model_type = 't5'
model_name = 't5-small'
tokenizer_name = 't5_qg_tokenizer'
trainset_path = 'data/train_data_qg_hl_t5.pt'
validset_path = 'data/valid_data_qg_hl_t5.pt'

def trim_batch(
    input_ids, pad_token_id, attention_mask=None,
):
    """Remove columns that are populated exclusively by pad_token_id"""
    keep_column_mask = input_ids.ne(pad_token_id).any(dim=0)
    if attention_mask is None:
        return input_ids[:, keep_column_mask]
    else:
        return (input_ids[:, keep_column_mask], attention_mask[:, keep_column_mask])


# prepares lm_labels from target_ids, returns examples with keys as expected by the forward method
# this is necessary because the trainer directly passes this dict as arguments to the model
# so make sure the keys match the parameter names of the forward method
class T2TDataCollator():
    def __init__(self, tokenizer, model_type="t5", mode='training', using_tpu=False):
        self.tokenizer = tokenizer
        self.model_type = model_type
        self.mode = mode
        self.using_tpu = False

    def __call__(self, batch: List) -> Dict[str, torch.Tensor]:
        """
        Take a list of samples from a Dataset and collate them into a batch.
        Returns:
            A dictionary of tensors
        """
        input_ids = torch.stack([example['source_ids'] for example in batch])
        target_ids = torch.stack([example['target_ids'] for example in batch])
        attention_mask = torch.stack([example['attention_mask'] for example in batch])

        pad_token_id = self.tokenizer.pad_token_id
        
        # don't trim on tpu, for some reason trimming leads to slower training on TPU
        if not self.using_tpu:
            input_ids, attention_mask = trim_batch(input_ids, pad_token_id, attention_mask=attention_mask)
            target_ids = trim_batch(target_ids, pad_token_id)
        
        if self.model_type == "t5":
            lm_labels = target_ids.clone()
            decoder_input_ids = self._shift_right_t5(lm_labels)
            if self.mode == 'training':
                lm_labels[lm_labels[:, :] == pad_token_id] = -100
        else:
            decoder_input_ids = target_ids[:, :-1].contiguous()
            lm_labels = target_ids[:, 1:].clone()
            if self.mode == 'training':
                lm_labels[target_ids[:, 1:] == pad_token_id] = -100

        params =  {
            "input_ids": input_ids, 
            "attention_mask": attention_mask,
            "labels": lm_labels,
            "decoder_input_ids": decoder_input_ids
        }
        
        return params
    
    def _shift_right_t5(self, input_ids):
        decoder_start_token_id = self.tokenizer.pad_token_id
        pad_token_id = self.tokenizer.pad_token_id

        # shift inputs to the right
        shifted_input_ids = input_ids.new_zeros(input_ids.shape)
        shifted_input_ids[..., 1:] = input_ids[..., :-1].clone()
        shifted_input_ids[..., 0] = decoder_start_token_id

        # replace possible -100 values in labels by `pad_token_id`
        shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

        return shifted_input_ids

class Trainer(HFTrainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    # override to support label smoothing
    def _training_step(
        self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]], optimizer: torch.optim.Optimizer
    ) -> float:
        model.train()
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(self.args.device)


        # Our model outputs do not work with DataParallel, so forcing return tuple.
        if isinstance(model, nn.DataParallel):
            inputs["return_tuple"] = True

        labels = inputs.pop("labels")
        labels[labels == -100] = model.config.pad_token_id
        outputs = model(**inputs)
        lprobs = torch.nn.functional.log_softmax(outputs[0], dim=-1)
        loss, nll_loss = label_smoothed_nll_loss(
            lprobs, labels, 0, ignore_index=model.config.pad_token_id
        )

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training
        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps

        if self.args.fp16:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        return loss.item()

def main():
    parser = HfArgumentParser(TrainingArguments)

    training_args = parser.parse_args_into_dataclasses()[0]

    if os.path.exists(training_args.output_dir) and not training_args.overwrite_output_dir:
        raise ValueError(
            f"Directory exists. Add --overwrite_output_dir."
        )

    set_seed(training_args.seed)

    os.environ["WANDB_PROJECT"] = "question-generation"

    tokenizer_cls = MODEL_TYPE_TO_TOKENIZER[model_type]
    tokenizer = tokenizer_cls.from_pretrained(
        tokenizer_name if tokenizer_name else model_name
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name
    )

    model.resize_token_embeddings(len(tokenizer))
    
    train_dataset = torch.load(trainset_path) if training_args.do_train else None
    valid_dataset = torch.load(validset_path) if training_args.do_eval else None
    
    # Initialize data_collator
    data_collator = T2TDataCollator(
        tokenizer=tokenizer,
        model_type=model_type,
        mode="training",
        using_tpu=False
    )

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=data_collator,
        prediction_loss_only=True
    )

    # Training
    if training_args.do_train:
        trainer.train(
            model_path=model_name
        )

        trainer.save_model()
        
        if trainer.is_world_master():
            tokenizer.save_pretrained(training_args.output_dir)

    # Evaluation
    results = {}
    if training_args.do_eval:
        eval_output = trainer.evaluate()
        output_eval_file = os.path.join(training_args.output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            for key in sorted(eval_output.keys()):
                writer.write("%s = %s\n" % (key, str(eval_output[key])))

        results.update(eval_output)
    
    return results

if __name__ == "__main__":
    main()