import itertools

from nltk import sent_tokenize
import random
import torch
from transformers import(
    PreTrainedModel,
    PreTrainedTokenizer,
)
from collections import OrderedDict
from sense2vec import Sense2Vec

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
        self.s2v = Sense2Vec().from_disk('sense2vec/s2v_old')

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

        qg_examples = self.prepare_inputs_for_qg_from_answers_hl(sents, answers)

        qg_inputs = [example['source_text'] for example in qg_examples]
        questions = self.generate_questions(qg_inputs)
        output = []
        for example, que in zip(qg_examples, questions):
            distractors = self.sense2vec_get_words(example['answer'])
            distractors = None if distractors is None else random.choices(distractors,  k=3)
            output.append(
                {
                    'answer': example['answer'],
                    'question': que,
                    'distractors': distractors
                }
            )
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

    def sense2vec_get_words(self, word):
        output = []
        word = word.lower()
        word = word.replace(" ", "_")

        sense = self.s2v.get_best_sense(word)
        if sense is None:
            return None
        most_similar = self.s2v.most_similar(sense, n=20)

        for each_word in most_similar:
            append_word = each_word[0].split("|")[0].replace("_", " ").lower()
            if append_word.lower() != word:
                output.append(append_word.title())

        out = list(OrderedDict.fromkeys(output))
        return out

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