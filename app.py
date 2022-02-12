from pipeline import QGPipeline

from transformers import(
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
)

def main():
    tokenizer = AutoTokenizer.from_pretrained("valhalla/t5-small-qg-hl")
    model = AutoModelForSeq2SeqLM.from_pretrained("valhalla/t5-small-qg-hl")
    ans_tokenizer = AutoTokenizer.from_pretrained("valhalla/t5-small-qa-qg-hl")
    ans_model = AutoModelForSeq2SeqLM.from_pretrained("valhalla/t5-small-qa-qg-hl")

    get_questions = QGPipeline(model=model, tokenizer=tokenizer, ans_model=ans_model, ans_tokenizer=ans_tokenizer)

    while(True):
        print()
        print("Please input a question:")
        text = input()
        questions = get_questions(text)
        for i, question in enumerate(questions):
            print()
            print('########################################################################')
            print('Question #{}: {}'.format(i + 1, question['question']))
            print('The answers is: {}'.format(question['answer']))
            print('The possible wrong answers are:')
            if question['distractors'] is None:
                print('No wrong answers')
            else:
                for distractor in question['distractors']:
                    print('---- {}'.format(distractor))


if __name__ == "__main__":
    main()