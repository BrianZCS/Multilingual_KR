import sys
from os.path import dirname, abspath
sys.path.insert(0, dirname(dirname(dirname(abspath(__file__)))))

from typing import List, Dict, Tuple, Set, Union
import traceback
import torch
from transformers import *
import transformers
import json
import numpy as np
import os
from tqdm import tqdm
import argparse
import logging
from collections import defaultdict
import pandas
import csv
import time
import re
from prompt import Prompt
from check_gender import load_entity_gender, Gender
from check_instanceof import load_entity_instance, load_entity_is_cate
from entity_lang import Alias, MultiRel
from tokenization_kobert import KoBertTokenizer

from fvt.fvt import FastVocabularyTransfer

logger = logging.getLogger('mLAMA')
logger.setLevel(logging.ERROR)

SUB_LABEL = '##'
PREFIX_DATA = '../LAMA/'
VOCAB_PATH = PREFIX_DATA + 'pre-trained_language_models/common_vocab_cased.txt'
RELATION_PATH = 'data/TREx-relations.jsonl'
PROMPT_LANG_PATH = 'data/TREx_prompts.csv'
LM_NAME = {
    # multilingual model
    'mbert_base': 'bert-base-multilingual-cased',
    'xlm_base': 'xlm-mlm-100-1280',
    'xlmr_base': 'xlm-roberta-base',
    # language-specific model
    'bert_base': 'bert-base-cased',
    'fr_roberta_base': 'camembert-base',
    'nl_bert_base': 'bert-base-dutch-cased',
    'es_bert_base': 'dccuchile/bert-base-spanish-wwm-cased',
    'ru_bert_base': 'pretrain/rubert_cased_L-12_H-768_A-12_v2',
    # 'DeepPavlov/rubert-base-cased' doesn't include lm head
    'zh_bert_base': 'bert-base-chinese',
    'tr_bert_base': 'dbmdz/bert-base-turkish-cased',
    'ko_bert_base': 'monologg/kobert-lm',
    # 'monologg/kobert' doesn't include lm head
    'el_bert_base': 'nlpaueb/bert-base-greek-uncased-v1'
}
DATASET = {
    'lama': {
        'entity_path': 'data/TREx/{}.jsonl',
        'entity_lang_path': 'data/TREx_unicode_escape.txt',
        'entity_gender_path': 'data/TREx_gender.txt',
        'entity_instance_path': 'data/TREx_instanceof.txt',
        'alias_root': 'data/alias/TREx',
        'multi_rel': 'data/TREx_multi_rel.txt',
        'is_cate': 'data/TREx_is_cate.txt',
    },
    'lama-uhn': {
        'entity_path': 'data/TREx_UHN/{}.jsonl',
        'entity_lang_path': 'data/TREx_unicode_escape.txt',
        'entity_gender_path': 'data/TREx_gender.txt',
        'entity_instance_path': 'data/TREx_instanceof.txt',
        'alias_root': 'data/alias/TREx',
        'multi_rel': 'data/TREx_multi_rel.txt',
        'is_cate': 'data/TREx_is_cate.txt',
    },
    'mlama': {
        'entity_path': 'data/mTREx/sub/{}.jsonl',
        'entity_lang_path': 'data/mTREx_unicode_escape.txt',
        'entity_gender_path': 'data/mTREx_gender.txt',
        'entity_instance_path': 'data/mTREx_instanceof.txt',
        'alias_root': 'data/alias/mTREx',
        'multi_rel': 'data/mTREx_multi_rel.txt',
        'is_cate': 'data/mTREx_is_cate.txt',
    },
    'mlamaf': {
        'entity_path': 'data/mTRExf/sub/{}.jsonl',
        'entity_lang_path': 'data/mTRExf_unicode_escape.txt',
        'entity_gender_path': 'data/mTRExf_gender.txt',
        'entity_instance_path': 'data/mTRExf_instanceof.txt',
        'alias_root': 'data/alias/mTRExf',
        'multi_rel': 'data/mTRExf_multi_rel.txt',
        'is_cate': 'data/mTRExf_is_cate.txt',
    }
}


def get_tokenizer(lang: str, name: str):
    if lang == 'en':
        tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
    elif lang == 'zh':
        tokenizer = AutoTokenizer.from_pretrained('bert-base-chinese')
    elif lang == 'es':
        tokenizer = AutoTokenizer.from_pretrained('dccuchile/bert-base-spanish-wwm-cased')
    elif lang == 'tr':
        tokenizer = AutoTokenizer.from_pretrained('dbmdz/bert-base-turkish-cased')
    if name in {'xlm-mlm-100-1280', 'xlm-roberta-base'}:
        tokenizer = AutoTokenizer.from_pretrained(name)
    print(tokenizer.__class__.__name__, ', Vocab Size', tokenizer.vocab_size)
    return tokenizer


if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description='probe LMs with multilingual LAMA')
    # parser.add_argument('--model', type=str, help='LM to probe file', default='mbert_base')
    # parser.add_argument('--lm_layer_model', type=str,
    #                     help='LM from which the final lm layer is used', default=None)
    # parser.add_argument('--lang', type=str, help='language to probe',
    #                     choices=['en', 'fr', 'nl', 'es', 'zh',
    #                              'mr', 'vi', 'ko', 'he', 'yo',
    #                              'el', 'tr', 'ru',
    #                              'ja', 'hu', 'bn', 'war', 'tl', 'sw',
    #                              'mg', 'pa', 'ilo', 'ceb'], default='en')
    # parser.add_argument('--sent', type=str, help='actual sentence with [Y]', default=None)

    # # dataset-related flags
    # parser.add_argument('--probe', type=str, help='probe dataset',
    #                     choices=['lama', 'lama-uhn', 'mlama', 'mlamaf'], default='mlamaf')
    # parser.add_argument('--pids', type=str, help='pids to run', default=None)
    # parser.add_argument('--portion', type=str, choices=['all', 'trans', 'non'], default='trans',
    #                     help='which portion of facts to use')
    # parser.add_argument('--facts', type=str, help='file path to facts', default=None)
    # parser.add_argument('--prompts', type=str, default=None,
    #                     help='directory where multiple prompts are stored for each relation')
    # parser.add_argument('--sub_obj_same_lang', action='store_true',
    #                     help='use the same language for sub and obj')
    # parser.add_argument('--skip_multi_word', action='store_true',
    #                     help='skip objects with multiple words (not sub-words)')
    # parser.add_argument('--skip_single_word', action='store_true',
    #                     help='skip objects with a single word')

    # # inflection-related flags
    # parser.add_argument('--prompt_model_lang', type=str, help='prompt model to use',
    #                     choices=['en', 'el', 'ru', 'es', 'mr'], default=None)
    # parser.add_argument('--disable_inflection', type=str, choices=['x', 'y', 'xy'])
    # parser.add_argument('--disable_article', action='store_true')

    # # decoding-related flags
    # parser.add_argument('--num_mask', type=int, help='the maximum number of masks to insert', default=5)
    # parser.add_argument('--max_iter', type=int, help='the maximum number of iteration in decoding', default=1)
    # parser.add_argument('--init_method', type=str, help='iteration method', default='all')
    # parser.add_argument('--iter_method', type=str, help='iteration method', default='none')
    # parser.add_argument('--no_len_norm', action='store_true', help='not use length normalization')
    # parser.add_argument('--reprob', action='store_true', help='recompute the prob finally')
    # parser.add_argument('--beam_size', type=int, help='beam search size', default=1)

    # # others
    # parser.add_argument('--use_gold', action='store_true', help='use gold objects')
    # parser.add_argument('--dry_run', type=int, help='dry run the probe to show inflection results', default=None)
    # parser.add_argument('--log_dir', type=str, help='directory to vis prediction results', default=None)
    # parser.add_argument('--pred_dir', type=str, help='directory to store prediction results', default=None)
    # parser.add_argument('--batch_size', type=int, help='the real batch size is this times num_mask', default=20)
    # parser.add_argument('--no_cuda', action='store_true', help='not use cuda')
    # args = parser.parse_args()

    # if (args.init_method != 'all' or args.iter_method != 'none') and args.max_iter:
    #     assert args.max_iter >= args.num_mask, 'the results will contain mask'
    # if args.sent:
    #     args.batch_size = 1
    #     args.pids = 'P19'

    # LM = LM_NAME[args.model] if args.model in LM_NAME else args.model  # use pre-defined models or path

    # # load data
    # print('load data')
    # tokenizer = get_tokenizer(args.lang, LM)
    # original_tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')#AutoTokenizer.from_pretrained(LM)
    # probe_iter = ProbeIterator(args, tokenizer)
    # pretrained_model = AutoModelWithLMHead.from_pretrained(LM)
    
    # fvt = FastVocabularyTransfer()
    # model = fvt.transfer(
    #     in_tokenizer=tokenizer,
    #     gen_tokenizer=original_tokenizer,
    #     gen_model=pretrained_model
    # ) 
    
    from datasets import load_dataset
    from datasets import load_from_disk
    print("hello")
    datasets = load_from_disk('zhwiki')#load_dataset('shaowenchen/wiki_zh')#load_dataset('suolyer/wiki_zh')#load_dataset('wikitext', 'wikitext-2-raw-v1')#shaowenchen/wiki_zh, streaming=True
    print("finish loading")
    #print(datasets["train"][10])

    from datasets import ClassLabel
    import random
    import pandas as pd

    # def show_random_elements(dataset, num_examples=10):
    #     assert num_examples <= len(dataset), "Can't pick more elements than there are in the dataset."
    #     picks = []
    #     for _ in range(num_examples):
    #         pick = random.randint(0, len(dataset)-1)
    #         while pick in picks:
    #             pick = random.randint(0, len(dataset)-1)
    #         picks.append(pick)
        
    #     df = pd.DataFrame(dataset[picks])
    #     for column, typ in dataset.features.items():
    #         if isinstance(typ, ClassLabel):
    #             df[column] = df[column].transform(lambda i: typ.names[i])
    def filter_fn(example):
        return int(example['id']) < 300

    def tokenize_function(examples):
        return tokenizer(examples["text"])
        #examples['input_ids'] = tokenizer.tokenize(examples["text"])
        #examples['attention_mask'] = [1 for i in range(len(examples['input_ids']))]
    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
            # customize this part to your needs.
        total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result
    # print(show_random_elements(datasets["train"]))
    model_checkpoint = "bert-base-multilingual-cased"
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)
    block_size = tokenizer.model_max_length
    datasets = datasets.filter(filter_fn)
    tokenized_datasets = datasets.map(tokenize_function, batched=True, batch_size=1, num_proc=4, remove_columns=["text","id", "url","title"])
    
    
    lm_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
        batch_size=1,
        num_proc=4,
    )
    #print(lm_datasets["train"][1])
    from transformers import AutoModelForMaskedLM
    model = AutoModelForMaskedLM.from_pretrained(model_checkpoint)
    #device = #torch.device("cuda:0,cuda:1,cuda:2,cuda:3,cuda:4,cuda:5")
    #model = torch.nn.DataParallel(model, device_ids=[0, 1, 2, 3, 4, 5]).cuda()
    model_name = model_checkpoint.split("/")[-1]
    training_args = TrainingArguments(
        f"{model_name}-finetuned-wikich",
        evaluation_strategy = "epoch",
        learning_rate=2e-5,
        weight_decay=0.01,
        # push_to_hub=True,
        remove_unused_columns=False,
        per_device_train_batch_size=1,#
        per_device_eval_batch_size=1,#
    )
    from transformers import DataCollatorForLanguageModeling
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)
    splits = lm_datasets["train"].train_test_split(test_size=0.1)
    trainer = Trainer(
        model=model,#.to(torch.device("cuda:0")),
        args=training_args,
        train_dataset=splits["train"],#.with_format("torch", device=torch.device("cuda:0")),
        eval_dataset=splits["test"],#.with_format("torch", device=torch.device("cuda:0")),
        data_collator=data_collator,
    )
    trainer.train()
    trainer.save_model("./mbert_base")
    trainer.log_metrics("train", train_results.metrics)
    trainer.save_metrics("train", train_results.metrics)
    trainer.save_state()
    # Evaluate
    metrics = trainer.evaluate()
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)
    print(metrics)
