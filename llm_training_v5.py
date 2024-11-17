import pandas as pd
import numpy as np
import torch
import argparse
import os
import random
import yaml
from ast import literal_eval
from dataclasses import dataclass
from functools import partial
from transformers import (
    BitsAndBytesConfig,
    Trainer, 
    TrainingArguments,
    EvalPrediction,
    AutoTokenizer,
    GemmaTokenizerFast,
    DataCollatorWithPadding
)
from modeling import Gemma2ForSequenceClassification, Gemma2ForMultiTaskClassification, LlamaForSequenceClassification, LlamaForMultiTaskClassification
from peft import prepare_model_for_kbit_training, LoraConfig, TaskType, PeftModelForSequenceClassification
from datasets import load_dataset, Dataset
from trl import SFTConfig, SFTTrainer
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, accuracy_score
from scipy.special import expit, softmax

import gc

from accelerate import Accelerator

import warnings
warnings.filterwarnings('ignore')

accelerator = Accelerator()


def process_prompt(input_str):
    stripped_str = input_str.strip('[]')
    sentences = [s.strip('"') for s in stripped_str.split('","')]
    return " ".join(sentences)

def process_response(input_str):
    stripped_str = input_str.strip('[]')
    sentences = [s.strip('"') for s in stripped_str.split('","')]
    return sentences[-1]

def prepare_prompt(row, max_len=600):
    prompt = row['prompt']
    response_a = row['response_a']
    response_b = row['response_b']
    if len(prompt) > max_len:
        prompt = prompt[:max_len] + "<cont>"
    if len(response_a) > max_len:
        response_a = response_a[-max_len:] + "<cont>"
    if len(response_b) > max_len:
        response_b = response_b[-max_len:] + "<cont>"

    prmpt = f"Prompt: {prompt}\n\nResponse_a:  {response_a}\n\nResponse_b: {response_b}"
    
    return prmpt

def prepare_prompt_aug(row, max_len=600):
    prompt = row['prompt']
    response_a = row['response_a']
    response_b = row['response_b']
    if len(prompt) > max_len:
        prompt = prompt[:max_len] + "<cont>"
    if len(response_a) > max_len:
        response_a = response_a[-max_len:] + "<cont>"
    if len(response_b) > max_len:
        response_b = response_b[-max_len:] + "<cont>"
    
    prmpt = f"Prompt: {prompt}\n\nResponse_a:  {response_b}\n\nResponse_b: {response_a}"
    
    return prmpt

# def prepare_tokenized(examples, max_length, tokenizer):
#     tokenized_samples = tokenizer(examples["text"], truncation=True, max_length=max_length)
#     return tokenized_samples


def prepare_tokenized(data, tokenizer, max_length=2048, spread_max_length=False):
    if isinstance(tokenizer, GemmaTokenizerFast):
        prompt = ["<prompt>: " + p for p in data['prompt']]
        response_a = ["\n\n<response_a>: " + r_a for r_a in data['response_a']]
        response_b = ["\n\n<response_b>: " + r_b for r_b in data['response_b']]
    else:
        prompt = ["User prompt: " + p for p in data['prompt']]
        response_a = ["\n\nModel A: \n" + r_a for r_a in data['response_a']]
        response_b = ["\n\nModel B: \n" + r_b for r_b in data['response_b']]

    if spread_max_length:
        prompt = tokenizer(prompt, max_length=prompt_len, truncation=True, padding=False).input_ids
        response_a = tokenizer(response_a, max_length=response_len, truncation=True, padding=False).input_ids
        response_b = tokenizer(response_b, max_length=response_len, truncation=True, padding=False).input_ids
        input_ids = [p + r_a + r_b for p, r_a, r_b in zip(prompt, response_a, response_b)]
        attention_mask = [[1]* len(i) for i in input_ids]
    else:
        prompt_len = max_length//4
        prompt = [p[:prompt_len] + "<cont>" if len(p) > prompt_len else p for p in prompt]

        response_len = [(max_length - len(p))//2 for p in prompt]
        response_a = [r[:rl] + "<cont>" if len(r)>rl else r for r, rl in zip(response_a, response_len)]
        response_b = [r[:rl] + "<cont>" if len(r)>rl else r for r, rl in zip(response_b, response_len)]

        text = [p + r_a + r_b for p, r_a, r_b in zip(prompt, response_a, response_b)]
        tokenized = tokenizer(text, max_length=max_length, truncation=True, padding=False)
        input_ids = tokenized.input_ids
        attention_mask = tokenized.attention_mask
    data.update({'input_ids': input_ids, 'attention_mask': attention_mask})
    return data


def compute_metrics(p: EvalPrediction):
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    if len(p.label_ids.shape)> 1 and p.label_ids.shape[1] > 1:
        _, labels = np.where(p.label_ids)
        labels = labels.astype('int64')
    else:
        labels = p.label_ids
    # probs = torch.from_numpy(preds).float().softmax(-1).numpy()
    loss = torch.nn.functional.cross_entropy(torch.from_numpy(preds).float(), torch.tensor(labels).squeeze().long())
    loss_11 = torch.nn.functional.cross_entropy(torch.from_numpy(preds).float()/1.1, torch.tensor(labels).squeeze().long())
    loss_15 = torch.nn.functional.cross_entropy(torch.from_numpy(preds).float()/1.5, torch.tensor(labels).squeeze().long())
    logits = torch.from_numpy(preds).float()
    
    probs = torch.softmax(logits, 1)
    targets = torch.nn.functional.one_hot(torch.tensor(labels).squeeze().long())
    ll = log_loss(targets, probs)
    
    probs2 = torch.softmax(logits/1.1, 1)
    ll2 = log_loss(targets, probs2)
    
    probs3 = torch.softmax(logits/1.5, 1)
    ll3 = log_loss(targets, probs3)
    
    probs4 = torch.softmax(logits[:, :-1], 1)
    probs5 = torch.sigmoid(logits[:, -1:])
    probs4 = (1-probs5)*probs4
    probs6 = torch.concat([probs4, probs5], dim=1)
    probs6 = probs6 / probs6.sum(axis=1, keepdim=True)
    ll4 = log_loss(targets, probs6)
    # acc = accuracy_score(y_true=labels, y_pred=preds.argmax(-1))
    return {"ce_loss": loss, "log_loss": ll, "log_loss_11": ll2, "log_loss_15": ll3, "log_loss_z": ll4}


def main(args):
    
    df = pd.read_csv("data/train.csv")
    if args.sanity:
        df = df.sample(200)
        args.experiment_name = "sanity"
        
    prepare_prompt_part = partial(prepare_prompt, max_len=args.max_length//3)
    prepare_prompt_aug_part = partial(prepare_prompt_aug, max_len=args.max_length//3)

    df['prompt'] = df['prompt'].fillna("").apply(process_prompt)
    df['response_a'] = df['response_a'].fillna("").apply(process_prompt)
    df['response_b'] = df['response_b'].fillna("").apply(process_prompt)
    

    df2 = pd.read_csv("data/lmsys-33k-deduplicated.csv")
    if args.sanity:
        df2 = df2.sample(200)
    df2['prompt'] = df2['prompt'].fillna("").apply(process_prompt)
    df2['response_a'] = df2['response_a'].fillna("").apply(process_prompt)
    df2['response_b'] = df2['response_b'].fillna("").apply(process_prompt)
    
    ## Load tokenizer
    if "gemma" in args.base_model_path:
        tokenizer = GemmaTokenizerFast.from_pretrained(args.base_model_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.base_model_path)
    
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    tokenizer.truncation_side = "left"
     
    if args.sanity:
        train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    else:
        ## Perform train val split
        folds = torch.load('data/3_folds.torch')
        train_df = df.loc[folds[args.foldid]['train']].copy()
        val_df = df.loc[folds[args.foldid]['val']].copy()    
    
    
    if args.use_extra_data:
        train_df = pd.concat([train_df, df2[train_df.columns]])

    if args.problem_type == 'multi_task_classification':
        train_df['labels'] = train_df[['winner_model_a', 'winner_model_b', 'winner_tie']].values.tolist()
        train_df['model_a']= train_df['model_a'].astype('category')
        train_df['model_b']= train_df['model_b'].astype('category')
        train_df['labels1'] = train_df['model_a'].cat.codes.values.tolist()
        train_df['labels2'] = train_df['model_b'].cat.codes.values.tolist()
    elif args.problem_type == 'single_label_classification':
        _, lbl = np.where(train_df[['winner_model_a', 'winner_model_b', 'winner_tie']].values)
        train_df['labels'] = lbl.astype('int64').tolist()
    else:
        train_df['labels'] = train_df[['winner_model_a', 'winner_model_b', 'winner_tie']].values.tolist()
        
    if args.problem_type == 'multi_task_classification':
        val_df['labels'] = val_df[['winner_model_a', 'winner_model_b', 'winner_tie']].values.tolist()
    elif args.problem_type == 'single_label_classification':
        _, lbl = np.where(val_df[['winner_model_a', 'winner_model_b', 'winner_tie']].values)
        val_df['labels'] = lbl.astype('int64').tolist()
    else:
        val_df['labels'] = val_df[['winner_model_a', 'winner_model_b', 'winner_tie']].values.tolist()

        
    aut_train_df = train_df.copy()
    aut_train_df['response_a'] = train_df['response_b'].values
    aut_train_df['response_b'] = train_df['response_a'].values
    if args.problem_type == 'multi_task_classification':
        aut_train_df['labels'] = aut_train_df[['winner_model_b', 'winner_model_a', 'winner_tie']].values.tolist()
        aut_train_df['model_a']= aut_train_df['model_a'].astype('category')
        aut_train_df['model_b']= aut_train_df['model_b'].astype('category')
        aut_train_df['labels1'] = aut_train_df['model_b'].cat.codes.values.tolist()
        aut_train_df['labels2'] = aut_train_df['model_a'].cat.codes.values.tolist()
    elif args.problem_type == 'single_label_classification':
        _, lbl2 = np.where(aut_train_df[['winner_model_b', 'winner_model_a', 'winner_tie']].values)
        aut_train_df['labels'] = lbl2.astype('int64').tolist()
    else:
        aut_train_df['labels'] = aut_train_df[['winner_model_b', 'winner_model_a', 'winner_tie']].values.tolist()
    
    if args.use_aug_data:
        train_df = pd.concat([train_df, aut_train_df])
    
    train_df.reset_index(inplace=True, drop=True)
    val_df.reset_index(inplace=True, drop=True)
    
    if args.problem_type == 'multi_task_classification':
        train_ds = Dataset.from_list(train_df[['prompt', 'response_a', 'response_b', 'labels', 'labels1', 'labels2']].to_dict('records'), split="train")
    else:
        train_ds = Dataset.from_list(train_df[['prompt', 'response_a', 'response_b', 'labels']].to_dict('records'), split="train")
    val_ds = Dataset.from_list(val_df[['prompt', 'response_a', 'response_b', 'labels']].to_dict('records'), split="test")
    
    train_tokenized_ds = train_ds.map(prepare_tokenized, batched=True, 
                                fn_kwargs={"max_length": args.max_length, "tokenizer": tokenizer, "spread_max_length": False},
                                remove_columns=['prompt', 'response_a', 'response_b'])

    val_tokenized_ds = val_ds.map(prepare_tokenized, batched=True, 
                                fn_kwargs={"max_length": args.max_length, "tokenizer": tokenizer, "spread_max_length": False},
                                remove_columns=['prompt', 'response_a', 'response_b'])
    
    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
        bnb_8bit_use_double_quant=False,
        bnb_8bit_compute_dtype=torch.float16,
        llm_int8_skip_modules=["score", "correctness", "classifier", "model_pred1", "model_pred2", "proj"]
    )
    
    
    if args.problem_type == 'multi_task_classification':
        num_model_types = max(len(pd.unique(train_df['model_a'])), len(pd.unique(train_df['model_b'])))
        print(num_model_types)
        if "gemma" in args.base_model_path:
            base_model = Gemma2ForMultiTaskClassification.from_pretrained(
                args.base_model_path,
                quantization_config=bnb_config,
                torch_dtype=torch.float16, 
                trust_remote_code=True,
                problem_type=args.problem_type, 
                num_labels=3,
                num_model_types = num_model_types,
            )
        elif "llama" in args.base_model_path:
            base_model = LlamaForMultiTaskClassification.from_pretrained(
                args.base_model_path,
                quantization_config=bnb_config,
                torch_dtype=torch.float16, 
                trust_remote_code=True,
                problem_type=args.problem_type, 
                num_labels=3,
                num_model_types = num_model_types,
            )
    else:
        if "gemma" in args.base_model_path:
            base_model = Gemma2ForSequenceClassification.from_pretrained(
                args.base_model_path,
                quantization_config=bnb_config,
                torch_dtype=torch.float16, 
                trust_remote_code=True,
                problem_type=args.problem_type, 
                num_labels=3,
            )
        elif "llama" in args.base_model_path:
            base_model = LlamaForSequenceClassification.from_pretrained(
                args.base_model_path,
                quantization_config=bnb_config,
                torch_dtype=torch.float16, 
                trust_remote_code=True,
                problem_type=args.problem_type, 
                num_labels=3,
            )

    base_model.config.pad_token_id = tokenizer.pad_token_id
    base_model.config.use_cache = False
    
    base_model = prepare_model_for_kbit_training(base_model)

    peft_config = LoraConfig(
        r=32,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.SEQ_CLS,
        inference_mode=False,
        target_modules = [
            'q_proj', 'k_proj', 'v_proj', 'o_proj'
        ],
        layers_to_transform=[i for i in range(len(base_model.model.layers)) if i >= args.freeze_layers],
        # init_lora_weights="gaussian",
        use_rslora=args.use_rslora,
        modules_to_save=["score", "model_pred1", "model_pred2"]
    )

    
    # base_model.enable_input_require_grads()
    if args.lora_path == "":
        model = PeftModelForSequenceClassification(base_model, peft_config)
    else:
        model = PeftModelForSequenceClassification.from_pretrained(base_model, args.lora_path, is_trainable=True)
    print(model.print_trainable_parameters())
    
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding="longest", return_tensors="pt")

    train_config = TrainingArguments(
        f"checkpoints/{args.experiment_name}",
        eval_strategy = "steps",
        save_strategy = "steps",
        learning_rate=args.lr,
        lr_scheduler_type='cosine',
        warmup_ratio=args.warmup_ratio,
        # label_smoothing_factor=0.1,
        # optim='adamw_8bit',
        # weight_decay=0.01,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        fp16=True,
        num_train_epochs=args.num_train_epochs,
        gradient_checkpointing=False,
        gradient_checkpointing_kwargs={'use_reentrant': False},
        # load_best_model_at_end=True,
        logging_steps=20,
        save_steps=10 if args.sanity else 50,
        eval_steps=5 if args.sanity else 50,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        report_to='tensorboard',
        ddp_find_unused_parameters=False
    )

    trainer = Trainer(
        model,
        train_dataset=train_tokenized_ds,
        eval_dataset=val_tokenized_ds,
        args=train_config,
        compute_metrics=compute_metrics,
        data_collator=data_collator,
    )

    trainer.train()
    
    
@dataclass
class Config:
    base_model_path: str
    experiment_name: str
    foldid: int = 0
    max_length: int = 2048
    batch_size: int = 2
    num_train_epochs: int = 10
    lr: float = 1e-4
    freeze_layers: int = 16
    sanity: bool = False
    problem_type: str = "single_label_classification"
    use_extra_data: bool = False
    use_aug_data: bool = True
    gradient_accumulation_steps: int = 5
    warmup_ratio: int = 0.03
    use_rslora: bool = False
    lora_path: str = ''
    
    
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--config_path', type=str, required=True)
    args = ap.parse_args()
    with open(args.config_path, 'r') as _f:
        config_d = yaml.safe_load(_f)
    config = Config(**config_d)
    main(config)
