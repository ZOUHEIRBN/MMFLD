# -*- coding:utf-8 _*-

import os
import sys
import time
import math
import random
import argparse
import numpy as np
from sklearn.metrics import (
    confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score)

import tqdm
import torch
from torch import cuda
from transformers import logging
from transformers import (
    MT5TokenizerFast,
    MT5ForConditionalGeneration)

from polynomial_lr_decay import PolynomialLRDecay
import csv, json

logging.set_verbosity_error()
os.environ["TOKENIZERS_PARALLELISM"] = "false"
device = 'cuda' if cuda.is_available() else 'cpu'

def read_insts(mode, lang, form, prompt, tokenizer, max_len=200, ups_num=5000):
    """
    Read instances
    """
    src, tgt = [], []
    literal = tokenizer.encode('Literal')
    figure = tokenizer.encode(form.capitalize())
    if len(prompt) > 0:
        prompt = prompt.format(form.capitalize())

    path = 'data/{}_{}_{}.{}'.format(mode, lang, form, '{}')
    with open(path.format(0), 'r') as f0, \
            open(path.format(1), 'r') as f1:
        f0 = f0.readlines()
        f1 = f1.readlines()
        if mode != 'test':

            # Keep label distribution balanced
            if len(f0) > len(f1):
                f0 = f0[:len(f1)]
            else:
                f1 = f1[:len(f0)]

            # upsample
            if mode == 'train' and len(f0) < ups_num:
                f0 = (f0 * math.ceil(ups_num / len(f0)))[:ups_num]
                f1 = (f1 * math.ceil(ups_num / len(f1)))[:ups_num]

        for seqs, label in zip([f0, f1], [literal, figure]):
            for seq in seqs:
                seq = tokenizer.encode(prompt + seq.strip())
                src.append(seq[:min(len(seq) - 1, max_len)] + seq[-1:])
                tgt.append(label)

    return src, tgt


def get_collate_fn(tokenizer):
    def collate_fn(insts):
        """
        Pad the instance to the max seq length in batch
        """
    
        pad_id = tokenizer.pad_token_id
        max_len = max(len(inst) for inst in insts)
    
        batch_seq = [inst + [pad_id] * (max_len - len(inst))
                     for inst in insts]
        batch_seq = torch.LongTensor(batch_seq)
    
        return batch_seq
    return collate_fn

def get_paired_collate_fn(tokenizer):
    def paired_collate_fn(insts):
        src_inst, tgt_inst = list(zip(*insts))
        collate_fn = get_collate_fn(tokenizer)
        src_inst = collate_fn(src_inst)
        tgt_inst = collate_fn(tgt_inst)
    
        return src_inst, tgt_inst
    return paired_collate_fn


class MMFLUDataset(torch.utils.data.Dataset):
    """ Seq2Seq Dataset """

    def __init__(self, src_inst, tgt_inst):
        self.src_inst = src_inst
        self.tgt_inst = tgt_inst

    def __len__(self):
        return len(self.src_inst)

    def __getitem__(self, idx):
        return self.src_inst[idx], self.tgt_inst[idx]


def MMFLUIterator(src, tgt, batch_size, tokenizer, shuffle=True):
    """
    Data iterator for classifier
    """

    loader = torch.utils.data.DataLoader(
        MMFLUDataset(
            src_inst=src,
            tgt_inst=tgt),
        num_workers=2,
        batch_size=batch_size,
        collate_fn=get_paired_collate_fn(tokenizer),
        shuffle=shuffle)

    return loader


def seq2label(seqs, tokenizer):
    pred = []
    for ids in seqs:
        x = tokenizer.decode(
            ids, skip_special_tokens=True,
            clean_up_tokenization_spaces=False)
        pred.append(x.strip('</s> '))
    return pred


def evaluate(model, loader, epoch, tokenizer):
    """
    Evaluation function
    """
    model.eval()
    pred, true = [], []
    with torch.no_grad():
        for i, batch in enumerate(loader):
            src, tgt = map(lambda x: x.to(device), batch)
            mask = src.ne(tokenizer.pad_token_id).long()
            outs = model.generate(
                input_ids=src,
                attention_mask=mask,
                num_beams=5,
                max_length=10)
            pred.extend(seq2label(outs, tokenizer))
            true.extend(seq2label(tgt, tokenizer))
    acc = sum([1 if i == j else 0 for i, j in zip(pred, true)]) / len(pred)
    model.train()

    print('[Info] {:02d}-valid: acc {:.4f}'.format(epoch, acc))

    return acc, confusion_matrix(pred, true).tolist(), classification_report(pred, true, output_dict=True)


def save_history(history, path):
    """
    Save the history to a CSV or JSON file based on file extension.
    """
    def _save_as_csv(history, path):
        with open(path, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=['subset', 'epoch', 'steps', 'loss', 'lr', 'sec', 'cm', 'clf_report', 'acc', 'prec', 'recall', 'f1'])
            writer.writeheader()
            for record in history:
                writer.writerow(record)

    def _save_as_json(history, path):
        with open(path, mode='w', encoding='utf-8') as file:
            json.dump(history, file, indent=4)

    path_dir = os.path.dirname(path)
    os.makedirs(path_dir, exist_ok=True)
    if path.endswith('.csv'):
        _save_as_csv(history, path)
    elif path.endswith('.json'):
        _save_as_json(history, path)
    else:
        raise ValueError("Unsupported file format. Please use .csv or .json")



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-seed', default=42, type=int, help='random seed')
    parser.add_argument(
        '-lang', nargs='+', help='language names', required=True)
    parser.add_argument(
        '-form', nargs='+', help='figure of speech', required=True)
    parser.add_argument(
        '-prompt', default='', type=str, help='prompt')
    parser.add_argument(
        '-batch_size', default=32, type=int, help='batch size')
    parser.add_argument(
        '-lr', default=1e-4, type=float, help='ini. learning rate')
    parser.add_argument(
        '-log_step', default=100, type=int, help='log every x step')
    parser.add_argument(
        '-epoch', default=80, type=int, help='force stop at x epoch')
    parser.add_argument(
        '-eval_step', default=1000, type=int, help='eval every x step')
    parser.add_argument(
        '-dataset_path', default='data', type=str, help='path to load training data from')
    parser.add_argument(
        '-history_path', default=None, type=str, help='path to save training history')
    parser.add_argument(
        '-ckpt_path', default='checkpoints', type=str, help='path to save trained checkpoints')
    parser.add_argument(
        '-base_model', default='google/mt5-base', type=str, help='base PLM for training')

    opt = parser.parse_args()
    launch(**dict(opt._get_kwargs()))

def launch(lang, form, seed=42, prompt='', batch_size=32, lr=1e-4, log_step=100, epoch=80, eval_step=1000, 
           dataset_path="data", history_path=None, ckpt_path='checkpoints', base_model='google/mt5-base'
        ):
    print('[Info]', locals())
    torch.manual_seed(seed)

    os.makedirs(ckpt_path, exist_ok=True)

    save_path_template = ckpt_path + '/{}_{}_{}.ckpt'
    save_path = save_path_template.format(
        base_model.replace('/', '_'),
        '_'.join(lang), '_'.join(form))

    model_name = base_model
    history_path = os.path.join(ckpt_path, history_path) if history_path is not None else save_path.replace('.ckpt', '.json')
    tokenizer = MT5TokenizerFast.from_pretrained(model_name)

    # read instances from input file
    train_src, train_tgt, valid_src, valid_tgt = [], [], [], []
    for lang_item in lang:
        for form_item in form:
            path = '{}/train_{}_{}.0'.format(dataset_path, lang_item, form_item)

            if not os.path.exists(path):
                continue

            train_0, train_1 = read_insts(
                'train', lang_item, form_item, prompt, tokenizer)
            valid_0, valid_1 = read_insts(
                'valid', lang_item, form_item, prompt, tokenizer)
            train_src.extend(train_0)
            train_tgt.extend(train_1)
            valid_src.extend(valid_0)
            valid_tgt.extend(valid_1)
            print('[Info] {} insts of train set in {}-{}'.format(
                len(train_0), lang_item, form_item))
            print('[Info] {} insts of valid set in {}-{}'.format(
                len(valid_0), lang_item, form_item))

    train_loader = MMFLUIterator(train_src, train_tgt, batch_size, tokenizer)
    valid_loader = MMFLUIterator(valid_src, valid_tgt, batch_size, tokenizer)

    
        
    model = MT5ForConditionalGeneration.from_pretrained(model_name)
    history = []
    start_schedule_from = 0
    if os.path.isfile(save_path):
        # Resume
        model.load_state_dict(torch.load(save_path))
        if os.path.isfile(history_path):
            with open(history_path) as f:
                history = json.load(f)
                start_schedule_from = max(e.get('steps', 0) for e in history)
    
    model = model.to(device).train()
    
    optimizer = torch.optim.Adam(
        filter(lambda x: x.requires_grad, model.parameters()),
        lr=lr, betas=(0.9, 0.98), eps=1e-09)

    scheduler = PolynomialLRDecay(
        optimizer,
        warmup_steps=1000,
        max_decay_steps=10000,
        end_learning_rate=5e-5,
        power=2)

    loss_list = []
    start = time.time()
    eval_acc, tab = 0, 0
    patience = 6
    for epoch_idx in range(epoch):
        pbar = tqdm.tqdm(train_loader, desc=f"Epoch {epoch_idx+1:02d}/{epoch}")
        for batch in pbar:
            if scheduler.steps < start_schedule_from:
                scheduler.step()
                continue
            src, tgt = map(lambda x: x.to(device), batch)
            optimizer.zero_grad()

            mask = src.ne(tokenizer.pad_token_id).long()
            loss = model(src, mask, labels=tgt)[0]
            loss.backward()
            scheduler.step()
            optimizer.step()
            loss_list.append(loss.item())

            if scheduler.steps % log_step == 0:
                lr_current = optimizer.param_groups[0]['lr']
                log_info = {
                    "subset": "train",
                    "epoch": epoch_idx,
                    "steps": scheduler.steps,
                    "loss": np.mean(loss_list),
                    "lr": lr_current,
                    "sec": time.time() - start
                }
                
                pbar.set_postfix(log_info)
                loss_list = []
                start = time.time()
                history.append(log_info)

                save_history(history, history_path)

            if ((len(train_loader) >= eval_step
                 and scheduler.steps % eval_step == 0)
                    or (len(train_loader) < eval_step
                        and scheduler.steps % len(train_loader) == 0
                        and scheduler.steps > 1000)
                    or scheduler.steps == 1000):
                valid_acc, valid_cm, valid_cr = evaluate(
                    model,
                    valid_loader,
                    epoch_idx,
                    tokenizer)
                
                log_info = {
                    "subset": "validation",
                    "epoch": epoch_idx,
                    "acc": valid_acc,
                    "cm": valid_cm,
                    "clf_report": valid_cr,
                    "lr": lr_current,
                    "sec": time.time() - start
                }
                
                start = time.time()
                history.append(log_info)
                save_history(history, history_path)

                if eval_acc < valid_acc:
                    eval_acc = valid_acc
                    torch.save(model.state_dict(), save_path)
                    pbar.set_postfix_str('The checkpoint has been updated.')
                    tab = 0
                else:
                    tab += 1
                    if tab == patience:
                        break

    # evaluation
    print('[Info] Evaluation')
    model.load_state_dict(torch.load(save_path))
    for form_item in form:
        for lang_item in lang:
            path = '{}/test_{}_{}.0'.format(dataset_path, lang_item, form_item)
            if not os.path.exists(path):
                continue
            test_0, test_1 = read_insts(
                'test', lang_item, form_item, prompt, tokenizer)
            test_loader = MMFLUIterator(test_0, test_1, batch_size, tokenizer)
            print('[Info] {} insts of {}-{}'.format(
                len(test_0), lang_item, form_item))
            test_acc, test_cm, test_cr = evaluate(
                model,
                test_loader,
                0,
                tokenizer)

            log_info = {
                "subset": "test",
                "epoch": epoch_idx,
                "acc": test_acc,
                "cm": test_cm,
                "clf_report": test_cr,
                "lr": lr_current,
                "sec": time.time() - start
            }
            
            start = time.time()
            history.append(log_info)
            save_history(history, history_path)

    # Save training history to CSV or JSON
    save_history(history, history_path)
