import torch
import torch.nn.functional as F
import os
from pathlib import Path
import pandas as pd 
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, BertConfig, RobertaTokenizer, RobertaForSequenceClassification
from transformers import get_linear_schedule_with_warmup
import numpy as np
import time
import datetime
import random
from collections import defaultdict
import pickle
from torch.utils.data import Dataset
from PIL import Image
import re
import torchvision.transforms as transforms

emotions = ['excitement', 'sadness', 'anger', 'contentment', 'something else', 'disgust', 'fear', 'amusement', 'awe']

def proc_ques(ques):
    words = re.sub(r"([.,'!?\"()*#:;])",'',ques.lower()).replace('-', ' ').replace('/', ' ')
    return words

def flat_accuracy(preds, labels):
    preds = (1 / (1 + np.exp(-preds)))
    pred_flat = np.round(preds).astype(int)
    labels_flat = labels
    true_labels = np.where(labels_flat > 0)
    pos_recall = np.sum((pred_flat == labels_flat)[true_labels]) / len(true_labels[0])
    weighted_recall = np.sum(pred_flat == labels_flat) / (labels_flat.shape[0] * labels_flat.shape[1])
    return weighted_recall, pos_recall

def format_time(elapsed):
    elapsed_rounded = int(round((elapsed)))
    return str(datetime.timedelta(seconds=elapsed_rounded))

import json
import argparse
from urllib.parse import unquote

parser = argparse.ArgumentParser()
# parser.add_argument("-e", "--emotions", required=True, help="Emotions Dict !!!")
parser.add_argument("-m", "--model", default='roberta', help="pretrained model name")
parser.add_argument("-s", "--model_size", default='base', help="pretrained model size")
parser.add_argument('--ckpt_path', type=str, default='finetune')
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--freq', type=int, default=10)
parser.add_argument('--visual_backbone', default=False, action="store_true")
parser.add_argument('--load_from_epoch', type=int, default=0)
parser.add_argument('--finetune', default=False, action="store_true")
parser.add_argument('--eval', default=False, action="store_true")
parser.add_argument('--answerer', default=False, action="store_true")
parser.add_argument('--dialog', default=True, action="store_true")
parser.add_argument('--checkpoint', default='/ibex/ai/home/shenx/visdial/vad_bert/checkpoints/trained_roberta_2023_02_01_02_10_14')
# parser.add_argument('--balance', action='store_true')
args = parser.parse_args()
# pdb.set_trace()
# jsonString = unquote(args.emotions)
# EMOTION_ID = json.loads(jsonString)

# EMOTION_ID = '{"amusement":0,"awe":1,"contentment":2,"excitement":3,"anger":4,"disgust":5,"fear":6,"sadness":7,"something else":8}'
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

ARTEMIS_EMOTIONS = ['amusement', 'awe', 'contentment', 'excitement',
                    'anger', 'disgust',  'fear', 'sadness', 'something else']

emotion_pairs ={
    'amusement': 'sadness',
    'awe': 'disgust',
    'contentment': 'disgust',
    'excitement': 'fear',
    'anger': 'contentment',
    'disgust': 'awe',
    'fear': 'excitement',
    'sadness': 'amusement'
}

EMOTION_ID = {e: i for i, e in enumerate(ARTEMIS_EMOTIONS)}


ID_EMOTION = {EMOTION_ID[e]: e for e in EMOTION_ID}

num_classes = len(np.unique(list(EMOTION_ID.values())))
print(f'Number of unique emotion categories: {num_classes}',flush=True)

print('Dataset Loaded ......',flush=True)

# pdb.set_trace()
if args.model == 'bert':
    BERT_version = 'bert-'+args.model_size+'-cased'
    Pretrained_tokenizer = BertTokenizer
    Pretrained_model = BertForSequenceClassification
elif args.model == 'roberta':
    BERT_version = 'roberta-'+args.model_size
    Pretrained_tokenizer = RobertaTokenizer
    Pretrained_model = RobertaForSequenceClassification
else:
    raise ValueError(f'model {args.model} is not implemented')

# BERT_version = 'bert-large-cased'
print('Start Tokenizing ......',flush=True)
tokenizer = Pretrained_tokenizer.from_pretrained(BERT_version, padding_side='right', model_max_length=512)
MAX_LEN = 256

print('Loading Model ......',flush=True)
if args.checkpoint is None:
    model = Pretrained_model.from_pretrained(
        BERT_version,
        num_labels = num_classes, 
        output_attentions = False,
        output_hidden_states = False,
    )
else:
    model = Pretrained_model.from_pretrained(
        args.checkpoint,
        num_labels = num_classes, 
        output_attentions = False,
        output_hidden_states = False,
    )
model.to(device)

seed_val = 42
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

model.eval()

with open('/ibex/ai/home/shenx/visdial/vad_bert/goemotion_scripts/test_with_options.pickle', 'rb') as f:
    data = pickle.load(f)

infos = []

for sample in data:
    text_a = ''
    text_a += proc_ques(sample['caption1'])
    text_a += proc_ques(sample['caption2'])

    for pairs in sample['dialog']:
        text_a += proc_ques(pairs['question'])
        text_a += proc_ques(pairs['answer'])
    text_a += proc_ques('what is the emotion')

    emotion = word_check(sample['emotion_before'])

    tokens = tokenizer.tokenize(text_a)
    seq_len = len(tokens)
    padding_len = 512 - seq_len
    tokens = tokens + ([tokenizer.pad_token] * padding_len)
    if len(tokens) > 512:
        tokens = tokens[:512]
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_ids = torch.tensor(input_ids, dtype=torch.long)

    input_ids = input_ids.to(device).unsqueeze(0)

    with torch.no_grad():        
        outputs = model(input_ids, token_type_ids=None)

    logits = outputs[0]
    probs = F.softmax(logits, dim=-1).squeeze()

    target_emotion = emotion_pairs[emotion]
    target = ARTEMIS_EMOTIONS.index(target_emotion)
    ori_prob = probs[target]

    max_indices = np.argmax(probs.cpu())
    pred_emotion = ARTEMIS_EMOTIONS[max_indices]

    if emotion != pred_emotion:
        continue

    print(emotion, pred_emotion, flush=True)

    history = dict()
    probs_dict = defaultdict(list)

    current_emotion = pred_emotion

    for turn in range(1, 10):
        max_prob = 0.0
        for change_id in range(len(sample['dialog'][turn]['answer_options'])):
            text_a = ''
            text_a += proc_ques(sample['caption1'])
            text_a += proc_ques(sample['caption2'])
            for idx, pairs in enumerate(sample['dialog']):
                text_a += proc_ques(pairs['question'])
                if idx ==0:
                    text_a += proc_ques(pairs['answer'])
                else:
                    if idx < turn:
                        text_a += proc_ques(sample['dialog'][idx]['answer_options'][history[idx]])
                    elif idx == turn:
                        text_a += proc_ques(sample['dialog'][idx]['answer_options'][change_id])
                    else:
                        text_a += proc_ques(sample['dialog'][idx]['answer'])
            text_a += proc_ques('what is the emotion')
            tokens = tokenizer.tokenize(text_a)
            seq_len = len(tokens)
            padding_len = 512 - seq_len
            tokens = tokens + ([tokenizer.pad_token] * padding_len)
            if len(tokens) > 512:
                tokens = tokens[:512]
            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            input_ids = torch.tensor(input_ids, dtype=torch.long)

            input_ids = input_ids.to(device).unsqueeze(0)

            with torch.no_grad():        
                outputs = model(input_ids, token_type_ids=None)

            logits = outputs[0]
            probs = F.softmax(logits, dim=-1).squeeze()
            max_indices = np.argmax(probs.cpu())
            pred_emotion = ARTEMIS_EMOTIONS[max_indices]

            if max_prob < probs[target].item():
                current_emotion = pred_emotion

            probs_dict[turn].append(probs[target].cpu())

        history[turn] = np.argmax(probs_dict[turn])
    if current_emotion == target_emotion:
        ori_dialog = []
        changed_dialog = []
        prob_change = []
        for idx, pairs in enumerate(sample['dialog']):
            ori_dialog.append(pairs['question'])
            ori_dialog.append(pairs['answer'])
            changed_dialog.append(pairs['question'])
            if idx == 0:
                changed_dialog.append(pairs['answer'])
                prob_change.append(ori_prob.item())
            else:
                changed_dialog.append(sample['dialog'][idx]['answer_options'][history[idx]])
                prob_change.append(np.float64(np.max(probs_dict[idx])))
        info = {
            'image_id': sample['image_id'],
            'ori_dial': ori_dialog,
            'new_dial': changed_dialog,
            'prob': prob_change,
            'ori_emotion': emotion,
            'target_emotion': target_emotion,
        }
        infos.append(info)

    with open('emotion_change.json', 'w') as f:
        json.dump(infos, f)
    