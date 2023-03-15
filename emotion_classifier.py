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
parser.add_argument('--checkpoint', default=None)
# parser.add_argument('--balance', action='store_true')
args = parser.parse_args()
# pdb.set_trace()
# jsonString = unquote(args.emotions)
# EMOTION_ID = json.loads(jsonString)

# EMOTION_ID = '{"amusement":0,"awe":1,"contentment":2,"excitement":3,"anger":4,"disgust":5,"fear":6,"sadness":7,"something else":8}'
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

ARTEMIS_EMOTIONS = ['amusement', 'awe', 'contentment', 'excitement',
                    'anger', 'disgust',  'fear', 'sadness', 'something else']

EMOTION_ID = {e: i for i, e in enumerate(ARTEMIS_EMOTIONS)}


ID_EMOTION = {EMOTION_ID[e]: e for e in EMOTION_ID}

num_classes = len(np.unique(list(EMOTION_ID.values())))
print(f'Number of unique emotion categories: {num_classes}',flush=True)

# # df_train = pd.read_csv('data/train.tsv', sep='\t', header=None, names=['utterance', 'emotion', 'id'])
# df_train = pd.read_csv(args.train_set)
# df_train['emotion'] = df_train['emotion_label']
# # df_train['emotion'] = df_train['emotion'].apply(lambda x: x.split(',')).apply(lambda x: [int(i) for i in x])
# sentences_train = df_train['utterance']
# labels_train = df_train['emotion'].values
# labels_pt_train = torch.zeros((labels_train.shape[0], num_classes))
# for i, emo in enumerate(labels_train):
#     labels_pt_train[i, emo] = 1

# df_val = pd.read_csv(args.test_set)
# df_val['emotion'] = df_val['emotion_label']
# sentences_val = df_val['utterance']
# labels_val = df_val['emotion'].values
# labels_pt_val = torch.zeros((labels_val.shape[0], num_classes))
# for i, emo in enumerate(labels_val):
#     labels_pt_val[i, emo] = 1

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

# if not Path(f'tokenized_dataset_go_train_{args.model}_{args.model_size}.pkl').exists():
#     print('tokenized_dataset_go_train.pkl does not exist\nCreating from raw text ......... ')
# tokenized_data_train = tokenizer(sentences_train.to_list(), add_special_tokens=True, max_length=MAX_LEN,
#                             truncation=True, padding='max_length', return_tensors='pt', return_attention_mask=True)
#     with open(f'tokenized_dataset_go_train_{args.model}_{args.model_size}.pkl', 'wb') as f:
#         pickle.dump(tokenized_data_train, f)
# else:
#     with open(f'tokenized_dataset_go_train_{args.model}_{args.model_size}.pkl', 'rb') as f:
#         tokenized_data_train = pickle.load(f)

# if not Path(f'tokenized_dataset_go_val_{args.model}_{args.model_size}.pkl').exists():
#     print('tokenized_dataset_go_val.pkl does not exist\nCreating from raw text ......... ')
# tokenized_data_val = tokenizer(sentences_val.to_list(), add_special_tokens = True, max_length = MAX_LEN, 
#                         truncation=True, padding='max_length', return_tensors='pt', return_attention_mask=True)
#     with open(f'tokenized_dataset_go_val_{args.model}_{args.model_size}.pkl', 'wb') as f:
#         pickle.dump(tokenized_data_train, f)
# else:
#     with open(f'tokenized_dataset_go_val_{args.model}_{args.model_size}.pkl', 'rb') as f:
#         tokenized_data_val = pickle.load(f)
print('Finished Tokenizing ......',flush=True)

# train_inputs, train_masks, train_labels = tokenized_data_train['input_ids'], tokenized_data_train['attention_mask'], labels_pt_train
# validation_inputs, validation_masks, validation_labels = tokenized_data_val['input_ids'], tokenized_data_val['attention_mask'], labels_pt_val

# pdb.set_trace()

batch_size = 32
print('Building Dataloaders ......',flush=True)
img_transform = transforms.Compose([transforms.Resize((224, 224)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
# train_dataset = TensorDataset(train_inputs, train_masks, train_labels)
# train_sampler = RandomSampler(train_dataset)
# train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=batch_size)

# validation_dataset = TensorDataset(validation_inputs, validation_masks, validation_labels)
# validation_sampler = SequentialSampler(validation_dataset)
# validation_dataloader = DataLoader(validation_dataset, sampler=validation_sampler, batch_size=batch_size)
class VQAXTrainDataset(Dataset):

    def __init__(self, path, transform, tokenizer, max_seq_len):

        self.tokenizer = tokenizer
        self.transform = transform
        self.max_seq_len = max_seq_len  # question + <bos> The answer is <answer> becase <explanation> <eos>
        with open(path, 'rb') as f:
            self.data = pickle.load(f)
        self.data = self.data
        self.ids_list = range(len(self.data))

    def __getitem__(self, i):
        dialog_id = self.ids_list[i]
        sample = self.data[dialog_id]
        image = sample['img_src']

        text_a = ''
        text_a += proc_ques(sample['caption1'])
        text_a += proc_ques(sample['caption2'])
        if args.dialog:
            for ut in sample['conversation']:
                text_a += proc_ques(ut)
        text_a += proc_ques('what is the emotion')

        answer = sample['emotion_before']

        emotion_id = ARTEMIS_EMOTIONS.index(answer)
        emotion = torch.zeros(9)
        emotion[emotion_id] = 1.0
        
        tokens = self.tokenizer.tokenize(text_a)
        seq_len = len(tokens)
        padding_len = self.max_seq_len - seq_len
        tokens = tokens + ([self.tokenizer.pad_token] * padding_len)
        if len(tokens) > self.max_seq_len:
            tokens = tokens[:self.max_seq_len]
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        input_ids = torch.tensor(input_ids, dtype=torch.long)

        return dialog_id, input_ids, emotion

    def __len__(self):
        return len(self.ids_list)

class VQAXEvalDataset(Dataset):

    def __init__(self, path, transform, tokenizer, max_seq_len):
        self.tokenizer = tokenizer
        self.transform = transform
        self.max_seq_len = max_seq_len  # question + <bos> The answer is <answer> becase <explanation> <eos>
        with open(path, 'rb') as f:
            self.data = pickle.load(f)
        self.ids_list = range(len(self.data))

    def __getitem__(self, i):
        dialog_id = self.ids_list[i]
        sample = self.data[dialog_id]
        image = sample['img_src']
        text_a = ''
        text_a += proc_ques(sample['caption1'])
        text_a += proc_ques(sample['caption2'])
        
        answer = sample['emotion_before']
        
        emotion_id = ARTEMIS_EMOTIONS.index(answer)
        emotion = torch.zeros(9)
        emotion[emotion_id] = 1.0

        tokens = self.tokenizer.tokenize(text_a)
        seq_len = len(tokens)
        padding_len = self.max_seq_len - seq_len
        tokens = tokens + ([self.tokenizer.pad_token] * padding_len)
        if len(tokens) > self.max_seq_len:
            tokens = tokens[:self.max_seq_len]

        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        input_ids = torch.tensor(input_ids, dtype=torch.long)

        return dialog_id, input_ids, emotion

    def __len__(self):
        return len(self.ids_list)

train_dataset = VQAXTrainDataset(path='train_data.pickle',
                                 transform=img_transform,
                                 tokenizer=tokenizer,
                                 max_seq_len=512)

train_loader = torch.utils.data.DataLoader(train_dataset,
                                            batch_size=batch_size,
                                            shuffle=True,
                                            pin_memory=True)

eval_dataset = VQAXEvalDataset(path = 'val_data.pickle',
                                transform=img_transform,
                                tokenizer=tokenizer,
                                max_seq_len=512)

val_loader = torch.utils.data.DataLoader(eval_dataset,
                                            batch_size=128,
                                            shuffle=False,
                                            pin_memory=True)
print('Done ......',flush=True)

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
print('Done ......',flush=True)

lr = 2e-5
optimizer = AdamW(model.parameters(),
                  lr = lr,
                  eps = 1e-8 
                )

print(f'Adam learning rate: {lr}',flush=True)

epochs = 10
total_steps = len(train_loader) * epochs
scheduler = get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps = 0,
                                            num_training_steps = total_steps)

seed_val = 42
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

# train_metrics = ClassificationMetrics(num_classes)
# train_confusion = ConfusionMatrix(num_classes, [ID_EMOTION[idx][0] for idx in range(num_classes)])
# val_metrics = ClassificationMetrics(num_classes)
# val_confusion = ConfusionMatrix(num_classes, [ID_EMOTION[idx][0] for idx in range(num_classes)])

loss_values = []
for epoch_i in range(0, epochs):
    
    # ========================================
    #               Training
    # ========================================
    
    # Perform one full pass over the training set.
    print("",flush=True)
    print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs),flush=True)
    print('Training...',flush=True)
    # Measure how long the training epoch takes.
    t0 = time.time()
    total_loss = []
    total_acc = []
    pos_recall = []
    model.train()
    # train_metrics.reset()
    # train_confusion.reset()
    for step, batch in enumerate(train_loader):

        img_id, b_input_ids, emotion = batch
        b_input_ids = b_input_ids.to(device)
        b_labels = emotion.to(device)
      
        optimizer.zero_grad()  
        outputs = model(b_input_ids, 
                    token_type_ids=None)
        
        # pdb.set_trace()

        logits = outputs[0]
        loss = F.binary_cross_entropy_with_logits(logits, b_labels)

        total_loss.append(loss.item())

        # pdb.set_trace()
        # train_metrics.update(logits.argmax(dim=1), b_labels)  
        # train_confusion.update(logits.argmax(dim=1), b_labels)  
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.cpu().numpy()
        train_acc = flat_accuracy(logits, label_ids)
        total_acc.append(train_acc[0])
        pos_recall.append(train_acc[1])

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()


        if step % 100 == 0 and not step == 0:
        # if step % int(len(train_dataloader)/200) == 0 and not step == 0:
            elapsed = format_time(time.time() - t0)
            print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.     Train_Loss: {:.5f} Train_acc: {:.3f} Emotion_recall: {:.3f}'.format(step, len(train_loader), elapsed, np.mean(total_loss), np.mean(total_acc), np.mean(pos_recall)),flush=True)
        # if step % 2 == 0 and not step == 0:
        #     break
    
    avg_train_loss = np.mean(total_loss)  
    avg_train_acc = np.mean(total_acc)    
    avg_emo_recall = np.mean(pos_recall)            

    loss_values.append(avg_train_loss)
    print("",flush=True)
    print("  Average training loss: {0:.2f}".format(avg_train_loss),flush=True)
    print("  Average training accuracy: {0:.2f}".format(avg_train_acc),flush=True)
    print("  Average training emotion recall: {0:.2f}".format(avg_emo_recall),flush=True)
    print("  Training epoch took: {:}".format(format_time(time.time() - t0)),flush=True)
    # print(train_metrics)
    # print(train_confusion)
        
    # ========================================
    #               Validation
    # ========================================
    print("",flush=True)
    print("Running Validation...",flush=True)
    t0 = time.time()
    model.eval()

    current_time = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    model.save_pretrained(f'trained_roberta_{epoch_i}_{current_time}')
    
    eval_loss, eval_accuracy, eval_recall = 0, 0, 0
    nb_eval_steps = 0

    # val_metrics.reset()
    # val_confusion.reset()
    for batch in val_loader:

        img_id, b_input_ids, emotion = batch
        b_input_ids = b_input_ids.to(device)
        b_labels = emotion.to(device)
        
        with torch.no_grad():        
            outputs = model(b_input_ids, 
                            token_type_ids=None)
        
        logits = outputs[0]
        # val_metrics.update(logits.argmax(dim=1), b_labels.to(device))
        # val_confusion.update(logits.argmax(dim=1), b_labels.to(device))
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.cpu().numpy()
        
        # Calculate the accuracy for this batch of test sentences.
        tmp_eval_accuracy = flat_accuracy(logits, label_ids)
        
        # Accumulate the total accuracy.
        eval_accuracy += tmp_eval_accuracy[0]
        eval_recall += tmp_eval_accuracy[1]
        # Track the number of batches
        nb_eval_steps += 1
    # Report the final accuracy for this validation run.
    print("  Accuracy: {0:.2f}".format(eval_accuracy/nb_eval_steps),flush=True)
    print("  Emotion recall: {0:.2f}".format(eval_recall/nb_eval_steps),flush=True)
    print("  Validation took: {:}".format(format_time(time.time() - t0)),flush=True)
    # print(val_metrics)
    # print(val_confusion)
print("Training complete!",flush=True)

current_time = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
model.save_pretrained(f'trained_roberta_{current_time}')

# with open(f'/ibex/scratch/mohameys/text_to_emotions/go_models/{out_file}_trained_bert_{current_time}.pkl', 'wb') as f:
#     pickle.dump(model, f)

