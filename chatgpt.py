import torch
import torch.utils.data
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import json
import pickle
import os
import argparse
import re

from sklearn.metrics import f1_score

import openai

def set_openai_key(key):
    openai.api_key = key

def proc_ques(ques):
    words = re.sub(r"([.,'!?\"()*#:;])",'',ques.lower()).replace('-', ' ').replace('/', ' ')
    return words

emotions = ['excitement', 'sadness', 'anger', 'contentment', 'something else', 'disgust', 'fear', 'amusement', 'awe']

def parse_option():
    parser = argparse.ArgumentParser('Visual Prompting for CLIP')

    parser.add_argument('--ckpt_path', type=str, default='finetune')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--freq', type=int, default=10)
    parser.add_argument('--visual_backbone', default=False, action="store_true")
    parser.add_argument('--load_from_epoch', type=int, default=0)
    parser.add_argument('--finetune', default=False, action="store_true")
    parser.add_argument('--eval', default=False, action="store_true")
    parser.add_argument('--answerer', default=False, action="store_true")
    parser.add_argument('--dialog', default=False, action="store_true")
    args = parser.parse_args()
    return args

class VQAXEvalDataset(Dataset):

    def __init__(self, path):
        self.max_seq_len = max_seq_len  # question + <bos> The answer is <answer> becase <explanation> <eos>
        with open(path, 'rb') as f:
            self.data = pickle.load(f)
        self.ids_list = range(len(self.data))

    def __getitem__(self, i):
        dialog_id = self.ids_list[i]
        sample = self.data[dialog_id]
        image = sample['img_src']
        batch_id = sample['batch_id']
        
        text_a = 'Give the below conversation talking about an artwork: '
        text_a += proc_ques(sample['caption1'])
        text_a += proc_ques(sample['caption2'])

        for ut in sample['conversation']:
            text_a += proc_ques(ut)

        text_a += proc_ques(' What is the emotion the artwork renders? Please choose one emotion only from [excitement\, sadness\, anger\, contentment\, something else\, disgust\, fear\, amusement and awe]. Give me just one-word answer')

        emotion = sample['emotion_before']

        exp = proc_ques(sample['explanation_before'])

        return image, batch_id, sample['dialog_id'], emotion, exp, text_a

    def __len__(self):
        return len(self.ids_list)


def sample_sequences(loader):
    saves = []
    gt_emotions = []
    pred_emotions = []
    openai_key = "your openai key"
    set_openai_key(openai_key)

    for i, batch in enumerate(loader):

        img_path, batch_id, dialog_id, emotion, exp, gpt3_prompt = batch

        emotion = emotion[0] if type(emotion) else emotion
        exp = exp[0] if type(exp) else exp

        messages = [{"role": "system", "content": gpt3_prompt[0]}]

        response = openai.ChatCompletion.create(model='gpt-3.5-turbo', messages=messages, temperature=0.6, max_tokens=10)
        answer = response['choices'][0]['message']['content'].strip().split('.')[0].lower()

        pred_emotion = list(set(answer.split(' ')).intersection(set(emotions)))
        if len(pred_emotion) == 0:
            continue
        else:
            pred_emotion = pred_emotion[0]

        print(pred_emotion, emotion, flush=True)

        messages.append({'role': 'assistant', 'content': answer})
        messages.append({'role': 'user', 'content': ' Please give me the reason why you chose this answer'})

        response = openai.ChatCompletion.create(model='gpt-3.5-turbo', messages=messages, temperature=0.6, max_tokens=40)
        explanation = response['choices'][0]['message']['content']

        print(explanation, flush=True)

        whole = {'gpt3_prompt': gpt3_prompt[0], 'explanation': explanation, 'gt_explanation':exp, 'pred_emotion': pred_emotion, 'gt_emotion':emotion, 'img_url': img_path, 'dialog_id': dialog_id, 'batch_id': batch_id.item()}
        
        if pred_emotion in emotions:
            pred_emotions.append(emotions.index(pred_emotion))
            gt_emotions.append(emotions.index(emotion))

        print(f1_score(gt_emotions, pred_emotions, average='weighted'),flush=True)
        saves.append(whole)
    with open('chat.json','w') as f:
        json.dump(saves, f)
        

args = parse_option()

finetune_pretrained = args.finetune  # if True, finetunes from the image captioning model
eval_batch_size = 1
img_size = 224
ckpt_path = args.ckpt_path
nle_data_train_path = 'train_data.pickle'
nle_data_eval_path = 'val_data.pickle'
nle_data_test_path = 'test_data.pkl'
max_seq_len = 400
no_sample = True
top_k = 0
top_p = 0.9
batch_size = args.batch_size  # per GPU
num_train_epochs = 1000
weight_decay = 0
learning_rate = 2e-5 if not finetune_pretrained else 1e-5
gradient_accumulation_steps = 1
start_epoch = 0
temperature = 1

img_transform = transforms.Compose([transforms.Resize((img_size, img_size)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])


test_dataset = VQAXEvalDataset(path=nle_data_test_path)

test_loader = torch.utils.data.DataLoader(test_dataset,
                                        batch_size=1,
                                        shuffle=False,
                                        pin_memory=True)
sample_sequences(test_loader)

