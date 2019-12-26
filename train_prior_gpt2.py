#!/usr/bin/env python

import transformers
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pickle
from rdkit import Chem
from rdkit import rdBase

import os
import json
import random
import numpy as np
import argparse
# from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from tqdm import tqdm
from torch.nn import DataParallel

from data_structs import MolData, Vocabulary
from utils import Variable, decrease_learning_rate
rdBase.DisableLog('rdApp.error')

from tqdm import trange
from transformers import GPT2LMHeadModel

parser = argparse.ArgumentParser()
parser.add_argument('--device', default='1', type=str, required=False, help='设置使用哪些显卡')
parser.add_argument('--model_config', default='./model_config_small.json', type=str, required=False,
                    help='选择模型参数')
parser.add_argument('--epochs', default=50, type=int, required=False, help='训练循环')
parser.add_argument('--batch_size', default=128, type=int, required=False, help='训练batch size')
parser.add_argument('--lr', default=5e-4, type=float, required=False, help='学习率')
parser.add_argument('--warmup_steps', default=2000, type=int, required=False, help='warm up步数')
parser.add_argument('--log_step', default=500, type=int, required=False, help='多少步汇报一次loss，设置为gradient accumulation的整数倍')
parser.add_argument('--stride', default=768, type=int, required=False, help='训练时取训练数据的窗口步长')
parser.add_argument('--gradient_accumulation', default=1, type=int, required=False, help='梯度积累')
parser.add_argument('--max_grad_norm', default=1.0, type=float, required=False)
parser.add_argument('--num_pieces', default=100, type=int, required=False, help='将训练语料分成多少份')
parser.add_argument('--min_length', default=128, type=int, required=False, help='最短收录文章长度')
parser.add_argument('--output_dir', default='saved_model/', type=str, required=False, help='模型输出路径')
parser.add_argument('--pretrained_model', default='', type=str, required=False, help='模型训练起点路径')

args = parser.parse_args()
print('args:\n' + args.__repr__())

os.environ["CUDA_VISIBLE_DEVICES"] = args.device  # 此处设置程序使用哪些显卡

model_config = transformers.modeling_gpt2.GPT2Config.from_json_file(args.model_config)
print('config:\n' + model_config.to_json_string())

n_ctx = model_config.n_ctx

epochs = args.epochs
batch_size = args.batch_size
lr = args.lr
warmup_steps = args.warmup_steps
log_step = args.log_step
stride = args.stride
gradient_accumulation = args.gradient_accumulation
max_grad_norm = args.max_grad_norm
num_pieces = args.num_pieces
min_length = args.min_length
output_dir = args.output_dir
pretrained_model = args.pretrained_model
assert log_step % gradient_accumulation == 0

if not os.path.exists(output_dir):
    os.mkdir(output_dir)

# Read vocabulary from a file
voc = Vocabulary(init_from_file="data/Voc")

# Create a Dataset from a SMILES file
moldata = MolData("data/mols_filtered.smi", voc)
data = DataLoader(moldata, batch_size=batch_size, shuffle=True, drop_last=True,
                  collate_fn=MolData.collate_fn)


def NLLLoss(inputs, targets, device = 'cpu'):
    """
        Custom Negative Log Likelihood loss that returns loss per example,
        rather than for the entire batch.

        Args:
            inputs : (batch_size, num_classes) *Log probabilities of each class*
            targets: (batch_size) *Target class index*

        Outputs:
            loss : (batch_size) *Loss for each example*
    """

    target_expanded = torch.zeros(inputs.size()).to(device)

    target_expanded.scatter_(1, targets.contiguous().view(-1, 1).data, 1.0)
    loss = target_expanded * inputs
    loss = torch.sum(loss, 1)
    return loss

def top_k_top_p_filtering(logits, top_k=0, top_p=1.0, filter_value=-float("Inf"), min_tokens_to_keep=1):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (batch size, vocabulary size)
            if top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            if top_p < 1.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
            Make sure we keep at least min_tokens_to_keep per batch example in the output
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    if top_k > 0:
        top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))  # Safety check
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold (token with 0 are kept)
        sorted_indices_to_remove = cumulative_probs > top_p
        if min_tokens_to_keep > 1:
            # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
            sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(
            dim=1, index=sorted_indices, source=sorted_indices_to_remove
        )
        logits[indices_to_remove] = filter_value
    return logits



def fast_sample_sequence(model, context, length, temperature=1.0, top_k=30, top_p=0.0, device='cuda'):
    sequences = [] # torch.Tensor([]).long().to(device)
    inputs = torch.LongTensor(context).view(batch_size, -1).to(device)
    # sequences.append(inputs)
    if inputs.shape[1] > 1:
        _, past = model(inputs[:, :-1], None)[:2]
        prev = inputs[:, -1].view(1, -1)
    else:
        past = None
        prev = inputs
    log_probs = torch.zeros(batch_size).to(device)
    finished = torch.zeros(batch_size).byte().to(device)
    entropy = torch.zeros(batch_size).to(device)
    with torch.no_grad():
        for i in trange(length):
            outputs = model(prev, past=past)
            next_token_logits = outputs[0][:,-1,:]
            past = outputs[1]
            next_token_logits = next_token_logits / temperature
            filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
            prob = torch.softmax(filtered_logits, dim=-1)
            log_prob = torch.log_softmax(filtered_logits, dim=-1)
            next_token = torch.multinomial(prob, num_samples=1)
            log_probs +=  NLLLoss(log_prob, next_token, device = device)
            entropy += -torch.sum((log_prob * prob), 1)
            prev = next_token.view(-1, 1)

            sequences.append(prev)

            x = next_token.data
            EOS_sampled = (x == voc.vocab['EOS']).data
            finished = torch.ge(finished + EOS_sampled, 1)
            if torch.prod(finished) == 1: break
        # print(len(sequences),sequences[-1].shape)
        sequences = torch.cat(sequences, -1)
        # return sequences.data, log_probs, entropy

    return sequences, log_probs, entropy

def pretrain(restore_from=None):
    """Trains the Prior RNN"""

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('using device:', device)

    if not args.pretrained_model:
        model = transformers.modeling_gpt2.GPT2LMHeadModel(config=model_config)
    else:
        model = transformers.modeling_gpt2.GPT2LMHeadModel.from_pretrained(args.pretrained_model)
    model.train()

    model.to(device)

    num_parameters = 0
    parameters = model.parameters()
    for parameter in parameters:
        num_parameters += parameter.numel()
    print('number of parameters: {}'.format(num_parameters))

    multi_gpu = False
    optimizer = transformers.AdamW(model.parameters(), lr=1.5e-4, correct_bias=True)
    scheduler = transformers.get_linear_schedule_with_warmup(optimizer, num_warmup_steps=2000,
                                                          num_training_steps=100000)
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = DataParallel(model, device_ids=[int(i) for i in args.device.split(',')])
        multi_gpu = True
    print('starting training')
    overall_step = 0
    running_loss = 0

    for epoch in range(epochs):
        print('epoch {}'.format(epoch + 1))
        now = datetime.now()
        print('time: {}'.format(now))

        for step, batch in tqdm(enumerate(data), total=len(data)):

            batch_inputs = batch.long().to(device)

            #  forward pass
            outputs = model.forward(input_ids=batch_inputs, labels=batch_inputs)
            loss, logits = outputs[:2]

            #  get loss
            if multi_gpu:
                loss = loss.mean()
            if gradient_accumulation > 1:
                loss = loss / gradient_accumulation

            #  loss backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            #  optimizer step
            if (overall_step + 1) % gradient_accumulation == 0:
                running_loss += loss.item()
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

            # sampling small molecules
            if (overall_step + 1) % log_step*2 == 0:
                context = torch.zeros(batch_size).long()
                context[:] = voc.vocab['GO']
                sequences, log_probs, entropy = fast_sample_sequence(model, context, n_ctx)
                valid = 0
                for i, seq in enumerate(sequences.cpu().numpy()):
                    smiles = voc.decode(seq)
                    if Chem.MolFromSmiles(smiles):
                        valid += 1
                    if i < 5:
                        tqdm.write(smiles)

                tqdm.write("\n{:>4.1f}% valid SMILES".format(100 * valid / len(sequences)))
                tqdm.write("*" * 50 + "\n")


            if (overall_step + 1) % log_step == 0:

                # tb_writer.add_scalar('loss', loss.item() * gradient_accumulation, overall_step)
                print('now time: {}:{}. Step {} of epoch {}, loss {}'.format(
                    datetime.now().hour,
                    datetime.now().minute,
                    step + 1,
                    epoch + 1,
                    running_loss * gradient_accumulation / (log_step / gradient_accumulation)))
                running_loss = 0
            overall_step += 1


        print('saving model for epoch {}'.format(epoch + 1))
        if not os.path.exists(output_dir + 'model_epoch{}'.format(epoch + 1)):
            os.mkdir(output_dir + 'model_epoch{}'.format(epoch + 1))
        model_to_save = model.module if hasattr(model, 'module') else model
        model_to_save.save_pretrained(output_dir + 'model_epoch{}'.format(epoch + 1))
        # torch.save(scheduler.state_dict(), output_dir + 'model_epoch{}/scheduler.pt'.format(epoch + 1))
        # torch.save(optimizer.state_dict(), output_dir + 'model_epoch{}/optimizer.pt'.format(epoch + 1))
        print('epoch {} finished'.format(epoch + 1))

        then = datetime.now()
        print('time: {}'.format(then))
        print('time for one epoch: {}'.format(then - now))

    print('training finished')
    if not os.path.exists(output_dir + 'final_model'):
        os.mkdir(output_dir + 'final_model')
    model_to_save = model.module if hasattr(model, 'module') else model
    model_to_save.save_pretrained(output_dir + 'final_model')
    # torch.save(scheduler.state_dict(), output_dir + 'final_model/scheduler.pt')
    # torch.save(optimizer.state_dict(), output_dir + 'final_model/optimizer.pt')

if __name__ == "__main__":
    pretrain()
