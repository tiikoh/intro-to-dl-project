# -*- coding: utf-8 -*-
"""
   Introduction to Deep Learning (LDA-T3114)
   Final Project
"""

#--------------------------------------------------------
# SIGMORPHON 2016 Shared Task: Morphological Reinflection
#
# Task 1 – Inflection
# Given a lemma (the dictionary form of a word) with its part-of-speech, generate a target inflected form.
#
# Task 2 – Reinflection
# Given an inflected form and its current tag, generate a target inflected form.
#
# Task 3 – Unlabeled Reinflection
# Given an inflected form without its current inflection, generate a target inflected form.
#------------------------------------------------------------------------------------------

import os
import sys
import inspect

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from random import choice, random, shuffle
from torch.nn.utils.rnn import pad_sequence


#--- hyperparameters ---
N_EPOCHS = 20
LEARNING_RATE = 0.001
REPORT_EVERY = 5
EMBEDDING_DIM = 200
HIDDEN_DIM = 200
BATCH_SIZE = 20
MAX_LEN = 30
N_LAYERS = 1
DROP_P = 0.1

START_SYMBOL = '<w>'
END_SYMBOL = '</w>'
UNK = '<unk>'
PAD = '<pad>'

# SPECIFY HERE WHICH TASK TO RUN ('task1', 'task2', 'task3').
TASK = 'task1'

# Use CUDA if available.
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')



#--- data ---
def read_lines(file):
    data = []
    with open(file, encoding='utf-8') as f:
        for line in f:
            line = line.strip('\n')
            if 'task2' in os.path.basename(file): # task 2 includes source tags
                source_tag, source_form, target_tag, target_form = line.split('\t')
                data.append({'SOURCE_TAG':source_tag,
                            'SOURCE_FORM':source_form,
                            'TARGET_TAG':target_tag,
                            'TARGET_FORM':target_form})
            else:
                source_form, target_tag, target_form = line.split('\t')
                data.append({'SOURCE_FORM':source_form,
                            'TARGET_TAG':target_tag,
                            'TARGET_FORM':target_form})
    return data


def read_datasets(task, language, data_dir):
    datasets = {'train': read_lines(os.path.join(data_dir, '%s-%s-%s' %
                                                (language, task, 'train.txt'))),
                'dev': read_lines(os.path.join(data_dir, '%s-%s-%s' %
                                                (language, task, 'dev.txt'))),
                'test': read_lines(os.path.join(data_dir, '%s-%s-%s' %
                                                (language, task, 'test.txt')))}
    return datasets



#--- auxiliary functions ---
def token_to_idx(dataset):
    # Indices for individual characters.
    characters = {c:i+2 for i,c in enumerate({c for lan in dataset
                                                for ex in dataset[lan]['train']
                                                for c in ex['TARGET_FORM']})}
    # Add START_SYMBOL and END_SYMBOL.
    characters[START_SYMBOL] = 0
    characters[END_SYMBOL] = 1

    # Indices for grammatical tags.
    tags = {c:i+len(characters) for i,c in enumerate({c for lan in dataset
                                                        for ex in dataset[lan]['train']
                                                        for c in ex['TARGET_TAG'].split(',')})}
    # Combine characters and tags.
    idx_map = {**characters, **tags}

    # Add unknown token and padding token.
    idx_map[UNK] = len(idx_map)
    idx_map[PAD] = len(idx_map)

    return idx_map


def idx_to_token(tensor, idx_map):
    # Convert tensor indices to characters.
    tensor_indices = [int(i) for i in tensor]
    removed = {idx_map[START_SYMBOL], idx_map[END_SYMBOL], idx_map[UNK], idx_map[PAD]}
    characters = []
    for i in tensor_indices:
        for char,idx in idx_map.items():
            if idx == i and idx not in removed:
                characters.append(char)
    word = ''.join(characters)
    return(word)


def compute_tensors(input_set, idx_map):
    # Compute input tensor from source word form, source tags and target tags.
    for example in input_set:
        if 'SOURCE_TAG' in example: # task 2
            example['TENSOR'] = torch.LongTensor([idx_map[START_SYMBOL]]
                                                    + [idx_map[c] if c in idx_map
                                                                    else idx_map[UNK] 
                                                                    for c in example['SOURCE_FORM']]
                                                    + [idx_map[c] if c in idx_map
                                                                    else idx_map[UNK]
                                                                    for c in example['SOURCE_TAG'].split(',')]
                                                    + [idx_map[c] if c in idx_map
                                                                    else idx_map[UNK]
                                                                    for c in example['TARGET_TAG'].split(',')]
                                                    + [idx_map[END_SYMBOL]])
        else: # task 1, task 3
            example['TENSOR'] = torch.LongTensor([idx_map[START_SYMBOL]]
                                                    + [idx_map[c] if c in idx_map
                                                                    else idx_map[UNK] 
                                                                    for c in example['SOURCE_FORM']]
                                                    + [idx_map[c] if c in idx_map
                                                                    else idx_map[UNK]
                                                                    for c in example['TARGET_TAG'].split(',')]
                                                    + [idx_map[END_SYMBOL]])

    # Compute target tensor from target word form.
    for example in input_set:
        example['TARGET_TENSOR'] = torch.LongTensor([idx_map[START_SYMBOL]]
                                                        + [idx_map[c] if c in idx_map
                                                                        else idx_map[UNK] 
                                                                        for c in example['TARGET_FORM']]
                                                        + [idx_map[END_SYMBOL]])
    return input_set


def get_minibatch(minibatch, idx_map):
    minibatch = compute_tensors([item for item in minibatch], idx_map)

    mb_x = pad_sequence([i['TENSOR'] for i in minibatch], padding_value=idx_map[PAD])
    mb_y = pad_sequence([i['TARGET_TENSOR'] for i in minibatch], padding_value=idx_map[PAD])

    return mb_x, mb_y


def evaluate(dataset, model, eval_batch_size, idx_map):
    correct = 0
    for i in range(0,len(dataset),eval_batch_size):
        minibatch = dataset[i:i+eval_batch_size]
        mb_x, mb_y = get_minibatch(minibatch, idx_map)

        # Run input through generator.
        log_probs = model.generate(mb_x.to(DEVICE), MAX_LEN)

        # Evaluate word by word.
        for word in range(log_probs.shape[1]):
            log_probs_for_word = log_probs[:, word, :]
            target = mb_y[1:, word]

            # Find predicted character indices.
            _, predicted = torch.max(log_probs_for_word, 1)
            end_indices = torch.nonzero(predicted == 1)

            if len(end_indices) > 0:
                first_end_location = end_indices[0].item()
                predicted = predicted[:first_end_location]

            # Convert indices to characters.
            pred_word = idx_to_token(predicted, idx_map)
            target_word = idx_to_token(target, idx_map)

            # Print example predicted-target pairs.
            if i == 0 and word == 0:
                max_len = MAX_LEN - 10
                print(f"{'PRED: ' + pred_word:<{max_len}} TARGET: {target_word}")
            
            correct += (1 if pred_word == target_word else 0)

    return correct * 100.0 / len(dataset)



#--- models ---
class EncoderDecoderRNN(nn.Module):
    def __init__(self,
                input_dim,
                hidden_dim,
                output_dim,
                embedding_dim,
                n_layers,
                drop_prob):
        super(EncoderDecoderRNN, self).__init__()

        # Define parameters.
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.embedding_dim = embedding_dim
        self.n_layers = n_layers
        self.drop_prob = drop_prob

        # Define layers.
        self.embed = nn.Embedding(self.input_dim, self.embedding_dim)
        self.encoder = nn.GRU(self.hidden_dim, self.hidden_dim, self.n_layers, bidirectional=True)
        self.decoder = nn.GRU(self.embedding_dim + self.hidden_dim, self.hidden_dim, self.n_layers)
        self.linear = nn.Linear(self.hidden_dim, self.output_dim)
        self.dropout = nn.Dropout(p=self.drop_prob)


    def generate(self, inputs, max_len):
        """ Generate output (word form) one character at a time. """
        # Embed input characters.
        embeds = self.dropout(self.embed(inputs.view(inputs.size(0), -1)))
        # Get context vector from encoder.
        enc_out, enc_hidden = self.encoder(embeds)
        enc_hidden = (enc_hidden[0,:,:] + enc_hidden[1,:,:]).view(1, inputs.size(1), self.hidden_dim)

        # Context vector is the last hidden state from encoder.
        context = enc_hidden

        # First input is START_SYMBOl.
        dec_input = inputs[0]
        dec_input = dec_input.unsqueeze(0) # [batch_size] -> [1, batch_size]

        pred_outputs = torch.zeros(max_len, inputs.size(1), self.output_dim)
        #pred_outputs[0] = dec_input
        for i in range(1,max_len):
            pred_embeds = self.embed(dec_input)
            dec_in = torch.cat((pred_embeds, context), 2)
            # First iteration, initialize decoder hidden state normally.
            # After first iteration, use previous hidden state.
            dec_out, dec_hidden = self.decoder(dec_in, (None if i == 1 else dec_hidden))
            output = F.log_softmax(self.linear(dec_out.squeeze(1)), -1)
            pred_outputs[i] = output
            # Next decoder input is the previous predicted character.
            _, pred_char = torch.max((output.squeeze(0) if BATCH_SIZE > 1 else output), 1)
            dec_input = pred_char.view(1, inputs.size(1))
            # Stop if END_SYMBOL encountered.
            if dec_input.squeeze(0)[0].item() == 1:
                break
        return pred_outputs


    def forward(self, inputs, targets):
        """ Run inputs through encoder and decoder. """
        # Embedding for inputs (characters + tags)
        src_embeds = self.dropout(self.embed(inputs.view(inputs.size(0), -1)))
        # Embedding for targets (characters).
        trg_embeds = self.embed(targets.view(targets.size(0), -1))

        # Run input embeddings through encoder, apply dropout.
        enc_out, enc_hidden = self.encoder(src_embeds)

        # Account for bidirectionality.
        enc_hidden = (enc_hidden[0,:,:] + enc_hidden[1,:,:]).view(1, inputs.size(1), self.hidden_dim)

        # Decoder input = concatenation of character embeddings and encoder hidden state.
        dec_in = [torch.cat((char, enc_hidden.squeeze(0)), 1) for char in trg_embeds]
        dec_in = torch.stack(dec_in)
        dec_out, dec_hidden = self.decoder(dec_in)

        # Return log probabilities.
        output = F.log_softmax(self.linear(dec_out.squeeze(1)), -1)
        output = (output if BATCH_SIZE > 1 else output.unsqueeze(1))
        return output



#--- main ---
if __name__ == "__main__":
    #--- initialization ---
    src_path = os.path.dirname(os.path.realpath(__file__))
    project_path = os.path.dirname(src_path)
    data_dir = os.path.join(project_path, 'data')

    # Initialize languages, tasks, datasets.
    languages = {'finnish', 'german', 'navajo'}
    tasks = ['task1', 'task2', 'task3']
    datasets = {task: {lan:None for lan in languages} for task in tasks}
    for task in datasets:
        for lan in languages:
            datasets[task][lan] = read_datasets(task, lan, data_dir)
    
    idx_map = token_to_idx(datasets['task1'])
    idx_map_dim = len(idx_map)

    #--- set up ---
    models = {lan:EncoderDecoderRNN(idx_map_dim,   # input_dim
                                    HIDDEN_DIM,    # hidden_dim
                                    idx_map_dim,   # output_dim
                                    EMBEDDING_DIM, # embedding dimension
                                    N_LAYERS,      # number of layers
                                    DROP_P)        # dropout probability
                                    for lan in languages}

    optimizers = {lan:optim.Adam(models[lan].parameters(), LEARNING_RATE)
                    for lan in languages}

    loss_function = nn.NLLLoss(ignore_index=idx_map[PAD])

    #--- training ---
    for epoch in range(N_EPOCHS):
        total_loss = 0
        correct = 0
        for lan in datasets[TASK]:
            trainset = datasets[TASK][lan]['train']
            # Initialize encoder-decoder + optimizer.
            model = models[lan].to(DEVICE)
            optimizer = optimizers[lan]
            # Shuffle training set.
            shuffle(trainset)

            for i in range(0, len(trainset), BATCH_SIZE):
                minibatch = trainset[i:i+BATCH_SIZE]
                mb_x, mb_y = get_minibatch(minibatch, idx_map)

                # For input, remove END_SYMBOL from targets.
                log_probs = model(mb_x.to(DEVICE), mb_y[:-1].to(DEVICE))

                seq_len = log_probs.shape[0]
                batch_size = log_probs.shape[1]
                num_chars = log_probs.shape[2]

                predicted = log_probs.view(seq_len * batch_size, num_chars)

                # For loss optimization, remove START_SYMBOL from targets.
                target = mb_y[1:].view(seq_len * batch_size)

                loss = loss_function(predicted.to(DEVICE), target.to(DEVICE))
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()


        print('epoch: %d, loss: %.4f' % ((epoch+1), total_loss))

        if ((epoch+1) % REPORT_EVERY) == 0:
            train_acc = evaluate(trainset, model, BATCH_SIZE, idx_map)
            dev_accs = {lan:None for lan in languages}
            for lan in datasets[TASK]:
                dev_acc = evaluate(datasets[TASK][lan]['dev'],models[lan], BATCH_SIZE, idx_map)
                dev_accs[lan] = dev_acc
            for lan in dev_accs:
                print('epoch: %d, loss: %.4f, train acc: %.2f%%, dev acc: %.2f%% (%s)' % 
                        (epoch+1, total_loss, train_acc, dev_accs.get(lan), lan.capitalize()))


    # --- test ---
    test_accs = {lan:None for lan in languages}
    for lan in datasets[TASK]:
        test_acc = evaluate(datasets[TASK][lan]['test'], models[lan], BATCH_SIZE, idx_map)
        test_accs[lan] = test_acc
    for lan in test_accs:
        print('test acc: %.2f%% (%s)' % (test_accs.get(lan), lan.capitalize()))
