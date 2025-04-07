
import nltk
nltk.download('gutenberg')
emma = nltk.corpus.gutenberg.words('austen-emma.txt')

import os


import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import torch.nn.functional as F
import torch

import matplotlib.pyplot as plt


device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "meta-llama/Llama-3.1-8B-Instruct"
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)

def get_multitoken_words(words):
  tokenizer.add_special_tokens({'pad_token': '[PAD]'})
  tokenized_words = []
  tokens_of_words = []
  for batch in range((len(words) // 99) + 1):
    tokenizer_run = tokenizer(words[batch * 99:(batch + 1) * 99], return_tensors="pt", padding=True)
    tokenized_words.append(tokenizer_run['attention_mask'])
    tokens_of_words.append(tokenizer_run['input_ids'])

  # tokenized_words = torch.stack(tokenized_words, dim=0)
  max_cols = max(t.size(1) for t in tokenized_words)

  # Pad each tensor along the last dimension if necessary
  padded_tensors_attention = [
      F.pad(t, (0, max_cols - t.size(1)), mode='constant', value=0)
      for t in tokenized_words
  ]

  padded_tensors_ids = [
      F.pad(t, (0, max_cols - t.size(1)), mode='constant', value=0)
      for t in tokens_of_words
  ]

  # Concatenate along the row dimension (dim=0)
  tokenized_words_tensor = torch.cat(padded_tensors_attention, dim=0)
  tokens_of_words = torch.cat(padded_tensors_ids, dim=0)
  

  token_counts_per_word = torch.count_nonzero(tokenized_words_tensor, dim=1)
  index = 1
  multitoken_indices = (token_counts_per_word >= 3).nonzero().squeeze()
  multitoken_words = tokens_of_words[multitoken_indices]
  token_counts_per_word = token_counts_per_word[multitoken_indices]
  return multitoken_words, token_counts_per_word


def get_hidden_states(word_tokens):
  word = tokenizer.convert_ids_to_tokens(word_tokens, skip_special_tokens=True)
  query = f"Repeat this word. 1){word} 2)"
  query_ids = tokenizer.encode(query, return_tensors="pt").to(device)
  print("Length of query: ", len(query_ids[0]))
  outputs = model.generate(query_ids, max_new_tokens=1, do_sample=False, return_dict_in_generate=True, output_hidden_states=True)
  output_ids = outputs.sequences
  hidden_states = outputs.hidden_states
  return hidden_states


def get_probabilities(hidden_states, word_ids):
  first_gen = hidden_states[0]
  last_hidden = first_gen[-1].squeeze(0)[-1]
  unembed = model.lm_head.forward(last_hidden)
  token = torch.argmax(unembed)
  
  word_tokens = tokenizer.convert_ids_to_tokens(word_ids, skip_special_tokens=True)

  probabilties = []
  rankings = []

  for layer in first_gen:
    layer = layer.squeeze(0)[-1]


    unembed = model.lm_head.forward(layer)
    probabilties.append(unembed[word_ids[1:]])
    tokens = torch.sort(unembed, descending=True)

    ranking_vec = []
    for word in word_ids[1:]:
      ranking = torch.where(tokens.values > unembed[word])
      ranking = torch.count_nonzero(ranking[0])
      ranking_vec.append(ranking)

    rankings.append(ranking_vec)


  rankings = torch.Tensor(rankings)
  probabilties = torch.stack(probabilties, dim=0)

  return probabilties, rankings


def plot_probabilities(probabilities, rankings):
  plt.figure(figsize=(8, 5))  # Set figure size
  probabilities = probabilities.to('cpu').detach().type(torch.float32)
  x = np.arange(0, len(probabilities))
  for i in range(probabilities.shape[1]):
    plt.plot(x, probabilities[:, i].numpy(), label=f"Prob of Token {i}", linestyle="-", marker="o")

  # Add labels and title
  plt.xlabel("X-axis")
  plt.ylabel("Y-axis")
  plt.title("Probability of Tokens")

  # Add a legend
  plt.legend()

  # Show the plot
  plt.savefig(f"probabilities.png")
  plt.close()

  plt.figure(figsize=(8, 5))  # Set figure size
  rankings = rankings.to('cpu').detach().type(torch.float32)
  x = np.arange(0, len(rankings))
  plt.yscale('log')
  for i in range(rankings.shape[1]):
    plt.plot(x, rankings[:, i].numpy() + 1, label=f"Ranking of Token {i}", linestyle="-", marker="o")

  # Add labels and title
  plt.xlabel("X-axis")
  plt.ylabel("Y-axis")
  plt.title("Ranking of Tokens")

  # Add a legend
  plt.legend()

  # Show the plot
  plt.savefig(f"rankings.png")
  plt.close()
  
if __name__ == "__main__":
  torch.manual_seed(42)
  
  multitoken_words, token_counts_per_word = get_multitoken_words(emma)

  num_tokens = 3
  token_words = multitoken_words[token_counts_per_word == (num_tokens + 1)] #one for begining of seq token
  random_sample_indices = torch.randint(0, len(token_words), (100,))

  avg_probabilities = []
  avg_rankings = []
  for idx in random_sample_indices:
    word = token_words[idx]
    word_ids = word[:(num_tokens + 1)]
    hidden_states = get_hidden_states(word_ids)
    probabilities, rankings = get_probabilities(hidden_states, word_ids)
    avg_probabilities.append(probabilities)
    avg_rankings.append(rankings)

  avg_probabilities = torch.stack(avg_probabilities, dim=0).mean(dim=0)
  avg_rankings = torch.stack(avg_rankings, dim=0).median(dim=0).values

  plot_probabilities(avg_probabilities, avg_rankings)