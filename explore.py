
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


def get_hidden_states(word_tokens, alr_gened = 0):
  word = tokenizer.convert_ids_to_tokens(word_tokens, skip_special_tokens=True)
  query = f"Repeat this word. 1)"
  word_tokens = word_tokens.to(device)
  query_ids = tokenizer.encode(query, return_tensors="pt").to(device)
  query_pt2_ids = tokenizer.encode("2)", return_tensors="pt").to(device)
  
  query_ids = query_ids.squeeze(0)
  query_pt2_ids = query_pt2_ids.squeeze(0)
  
  
  query_full_ids = torch.cat([query_ids, word_tokens[1:], query_pt2_ids, word_tokens[1:1+alr_gened]]).unsqueeze(0)
  
  
  outputs = model.generate(query_full_ids, max_new_tokens=1, do_sample=False, return_dict_in_generate=True, output_hidden_states=True)
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
    prob_t = torch.nn.functional.softmax(unembed, dim=0)
    probabilties.append(prob_t[word_ids[1:]])
    tokens = torch.sort(prob_t, descending=True)
    

    ranking_vec = []
    for word in word_ids[1:]:
      ranking = torch.where(tokens.values > prob_t[word])
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
  plt.yscale('log')
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
  
def get_word_representation(word_ids):
  with torch.no_grad():
    outputs = model(word_ids.to(device).unsqueeze(0), output_hidden_states=True)
  last_token_hidden = outputs.hidden_states
  word_representation = last_token_hidden[6][0][-1]
  return word_representation

def complex_metrics(probabilities, hidden_states):
  all_token_probabilities = torch.sum(probabilities, dim=1)
  all_token_proportions = probabilities / all_token_probabilities.unsqueeze(-1)
  
  summed_probabilities = torch.sum(probabilities, dim=1)
  
  word_representation = get_word_representation(word_ids).unsqueeze(0)
  dist_to_word_representation = torch.nn.functional.cosine_similarity(hidden_states, word_representation, dim=1)
  
  
  return all_token_proportions, summed_probabilities, dist_to_word_representation
  
def plot_proportions(proportions):
  plt.figure(figsize=(8, 5))  # Set figure size
  x = np.arange(0, len(proportions))
  for i in range(proportions.shape[1]):
    plt.plot(x, proportions[:, i].to('cpu').detach().type(torch.float32).numpy(), label=f"Token {i}", linestyle="-", marker="o")

  # Add labels and title
  plt.xlabel("X-axis")
  plt.ylabel("Y-axis")
  plt.title("Proportions")

  # Add a legend
  plt.legend()  
  plt.savefig(f"proportions.png")
  plt.close()

def plot_summed_probabilities(summed_probabilities):
  plt.figure(figsize=(8, 5))  # Set figure size
  x = np.arange(0, len(summed_probabilities))
  plt.plot(x, summed_probabilities.to('cpu').detach().type(torch.float32).numpy(), label="Summed Probabilities", linestyle="-", marker="o")

  plt.xlabel("X-axis")
  plt.ylabel("Y-axis")
  plt.title("Summed Probabilities")

  # Add a legend
  plt.legend()
  plt.savefig(f"summed_probabilities.png")
  plt.close()

def plot_dist_to_word_representation(dist_to_word_representation):
  plt.figure(figsize=(8, 5))  # Set figure size
  x = np.arange(0, len(dist_to_word_representation))
  plt.plot(x, dist_to_word_representation.to('cpu').detach().type(torch.float32).numpy(), label="Dist to Word Representation", linestyle="-", marker="o")

  plt.xlabel("X-axis")
  plt.ylabel("Y-axis")
  plt.title("Dist to Word Representation")

  # Add a legend
  plt.legend()
  plt.savefig(f"dist_to_word_representation.png")
  plt.close()

from tqdm import tqdm
import gc
if __name__ == "__main__":
  torch.manual_seed(42)
  
  multitoken_words, token_counts_per_word = get_multitoken_words(emma)

  num_tokens = 2
  token_words = multitoken_words[token_counts_per_word == (num_tokens + 1)] #one for begining of seq token
  random_sample_indices = torch.multinomial(torch.ones(len(token_words)), min(30, len(token_words)), replacement=False)

  avg_probabilities = []
  avg_summed_probabilities = []
  avg_dist_to_word_representation = []
  
  avg_metrics = []
  avg_rankings = []
  
  
  for idx in tqdm(random_sample_indices):
    word = token_words[idx]
    word_ids = word[:(num_tokens + 1)]
    hidden_states = get_hidden_states(word_ids, alr_gened = 0)
    probabilities, rankings = get_probabilities(hidden_states, word_ids)
    
    hidden_states = torch.stack(hidden_states[0], dim=0).squeeze(1)
    
    
    proportions, summed_probabilities, dist_to_word_representation = complex_metrics(probabilities, hidden_states[:, -1, :])
    
    
    avg_metrics.append(proportions)
    avg_summed_probabilities.append(summed_probabilities)
    avg_dist_to_word_representation.append(dist_to_word_representation)
    
    avg_probabilities.append(probabilities)
    avg_rankings.append(rankings)
    
    torch.cuda.empty_cache()
    gc.collect()

  avg_metrics = torch.stack(avg_metrics, dim=0).mean(dim=0)
  avg_summed_probabilities = torch.stack(avg_summed_probabilities, dim=0).mean(dim=0)
  avg_dist_to_word_representation = torch.stack(avg_dist_to_word_representation, dim=0).mean(dim=0)
  
  
  avg_probabilities = torch.stack(avg_probabilities, dim=0).mean(dim=0)
  avg_rankings = torch.stack(avg_rankings, dim=0).median(dim=0).values
  

  plot_probabilities(avg_probabilities, avg_rankings)
  plot_proportions(avg_metrics)
  plot_summed_probabilities(avg_summed_probabilities)
  plot_dist_to_word_representation(avg_dist_to_word_representation)
