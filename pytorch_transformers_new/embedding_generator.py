import os
import numpy as np
import torch
import copy
from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from pytorch_pretrained_bert import BertTokenizer, BertModel


# load data
# processor = NerProcessor()
# label_list = processor.get_labels()
# num_labels = len(label_list) + 1
#
# train_examples = processor.get_train_examples("../BERT-NER/data/")
# test_examples = processor.get_test_examples("../BERT-NER/data/")
# dev_examples = processor.get_dev_examples("../BERT-NER/data/")

print("loading embedding model")
cache_dir = os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE), 'distributed_{}'.format(0))
Embedding_model = BertModel.from_pretrained("/home/michaelchen/wwm_uncased_L-24_H-1024_A-16/")
Embedding_model.eval()
Embedding_model.to('cuda')
for i, p in enumerate(Embedding_model.parameters()):
    p.requires_grad = False
Embedding_tokenizer = BertTokenizer.from_pretrained('/home/michaelchen/wwm_uncased_L-24_H-1024_A-16/')
print("finish loading embedding model")

# print("loading embedding model")
# cache_dir = os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE), 'distributed_{}'.format(0))
# Embedding_model = BertModel.from_pretrained("bert-base-uncased")
# Embedding_model.eval()
# Embedding_model.to('cuda')
# Embedding_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# print("finish loading embedding model")

print("loading point cloud data")
PC_data = np.load('/home/michaelchen/bert-embedding/clustered-embedding-921-237.npy', allow_pickle=True)
PC_data = PC_data.item()
print("finish loading point cloud data")



# def pad_tokenids(tokens_tensor, max_len=MAX_LEN):
#     while len(tokens_tensor) < max_len:
#         tokens_tensor.append(0)
#     return tokens_tensor

def gen_embedding(input_ids, token_type_ids, model=Embedding_model):
    with torch.no_grad():
        dev = input_ids.get_device()
        # print(dev)

        model.to(device='cuda:'+str(dev))
        encoded_layers = model(input_ids, token_type_ids)

    batch_of_embeddings = encoded_layers[0][-1]
    return batch_of_embeddings


def find_words(tokens):
    words = []
    word_indices = []
    word = ""
    index = [0, 0]  # start, length
    for i in range(len(tokens)):
        if tokens[i][0].isalpha() and len(tokens[i]) > 1:
            # is a word or beginning of a word
            if word:
                words.append(word)
                word_indices.append(index)
                index = [0, 0]
            word = tokens[i]
            index[0] = i
            index[1] = 1
        elif tokens[i][:2] == "##":
            # is continuation of a word
            # Note: sometimes words start with ##, e.g 2.5million -> ##mill, ##ion
            if not word:
                word = tokens[i][2:]
                index[0] = i
                index[1] = 1
            else:
                word += tokens[i][2:]
                index[1] += 1
        else:
            # clear word cache and do nothing
            if word:
                words.append(word)
                word = ""
                word_indices.append(index)
                index = [0, 0]
    if word:
        words.append(word)
        word_indices.append(index)
    return words, word_indices

def gen_smoothed_embedding(input_ids, token_type_ids, batch_initial_embeddings, data=PC_data, tokenizer = Embedding_tokenizer):

    batch_of_embeddings = gen_embedding(input_ids, token_type_ids)

    for i in range(len(batch_of_embeddings)):
        sentence = batch_of_embeddings[i]                # [128 * 1024]
        tokens = tokenizer.convert_ids_to_tokens(input_ids[i].tolist())
        words, word_indices = find_words(tokens)
        # print(word_indices)
        # do batch matrix multiplication here to speed up (sentence-wise)
        # sentence:[128 x 1024]
        # sentence = sentence.unsqueeze(1) # sentence: [128 x 1 x 1024]
        # get word embeddings for each word strings in sentence i from the dict
        for j, word in enumerate(words):
            if word not in data.keys():
                continue
            else:
                
                pc_embeddings = torch.from_numpy(data[word]).to("cuda")
                num_subwords = pc_embeddings.shape[1] // 1024
                current_embedding = sentence[word_indices[j][0]:word_indices[j][0]+word_indices[j][1]].view([1,-1])
            
                # print(pc_embeddings.shape)
                # print(word_indices[j][0])
                # print(word_indices[j][0]+word_indices[j][1])
                # print(sentence[word_indices[j][0]+i+1].shape)
                # print(current_embedding.shape)
                # print(sentence.shape)
                if pc_embeddings.shape[1] != current_embedding.shape[1]:
                    print("Error")
                    continue
                else:
                    distance = pc_embeddings - current_embedding
                    norm = torch.norm(distance, 2) # [10 x (1024k)] - [1 x 1024k],
                    best_embedding = pc_embeddings[torch.argmin(norm)]
                    if num_subwords == 1:
                        # print('=======================')
                        # print(batch_initial_embeddings[i][word_indices[j][0]])
                        # print(best_embedding)
                        # print('=======================')
                        try:
                            batch_initial_embeddings[i][word_indices[j][0]] = best_embedding
                        except:
                            continue
                    else:
                        for i in range(num_subwords):
                            try:
                                batch_initial_embeddings[i][word_indices[j][0]+i] = best_embedding[i*1024:(i+1)*1024]
                            except:
                                continue
    return batch_initial_embeddings

        # skip_idxs = []
        # batch = []
        # for j, word_str in words:
        #     word_tensor = PC_data[word_str]  # word_str: [10 x 1024] or [10 x 1024+]
        #     if word_tensor.size()[1] > 1024:
        #         word_tensor = torch.zeros([10, 1024])
        #         # mark down this idx for later skipping
        #         skip_idxs.append((i,j)) # the jth word in the ith sentence
        #     word_tensor = torch.transpose(word_tensor, 0, 1)
        #     word_tensor = word_tensor.unsqueeze(0) # [1 x 1024 x 10]
        #     batch.append(word_tensor)
        # batch_of_words_dict = torch.cat(batch, 0) # [128 x 1024 x 10)], 128 is len(batch_of_words[i])
        #
        # sentence_sim_scores = torch.bmm(sentence, batch_of_words_dict) # [128 x 1 x 10]
