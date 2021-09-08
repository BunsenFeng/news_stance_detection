# user feature generation: RoBERTa with bio
from transformers import pipeline
import torch
from transformers import *
pretrained_weights = 'roberta-base'
tokenizer = RobertaTokenizer.from_pretrained(pretrained_weights)
feature_extractor = pipeline('feature-extraction', model = RobertaModel.from_pretrained(pretrained_weights), tokenizer = tokenizer, device = 1)

import torch
import json
import numpy
import datetime

user_features = []
f = open('article.txt')
i = 0
for line in f:
    article_features = []
    print(i)
    i += 1
    thing = line.strip().split('<SEP>')
    for para in thing:
        text = para.strip()
        result = tokenizer(text)
        while len(result['input_ids']) > 500:
            text = text[:int(len(text)/2)]
            result = tokenizer(text)
        feature_temp = torch.tensor(feature_extractor(text))
        feature_temp = torch.mean(feature_temp.squeeze(0), dim=0).unsqueeze(0)
        article_features.append(feature_temp)
    user_features.append(article_features)

#user_features = torch.stack(user_features).squeeze(1)
torch.save(user_features, 'article_roberta.pt')
temp = torch.load('article_roberta.pt')
print(temp[0][0].size())