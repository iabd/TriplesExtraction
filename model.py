import torch
import torch.nn as nn
from transformers import BertModel, BertConfig
from config import Config

TRANSFORMERS = {
    "bert-base-cased": {
        "model_config": (BertModel, BertConfig),
    },
    "bert-base-uncased": {
        "model_config": (BertModel, BertConfig),
    }
}

class Transformer(nn.Module):
    def __init__(self, model, maxlen=128):
        super().__init__()
        self.name = model
        model_type, config_type = TRANSFORMERS[model]['model_config']
        if Config.pretrained:
            self.transformer = model_type.from_pretrained(model, output_hidden_states=True,
                                                          num_labels=Config.num_labels)
        else:
            config_file = TRANSFORMERS[model]['config']
            config = config_type.from_json_file(config_file)
            config.num_labels = Config.num_labels
            config.output_hidden_states = True
            self.transformer = model_type(config)

        self.nb_features = self.transformer.pooler.dense.out_features
        if "roberta" in self.name:
            self.pad_idx = 1
        else:
            self.pad_idx = 0
        self.logits = nn.Sequential(
            nn.Linear(self.nb_features, self.nb_features),
            nn.Tanh(),
            nn.Linear(self.nb_features, Config.num_labels),
        )

    def forward(self, input_ids):
        hidden_states = self.transformer(
            input_ids,
            attention_mask=(input_ids != self.pad_idx).long(),
        )[-1]

        features = hidden_states[-1]
        logits = torch.sigmoid(self.logits(features))

        return logits