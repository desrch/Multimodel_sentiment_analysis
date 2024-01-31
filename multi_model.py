import torch


class Multimodel_attention_cat_trans(torch.nn.Module):
    def __init__(self, bert_model, resnet_model, num_labels, hidden_dim=64, attention_nhead=8,
                 attention_dropout=0.4, ):
        super(Multimodel_attention_cat_trans, self).__init__()

        self.bert_model = bert_model
        self.resnet_model = resnet_model
        self.fc_txt = torch.nn.Linear(self.bert_model.config.hidden_size, hidden_dim)
        self.fc_img = torch.nn.Linear(1000, hidden_dim)
        self.attention = torch.nn.TransformerEncoderLayer(
            d_model=hidden_dim * 2,
            nhead=attention_nhead,
            dropout=attention_dropout
        )
        self.fc1 = torch.nn.Linear(2 * hidden_dim, hidden_dim)
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(0.2)
        self.fuse_dropout = torch.nn.Dropout(0.5)
        self.softmax = torch.nn.Softmax(dim=1)
        self.fc_out = torch.nn.Linear(hidden_dim, num_labels)

    def forward(self, texts, images, masks):
        assert texts.shape == masks.shape, 'error! bert_inputs and masks must have same shape!'
        bert_outputs = self.bert_model(input_ids=texts, attention_mask=masks)
        text_embeddings = bert_outputs['pooler_output']
        image_embeddings = self.resnet_model(images)

        text_embeddings = self.fc_txt(text_embeddings)
        image_embeddings = self.fc_img(image_embeddings)
        text_embeddings = self.relu(text_embeddings)
        image_embeddings = self.relu(image_embeddings)
        text_embeddings = self.dropout(text_embeddings)
        image_embeddings = self.dropout(image_embeddings)

        attention_out = self.attention(torch.cat(
            [text_embeddings.unsqueeze(0), image_embeddings.unsqueeze(0)],
            dim=2)).squeeze()

        prob_vec = self.fc1(attention_out)
        prob_vec = self.relu(prob_vec)
        prob_vec = self.dropout(prob_vec)
        prob_vec = self.fc_out(prob_vec)

        logits = self.softmax(prob_vec)
        return logits


class Multimodel_decision_weighted_avg(torch.nn.Module):
    def __init__(self, bert_model, resnet_model, num_labels, hidden_dim=64):
        super(Multimodel_decision_weighted_avg, self).__init__()
        self.bert_model = bert_model
        self.resnet_model = resnet_model
        self.fc_txt = torch.nn.Linear(self.bert_model.config.hidden_size, hidden_dim)
        self.fc_img = torch.nn.Linear(1000, hidden_dim)
        self.fc1 = torch.nn.Linear(hidden_dim, 2 * hidden_dim)
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(0.2)
        self.fuse_dropout = torch.nn.Dropout(0.5)

        self.fc_out = torch.nn.Linear(2 * hidden_dim, num_labels)
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, texts, images, masks):
        assert texts.shape == masks.shape, 'error! bert_inputs and masks must have same shape!'
        bert_outputs = self.bert_model(input_ids=texts, attention_mask=masks)
        text_embeddings = bert_outputs['pooler_output']
        image_embeddings = self.resnet_model(images)

        text_embeddings = self.fc_txt(text_embeddings)
        image_embeddings = self.fc_img(image_embeddings)
        text_embeddings = self.relu(text_embeddings)
        image_embeddings = self.relu(image_embeddings)
        text_embeddings = self.dropout(text_embeddings)
        image_embeddings = self.dropout(image_embeddings)

        text_embeddings = self.fc1(text_embeddings)
        image_embeddings = self.fc1(image_embeddings)
        text_embeddings = self.relu(text_embeddings)
        image_embeddings = self.relu(image_embeddings)
        text_embeddings = self.fuse_dropout(text_embeddings)
        image_embeddings = self.fuse_dropout(image_embeddings)

        text_embeddings = self.fc_out(text_embeddings)
        image_embeddings = self.fc_out(image_embeddings)
        text_prob_vec = self.softmax(text_embeddings)
        image_prob_vec = self.softmax(image_embeddings)

        logits = self.softmax((text_prob_vec + image_prob_vec)/2)

        return logits
