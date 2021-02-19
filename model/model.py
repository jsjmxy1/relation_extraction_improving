import torch
import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel
from .encoder import CNNEncoder, PCNNEncoder
from .embedding import WordPosEmbedding, WordEmbedding
from .selector import BagAttention, BagAttention_v1
import pdb

from transformer.SubLayers import MultiHeadAttention, PositionwiseFeedForward


class BasePCNNAttModel(BaseModel):
    def __init__(
        self,
        word_vec_mat,
        word_embedding_dim,
        pos_embedding_dim,
        max_length,
        hidden_size,
        kernel_size,
        stride_size,
        activation,
        dropout_prob,
        relation_num,
    ):
        super(BasePCNNAttModel, self).__init__()
        self.word_pos_embedding = WordPosEmbedding(
            word_vec_mat,
            word_embedding_dim=word_embedding_dim,
            pos_embedding_dim=pos_embedding_dim,
            max_length=max_length,
        )

        self.word_embedding = WordEmbedding(word_vec_mat, word_embedding_dim)

        #self.self_attention = MultiHeadAttention(n_head=3, d_model=60, d_k=20, d_v=20, dropout=0.1)
        #self.pos_ffn = PositionwiseFeedForward(d_in=690, d_hid=53, dropout=0.1)
        #self.layer_norm = nn.LayerNorm(690, eps=1e-6)
        #self.rrr = nn.Parameter(torch.ones(word_embedding_dim), requires_grad=True)   # relation query
        self.flc = nn.Linear(3 * hidden_size, word_embedding_dim, bias=True)
        self.ff_sml = nn.Sequential(
            nn.Linear(1, 50),
            nn.Sigmoid(),
            nn.Linear(50, 1),
        )

        self.sentence_encoder = PCNNEncoder(
            word_embedding_dim + 2 * pos_embedding_dim,
            hidden_size,
            kernel_size,
            stride_size,
            activation,
            dropout_prob=dropout_prob,
        )
        self.bag_aggregater = BagAttention_v1(
            relation_num, 3 * hidden_size, dropout_prob=dropout_prob
        )
        
    def __dist__(self, x, y, dim, score=None):
        if score is None:
            return (torch.pow(x - y, 2)).sum(dim)
        else:
            return (torch.pow(x - y, 2) * score).sum(dim)
        
    def b2(self, s):
        zero = torch.zeros_like(s)
        one = torch.ones_like(s)
        ss = torch.where(s > 0.16, zero, s)
        sss = torch.where(ss > 0, one, ss)
        return sss

    def forward(self, data, is_train=True):
        """
        train:
        final_logit: [num_bag, r]
        test:
        final_logit: [num_bag, r, r]
        """
        word = data["sentences"]["word"]
        pos1 = data["sentences"]["pos1"]
        pos2 = data["sentences"]["pos2"]
        mask = data["sentences"]["mask"]
        bag_id = data["sentences"]["bag_id"]
        in_bag_index = data["sentences"]["in_bag_index"]
        num_bag = data["num_bag"]
        max_bag = data["max_bag"]
        size = data["sentences"]['size']
        scope = data["sentences"]["scope"]
        relation = data["sentences"].get("relation", None)

        #pdb.set_trace()
        head = data["sentences"]["head"]
        tail = data["sentences"]["tail"]
        head_vec = self.word_embedding(head)
        tail_vec = self.word_embedding(tail)

        x = self.word_pos_embedding(word, pos1, pos2)  # [148, 120, 60]
        #xx = self.self_attention(x, x, x, None)
        xxx = self.sentence_encoder(x, mask)   # [148, 690]
        #xxx = self.pos_ffn(xxx)
        
        y = self.flc(xxx)
        ent_query = tail_vec - head_vec
        sml = self.__dist__(y, ent_query, dim=1)
        #sml = torch.cosine_similarity(y, ent_query, dim=1)
        sml = self.ff_sml(sml.view(-1, 1)).squeeze(1)   #[148]
        sml_res = self.b2(sml).long()
        new_relation = torch.where(sml_res>0, relation, sml_res)
        
        _, final_logit, _ = self.bag_aggregater(
            xxx,
            bag_id,
            in_bag_index,
            num_bag,
            max_bag,
            relation=new_relation,
            is_train=is_train,
        )   # [148, 53]
        # final_logit = self.layer_norm(final_logit)
        return final_logit

class TablePCNNAttModel_v1(BaseModel):
    def __init__(
        self,
        word_vec_mat,
        word_embedding_dim,
        pos_embedding_dim,
        max_length,
        hidden_size,
        kernel_size,
        stride_size,
        activation,
        dropout_prob,
        relation_num,
    ):
        super(TablePCNNAttModel_v1, self).__init__()
        self.word_pos_embedding = WordPosEmbedding(
            word_vec_mat,
            word_embedding_dim=word_embedding_dim,
            pos_embedding_dim=pos_embedding_dim,
            max_length=max_length,
        )

        self.word_embedding = WordEmbedding(word_vec_mat, word_embedding_dim)
        
        #self.self_attention = MultiHeadAttention(n_head=3, d_model=60, d_k=20, d_v=20, dropout=0.1)
        #self.pos_ffn = PositionwiseFeedForward(d_in=60, d_hid=230, dropout=0.1)
        #self.rrr = nn.Parameter(torch.ones(word_embedding_dim), requires_grad=True)   # relation query
        self.flc = nn.Linear(3 * hidden_size, word_embedding_dim, bias=True)
        self.ff_sml = nn.Sequential(
            nn.Linear(1, 50),
            nn.Sigmoid(),
            nn.Linear(50, 1),
        )

        self.sentence_encoder = PCNNEncoder(
            word_embedding_dim + 2 * pos_embedding_dim,
            hidden_size,
            kernel_size,
            stride_size,
            activation,
            dropout_prob=dropout_prob,
        )
        self.bag_aggregater = BagAttention(
            relation_num, 3 * hidden_size, dropout_prob=dropout_prob
        )
        self.table_aggregrater = nn.Linear(9 * hidden_size, 1, bias=True)
        self.bias = nn.Parameter(torch.zeros(1, relation_num))
        self.dropout = nn.Dropout(dropout_prob)

    def __dist__(self, x, y, dim, score=None):
        if score is None:
            return (torch.pow(x - y, 2)).sum(dim)
        else:
            return (torch.pow(x - y, 2) * score).sum(dim)
        
    def b2(self, s):
        zero = torch.zeros_like(s)
        one = torch.ones_like(s)
        ss = torch.where(s > 0.16, zero, s)
        sss = torch.where(ss > 0, one, ss)
        return sss
    
    def _final_logit(self, final_repr, is_train=True):
        """
        training: bag_repr [num_bag, hidden_size]
        testing: bag_repr [num_bag, r, hidden_size]
        """
        logits = torch.matmul(
            final_repr, torch.transpose(self.bag_aggregater.relation_embedding.weight, 0, 1)
        )
        if is_train:
            logits += self.bias
        else:
            logits += self.bias.unsqueeze(1)
        return logits

    def forward(self, data, is_train=True):
        """
        train:
        final_logit: [num_bag, r]
        test:
        final_logit: [num_bag, r, r]
        """
        # sentences
        word = data["sentences"]["word"]
        pos1 = data["sentences"]["pos1"]
        pos2 = data["sentences"]["pos2"]
        mask = data["sentences"]["mask"]
        bag_id = data["sentences"]["bag_id"]
        in_bag_index = data["sentences"]["in_bag_index"]
        num_bag = data["num_bag"]
        max_bag = data["max_bag"][0]
        size = data["sentences"]['size']
        scope = data["sentences"]["scope"]
        relation = data["sentences"].get("relation", None)

        head = data["sentences"]["head"]
        tail = data["sentences"]["tail"]
        head_vec = self.word_embedding(head)
        tail_vec = self.word_embedding(tail)

        #pdb.set_trace()
        x = self.word_pos_embedding(word, pos1, pos2)
        #xx = self.self_attention(x, x, x, None)
        #xx = self.pos_ffn(xx)    # [148, 120, 60]
        xxx = self.sentence_encoder(x, mask)   # [148, 690]
        
        y = self.flc(xxx)
        ent_query = tail_vec - head_vec
        sml = self.__dist__(y, ent_query, dim=1)
        #sml = torch.cosine_similarity(y, ent_query, dim=1)
        sml = self.ff_sml(sml.view(-1, 1)).squeeze(1)   #[148]
        sml_res = self.b2(sml).long()
        new_relation = torch.where(sml_res>0, relation, sml_res)
        
        s_repr, s_final_logit,s_att = self.bag_aggregater(
            xxx,
            size,
            scope,
            relation=new_relation,
            is_train=is_train,
        )
        # extend_sentences
        word = data["extend_sentences"]["word"]
        pos1 = data["extend_sentences"]["pos1"]
        pos2 = data["extend_sentences"]["pos2"]
        mask = data["extend_sentences"]["mask"]
        bag_id = data["extend_sentences"]["bag_id"]
        in_bag_index = data["extend_sentences"]["in_bag_index"]
        max_bag = data["max_bag"][1]
        size = data["extend_sentences"]['size']
        scope = data["extend_sentences"]["scope"]
        relation = data["extend_sentences"].get("relation", None)

        head = data["extend_sentences"]["head"]
        tail = data["extend_sentences"]["tail"]
        head_vec = self.word_embedding(head)
        tail_vec = self.word_embedding(tail)

        if word.shape[0] == 0:
            return s_final_logit
        else:
            x = self.word_pos_embedding(word, pos1, pos2)
            #xx = self.self_attention(x, x, x, None)
            #xx = self.pos_ffn(xx)    # [148, 120, 60]
            xxx = self.sentence_encoder(x, mask)   # [148, 690]
            
            y = self.flc(xxx)
            ent_query = tail_vec - head_vec
            sml = self.__dist__(y, ent_query, dim=1)
            #sml = torch.cosine_similarity(y, ent_query, dim=1)
            sml = self.ff_sml(sml.view(-1, 1)).squeeze(1)   #[148]
            sml_res = self.b2(sml).long()
            new_relation = torch.where(sml_res>0, relation, sml_res)
            
            es_repr, es_final_logit, es_att = self.bag_aggregater(
                xxx,
                size,
                scope,
                relation=new_relation,
                is_train=is_train,
            )
            extend_mask = (size == 0).unsqueeze(1)
            relation = data.get("relation", None)
            if is_train:
                # s_repr: [num_bag, hidden_size]
                relation_v = self.bag_aggregater.relation_embedding(relation)
                alpha = F.sigmoid(self.table_aggregrater(torch.cat((s_repr, es_repr, relation_v), dim=1)))
            else:
                # s_repr: [num_bag, r, hidden_size]
                alpha = F.sigmoid(self.table_aggregrater(torch.cat((s_repr, es_repr, self.bag_aggregater.relation_embedding.weight.unsqueeze(0).repeat(num_bag, 1, 1)), dim=2)))
                extend_mask = extend_mask.unsqueeze(2)
            masked_alpha = alpha * (~extend_mask).float() + extend_mask.float()
            final_repr = masked_alpha * s_repr + (1 - masked_alpha) * es_repr
            final_logit = self._final_logit(final_repr, is_train)
            return final_logit
