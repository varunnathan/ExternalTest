import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, PreTrainedModel, BertModel
from transformers.modeling_outputs import SequenceClassifierOutput, QuestionAnsweringModelOutput
from main.data_model.config import ModelConfig
from main.train.utils import ModuleType


# Tokenizer
def build_tokenizer(model_config: ModelConfig):
    tok = AutoTokenizer.from_pretrained(model_config.model_name, use_fast=True)
    return tok


# Model Class
class IntentHead(PreTrainedModel):
    config_class = None

    def __init__(self, tok: AutoTokenizer, model_config: ModelConfig):
        super().__init__(AutoModel.from_pretrained(model_config.model_name).config)
        self.bert = AutoModel.from_pretrained(model_config.model_name)
        self.bert.resize_token_embeddings(len(tok))
        hidden = self.bert.config.hidden_size
        self.ctx_embed = nn.Embedding(model_config.ctx_dim[ModuleType.INTENT], hidden)
        self.ff = nn.Linear(hidden, hidden)
        self.v = nn.Linear(hidden, 1, bias=False)
        self.loss = nn.CrossEntropyLoss()

    def _restricted_attn_mask(self, intent_ids, attn_mask):
        # intent_ids: (B,L)  -1 utterance/SEP, 0..n-1 for intent tokens
        B, L = intent_ids.size()
        base = attn_mask.unsqueeze(1).repeat(1,L,1)  # (B,L,L)
        for b in range(B):
            ids = intent_ids[b]
            for i in range(L):
                if ids[i] >= 0:                       # token belongs to intent k
                    k = ids[i]
                    allowed = (ids == k) | (ids == -1)
                    base[b,i] &= allowed
        # convert bool→float, 1 allowed, 0 blocked → use -inf
        ext = (~base).float()*-1e4
        return ext.unsqueeze(1)  # (B,1,L,L)

    def forward(self, input_ids, token_type_ids, context_ids, intent_ids,
                attention_mask, intent_mask, label_id):
        # build embeddings
        pos_ids = torch.arange(input_ids.size(1), device=input_ids.device)
        pos_emb = self.bert.embeddings.position_embeddings(pos_ids)
        emb = self.bert.embeddings.word_embeddings(input_ids) \
            + self.bert.embeddings.token_type_embeddings(token_type_ids) \
            + self.ctx_embed(context_ids) \
            + pos_emb
        emb = self.bert.embeddings.LayerNorm(emb)
        emb = self.bert.embeddings.dropout(emb)
        ext_mask = self._restricted_attn_mask(intent_ids, attention_mask)
        enc = self.bert.encoder(
                hidden_states=emb,
                attention_mask=ext_mask,
                head_mask=None,
                output_attentions=False,
                output_hidden_states=False,
                return_dict=False,
        )[0]
        # max‑pool per intent
        B, L, H = enc.size()
        max_intents = intent_mask.size(1)
        rep = torch.full((B, max_intents, H), -1e4, device=enc.device)
        for k in range(max_intents):
            mask_k = (intent_ids == k).unsqueeze(-1)    # (B,L,1)
            masked = enc.masked_fill(~mask_k, -1e4)
            rep[:, k] = masked.max(dim=1).values
        rep = rep.masked_fill(~intent_mask.unsqueeze(-1).bool(), 0)
        s = self.v(torch.tanh(self.ff(rep))).squeeze(-1)  # (B,max_intents)
        loss = self.loss(s, label_id)
        return {"logits": s, "loss": loss}
    

class CatSlotHead(nn.Module):
    def __init__(self, model_config: ModelConfig):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_config.model_name)
        hid = self.bert.config.hidden_size
        self.ctx_emb = nn.Embedding(model_config.ctx_dim[ModuleType.CAT_SLOT], hid)
        self.ff = nn.Linear(hid, hid)
        self.v  = nn.Linear(hid, 1, bias=False)
        self.layer_norm = self.bert.embeddings.LayerNorm
        self.dropout    = self.bert.embeddings.dropout
    
    def forward(self, input_ids, token_type_ids, context_ids,
                attention_mask, val_spans, labels=None):
        base = self.bert.embeddings(input_ids=input_ids,
                                    token_type_ids=token_type_ids,
                                    position_ids=None)
        out  = base + self.ctx_emb(context_ids)
        out  = self.layer_norm(out)
        out  = self.dropout(out)
        
        ext = attention_mask.unsqueeze(1)          # (B,1,L,L)
        enc = self.bert.encoder(hidden_states=out,
                                attention_mask=ext,
                                return_dict=False)[0]           # (B,L,H)

        reps = []   # max‑pooled span reps
        for b, spans in enumerate(val_spans):
            vecs = [torch.zeros_like(enc[b, 0]) if s == -1
                    else enc[b, s: e].amax(dim=0) for s, e in spans]
            reps.append(torch.stack(vecs))
        reps   = torch.stack(reps)                 # (B,V,H)
        
        logits = self.v(torch.tanh(self.ff(reps))).squeeze(-1)  # (B,V)
        
        loss   = F.cross_entropy(logits, labels) if labels is not None else None
        return SequenceClassifierOutput(loss=loss, logits=logits)
    

class FreeSlotHead(nn.Module):
    """
    Free-form slot value module (SQuAD-style start/end prediction).
    """
    def __init__(self, model_config: ModelConfig):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_config.model_name)
        hid = self.bert.config.hidden_size
        self.ctx_emb = nn.Embedding(model_config.ctx_dim[ModuleType.FREE_SLOT], hid)
        self.start_vec = nn.Parameter(torch.zeros(hid))
        self.end_vec   = nn.Parameter(torch.zeros(hid))
        nn.init.normal_(self.start_vec, std=0.02)
        nn.init.normal_(self.end_vec,   std=0.02)

        self.layer_norm = self.bert.embeddings.LayerNorm
        self.dropout    = self.bert.embeddings.dropout

    def forward(self,
                input_ids, token_type_ids, context_ids,
                attention_mask, start_positions=None, end_positions=None):

        emb = self.bert.embeddings(input_ids=input_ids,
                                   token_type_ids=token_type_ids)
        emb = emb + self.ctx_emb(context_ids)
        emb = self.layer_norm(emb)
        emb = self.dropout(emb)

        enc = self.bert.encoder(hidden_states=emb,
                                attention_mask=attention_mask.unsqueeze(1),
                                return_dict=False)[0]        # (B,L,H)

        # eqns (3) & (4)
        start_logits = torch.matmul(enc, self.start_vec)     # (B,L)
        end_logits   = torch.matmul(enc, self.end_vec)

        loss = None
        if start_positions is not None and end_positions is not None:
            loss = (F.cross_entropy(start_logits, start_positions) +
                    F.cross_entropy(end_logits,   end_positions)) / 2

        return QuestionAnsweringModelOutput(loss=loss,
                                            start_logits=start_logits,
                                            end_logits=end_logits)
    

class ReqSlotHead(nn.Module):
    """
    Requested-slot classifier: [CLS] -> tanh(W r) -> vᵀ -> sigmoid
    """
    def __init__(self, model_config: ModelConfig):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_config.model_name)
        hid = self.bert.config.hidden_size
        self.ctx_emb = nn.Embedding(model_config.ctx_dim[ModuleType.REQUESTED_SLOT], hid)
        self.ff = nn.Linear(hid, hid)
        self.v  = nn.Linear(hid, 1, bias=False)

        self.layer_norm = self.bert.embeddings.LayerNorm
        self.dropout    = self.bert.embeddings.dropout

    def forward(self, input_ids, token_type_ids, context_ids,
                attention_mask, labels=None):

        emb = self.bert.embeddings(input_ids=input_ids,
                                   token_type_ids=token_type_ids)
        emb = emb + self.ctx_emb(context_ids)
        emb = self.layer_norm(emb)
        emb = self.dropout(emb)

        enc = self.bert.encoder(hidden_states=emb,
                                attention_mask=attention_mask.unsqueeze(1),
                                return_dict=False)[0]    # (B,L,H)
        cls = enc[:, 0]                                  # [CLS] vector
        logits = self.v(torch.tanh(self.ff(cls))).squeeze(-1)  # (B,)

        loss = (F.binary_cross_entropy_with_logits(logits.float(), labels.float())
                if labels is not None else None)

        return SequenceClassifierOutput(loss=loss, logits=logits)
    

class InXferHead(nn.Module):
    """
    In-domain slot-transfer binary classifier (sequence-level on [CLS]).
    """
    def __init__(self, model_config: ModelConfig):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_config.model_name)
        hid = self.bert.config.hidden_size
        self.ctx_emb = nn.Embedding(model_config.ctx_dim[ModuleType.IN_DOMAIN_SLOT], hid)
        self.ff = nn.Linear(hid, hid)
        self.v  = nn.Linear(hid, 1, bias=False)

        self.ln = self.bert.embeddings.LayerNorm
        self.dp = self.bert.embeddings.dropout

    def forward(self, input_ids, token_type_ids, context_ids,
                attention_mask, labels=None):

        emb = self.bert.embeddings(input_ids=input_ids,
                                   token_type_ids=token_type_ids)
        emb = emb + self.ctx_emb(context_ids)
        emb = self.ln(emb)
        emb = self.dp(emb)

        enc = self.bert.encoder(hidden_states=emb,
                                attention_mask=attention_mask.unsqueeze(1),
                                return_dict=False)[0]
        cls = enc[:, 0]
        logits = self.v(torch.tanh(self.ff(cls))).squeeze(-1)

        loss = (F.binary_cross_entropy_with_logits(logits.float(), labels.float())
                if labels is not None else None)

        return SequenceClassifierOutput(loss=loss, logits=logits)
    

class CrossXferHead(nn.Module):
    """
    Cross-domain slot-transfer binary classifier on [CLS].
    """
    def __init__(self, model_config: ModelConfig):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_config.model_name)
        hid = self.bert.config.hidden_size
        self.ctx_emb = nn.Embedding(model_config.ctx_dim[ModuleType.CROSS_DOMAIN_SLOT], hid)
        self.ff = nn.Linear(hid, hid)
        self.v  = nn.Linear(hid, 1, bias=False)
        self.ln = self.bert.embeddings.LayerNorm
        self.dp = self.bert.embeddings.dropout

    def forward(self, input_ids, token_type_ids, context_ids,
                attention_mask, labels=None):
        emb = self.bert.embeddings(input_ids=input_ids,
                                   token_type_ids=token_type_ids)
        emb = emb + self.ctx_emb(context_ids)
        emb = self.ln(emb)
        emb = self.dp(emb)

        enc = self.bert.encoder(hidden_states=emb,
                                attention_mask=attention_mask.unsqueeze(1),
                                return_dict=False)[0]
        cls = enc[:, 0]
        logits = self.v(torch.tanh(self.ff(cls))).squeeze(-1)
        loss = (F.binary_cross_entropy_with_logits(logits.float(), labels.float())
                if labels is not None else None)
        return SequenceClassifierOutput(loss=loss, logits=logits)
