import torch.nn as nn
import torchvision
import torch
import numpy as np
from einops import rearrange
from torch.nn.functional import kl_div, softmax, log_softmax
from loss import RankingLoss, CosineLoss, NewCrossEntropyLoss, NewCrossEntropyLoss2,SampleCrossEntropyLoss, NewCrossEntropyLoss3,ClipLoss
import torch.nn.functional as F
import torchvision.models as models

import copy
from sklearn.metrics import roc_auc_score
from models.BCN import BatchChannelNorm
class ZSLNet(nn.Module):
    def __init__(self, args, textual_embeddings=None, device='cpu'):
        super(ZSLNet, self).__init__()
        self.args = args
        self.device = device
        self.statistical_categories = []
        self.batchi = 0

        self.class_ids_loaded_unseen = np.array([9, 6, 10, 11])
        self.vision_backbone = getattr(torchvision.models, self.args.vision_backbone)(pretrained=self.args.pretrained)
        print(f'加载模型：{self.args.vision_backbone}')

        classifiers = ['classifier', 'fc']
        for classifier in classifiers:
            cls_layer = getattr(self.vision_backbone, classifier, None)
            if cls_layer is None:
                continue
            d_visual = cls_layer.in_features
            setattr(self.vision_backbone, classifier, nn.Identity(d_visual))
            break
        if self.args.bce_only:
            self.bce_loss = torch.nn.BCELoss(size_average=True)
        else:
            self.textual_embeddings = torch.from_numpy(textual_embeddings).to(self.device)
            self.emb_loss = CosineLoss()
            self.cross_loss = NewCrossEntropyLoss()
            self.cross_loss_2 = NewCrossEntropyLoss()
            self.bce_loss = nn.BCEWithLogitsLoss()
            self.sample_cross_loss = SampleCrossEntropyLoss()

            self.ranking_loss = RankingLoss(neg_penalty=self.args.neg_penalty)

            d_textual = self.textual_embeddings.shape[-1]
            self.d_space = 128

            self.fc_v = nn.Sequential(
                nn.Linear(d_visual, 512),

                nn.ReLU(),
                nn.Linear(512, 256),

                nn.ReLU(),
                nn.Linear(256, 128),
            )

            self.fc_t = nn.Sequential(
                nn.Linear(d_textual, 512),

                nn.ReLU(),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
            )

            self.beta = 0.45

            self.bcn = BatchChannelNorm(num_channels=3)

    def forward(self, x, labels=None, class_index=None, epoch=0, n_crops=0, bs=16):
        if self.args.bce_only:
            return self.forward_bce_only(x, labels=labels, n_crops=n_crops, bs=bs)
        else:
            return self.forward_ranking201(x, labels=labels, class_index=class_index, epoch=epoch, n_crops=n_crops, bs=bs)

    def forward_bce_only(self, x, labels=None, n_crops=0, bs=16):
        lossvalue_bce = torch.zeros(1).to(self.device)

        visual_feats = self.vision_backbone(x)
        preds = self.classifier(visual_feats)

        if labels is not None:
            lossvalue_bce = self.bce_loss(preds, labels)

        return preds, lossvalue_bce, f'bce:\t {lossvalue_bce.item():0.4f}'


    def forward_ranking201(self, x, labels=None, class_index=None, epoch=0, n_crops=0, bs=16):

        loss_rank = torch.zeros(1).to(self.device)
        loss_allignment_cos = torch.zeros(1).to(self.device)
        loss_allignment_cos_fantasy = torch.zeros(1).to(self.device)
        loss_mapping_consistency = torch.zeros(1).to(self.device)

        x = self.bcn(x)
        visual_feats = self.vision_backbone(x)
        text_feats = self.textual_embeddings
        if class_index!=None:
            text_feats = text_feats[class_index, :]

        visual_feats = self.fc_v(visual_feats)
        text_feats = self.fc_t(text_feats)

        if labels is not None and self.training:
            w_visual = self.sim_score(visual_feats, visual_feats)

            visual_feats_norm = F.normalize(visual_feats, p=2, dim=-1, eps=1e-12)
            w_visual = F.normalize(w_visual, p=2, dim=-1, eps=1e-12)
            visual_diag = torch.diag_embed(torch.diag(w_visual))
            w_visual -= visual_diag
            visual_feats_fantasy = 3 * torch.matmul(w_visual, visual_feats_norm)
            w_text = self.sim_score(text_feats, text_feats)
            text_feats_norm = F.normalize(text_feats, p=2, dim=-1, eps=1e-12)
            w_text = F.normalize(w_text, p=2, dim=-1, eps=1e-12)
            text_diag = torch.diag_embed(torch.diag(w_text))
            w_text -= text_diag
            text_feats_fantasy = 3 * torch.matmul(w_text, text_feats_norm)

            visual_feats_fantasy = self.beta * visual_feats + (1 - self.beta) * visual_feats_fantasy
            text_feats_fantasy = self.beta * text_feats + (1 - self.beta) * text_feats_fantasy

            ranks = self.sim_score(visual_feats, text_feats)
        else:
            ranks = self.sim_score(visual_feats, text_feats)

        if labels is not None and self.training:

            ranks_f = self.sim_score(visual_feats_fantasy, text_feats_fantasy)

            loss_allignment_cos_fantasy = self.cross_loss_2(ranks_f[:, self.class_ids_loaded],
                                                            labels[:, self.class_ids_loaded], self.device)
        if n_crops > 0:
            ranks = ranks.view(bs, n_crops, -1).mean(1)

        if labels is not None:
            loss_rank = self.ranking_loss(ranks, labels, self.class_ids_loaded, self.device)
        if labels is not None and self.training:
            loss_allignment_cos = self.cross_loss(ranks[:, self.class_ids_loaded], labels[:, self.class_ids_loaded], self.device)
        loss_rank = 1 * loss_rank
        loss_allignment_cos = (0.01 * loss_allignment_cos)
        loss_allignment_cos_fantasy = (0.01 * loss_allignment_cos_fantasy)

        losses = loss_rank + loss_allignment_cos + loss_allignment_cos_fantasy

        if self.batchi % 14 == 0 and self.training:
            losses_p = round(losses.item(), 3)
            loss_rank_p = round(loss_rank.item(), 2)
            loss_allignment_cos_fantasy_p = round(loss_allignment_cos_fantasy.item(), 2)
            loss_mapping_consistency_p = round(loss_mapping_consistency.item(), 2)
            loss_allignment_cos_p = round(loss_allignment_cos.item(), 2)
            print(f'losses : {losses_p} loss_rank : {loss_rank_p}   loss_allignment_cos_fantasy : {loss_allignment_cos_fantasy_p} loss_allignment_cos: {loss_allignment_cos_p} loss_mapping_consistency:{loss_mapping_consistency_p}')


        self.batchi += 1
        if self.args.visual:
            visual_feats_fantasy = visual_feats_fantasy.view(bs, n_crops, -1).mean(1)
            visual_feats = visual_feats.view(bs, n_crops, -1).mean(1)
            return ranks, losses, visual_feats, visual_feats_fantasy
        else:
            return ranks, losses

    def compute_align_loss(self, visual_feats, text_feats):
        logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        device = visual_feats.device
        logits_per_image = logit_scale * visual_feats @ text_feats.T
        logits_per_text = logit_scale * visual_feats @ text_feats.T

        num_logits = logits_per_image.shape[0]
        labels = torch.eye(num_logits, device=device, dtype=torch.float)
        pred_1 = F.log_softmax(logits_per_image, dim=-1)
        pred_2 = F.log_softmax(logits_per_text, dim=-1)
        loss_a = F.kl_div(pred_1, labels, reduction='sum') / num_logits
        loss_b = F.kl_div(pred_2, labels, reduction='sum') / num_logits
        total_loss = (loss_a + loss_b) / 2
        return total_loss

    def sim_score(self, a, b):
        a_norm = a / a.norm(dim=1)[:, None]
        b_norm = b / (1e-6 + b.norm(dim=1))[:, None]
        score = (torch.mm(a_norm, b_norm.t()))
        return score

    def compute_image_similarity(self,x):
        flatten_x = x.view(x.size(0), -1)
        similarity_matrix = F.cosine_similarity(flatten_x.unsqueeze(1), flatten_x.unsqueeze(0), dim=2)

        return similarity_matrix
    def map_visual_text(self, visual_feats, labels, labels_embd):

        mapped_labels_embd = []
        labels == 1
        for i in range(0, labels.shape[0]):
            class_embd = labels_embd[labels[i] == 1].mean(dim=0)[None, :]
            mapped_labels_embd.append(class_embd)
        mapped_labels_embd = torch.cat(mapped_labels_embd)

        return visual_feats.detach(), mapped_labels_embd.detach()

    def maskAandB(self, a, b, mask_ratio):
        num_to_replace = int(mask_ratio * a.shape[0] * b.shape[1])
        replace_idx = torch.randperm(a.shape[0] * a.shape[1])[:num_to_replace]
        a.view(-1)[replace_idx] = 0
        return a

    def compute_prob_dist_matrix(self, vectors):
        maen = vectors.mean()
        num_vectors, vector_dim = vectors.size()
        prob_dist_matrix = torch.zeros(vector_dim, vector_dim)

        for i in range(vector_dim):
            for j in range(vector_dim):
                count = torch.sum((vectors[:, i] >= maen) & (vectors[:, j] >= maen))
                prob_dist_matrix[i, j] = count / num_vectors

        return prob_dist_matrix


