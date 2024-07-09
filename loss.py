"""
Author: Nasir Hayat (nasirhayat6160@gmail.com)
Date: June 10, 2020
"""

from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from numpy import linalg as LA
from torch.nn.functional import kl_div, softmax, log_softmax


class ClipLoss(nn.Module):
    def __init__(
            self,
            local_loss=False,
            gather_with_grad=False,
            cache_labels=False,
            rank=0,
            world_size=1,
            use_horovod=False,
    ):
        super().__init__()
        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size
        self.use_horovod = use_horovod

        
        self.prev_num_logits = 0
        self.labels = {}

    def forward(self, image_features, text_features):  
        logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))  
        device = image_features.device
        if self.world_size > 1:
            all_image_features, all_text_features = gather_features(
                image_features, text_features,
                self.local_loss, self.gather_with_grad, self.rank, self.world_size, self.use_horovod)

            if self.local_loss:
                logits_per_image = logit_scale * image_features @ all_text_features.T
                logits_per_text = logit_scale * text_features @ all_image_features.T
            else:
                logits_per_image = logit_scale * all_image_features @ all_text_features.T
                logits_per_text = logits_per_image.T
        else:
            logits_per_image = logit_scale * image_features @ text_features.T  
            logits_per_text = logit_scale * text_features @ image_features.T  

        
        num_logits = logits_per_image.shape[0]  
        labels = torch.eye(num_logits, device=device, dtype=torch.float)  
        pred_1 = F.log_softmax(logits_per_image, dim=-1)  
        pred_2 = F.log_softmax(logits_per_text, dim=-1)  
        loss_a = F.kl_div(pred_1, labels, reduction='sum') / num_logits  
        loss_b = F.kl_div(pred_2, labels, reduction='sum') / num_logits
        total_loss = (loss_a + loss_b) / 2
        return total_loss

class KLDivLoss(nn.Module):
    def __init__(self, temperature=0.2):
        super(KLDivLoss, self).__init__()

        self.temperature = temperature

    def forward(self, emb1, emb2):
        emb1 = softmax(emb1 / self.temperature, dim=1).detach()
        emb2 = log_softmax(emb2 / self.temperature, dim=1)
        loss_kldiv = kl_div(emb2, emb1, reduction='none')
        loss_kldiv = torch.sum(loss_kldiv, dim=1)
        loss_kldiv = torch.mean(loss_kldiv)
        return loss_kldiv


class RankingLoss(nn.Module):
    def __init__(self, neg_penalty=0.03):
        super(RankingLoss, self).__init__()

        self.neg_penalty = neg_penalty

    def forward(self, ranks, labels, class_ids_loaded, device):
        '''
        for each correct it should be higher then the absence
        '''
        labels = labels[:, class_ids_loaded]
        ranks_loaded = ranks[:, class_ids_loaded]
        neg_labels = 1 + (labels * -1)
        loss_rank = torch.zeros(1).to(device)
        label_len = len(labels)
        for i in range(len(labels)):
            if torch.all(labels[i] == 0):
                label_len -= 1
                continue
            else:
                correct = ranks_loaded[i, labels[i] == 1]
                wrong = ranks_loaded[i, neg_labels[i] == 1]
                correct = correct.reshape((-1, 1)).repeat((1, len(wrong)))
                wrong = wrong.repeat(len(correct)).reshape(len(correct), -1)
                image_level_penalty = ((self.neg_penalty + wrong) - correct)
                image_level_penalty[image_level_penalty < 0] = 0
                loss_rank += image_level_penalty.sum()
        loss_rank /= label_len

        return loss_rank

class CosineLoss(nn.Module):

    def forward(self, t_emb, v_emb):
        a_norm = v_emb / v_emb.norm(dim=1)[:, None]
        b_norm = t_emb / t_emb.norm(dim=1)[:, None]
        
        loss = 1 - torch.mean(torch.diagonal(torch.mm(a_norm, b_norm.t()), 0))
        return loss

class CrossEntropyLoss(nn.Module):
    def forward(self, pred,batch_label):
        log_softmax_func = nn.LogSoftmax(dim=0)

        Prob = pred / pred.norm(dim=1)[:, None]  
        BatC = batch_label / batch_label.norm(dim=1)[:, None]  

        loss = torch.diagonal(torch.mm(Prob, BatC.t())) / 0.2
        loss = - log_softmax_func(loss)
        loss = torch.mean(loss)
        return loss

class NewCrossEntropyLoss(nn.Module):
    def forward(self, score, batch_label, device):
        
        p_score = torch.mul(score, batch_label)
        labels_c = torch.where(batch_label == 0, torch.tensor(1).to(device), torch.tensor(0).to(device))
        n_score = torch.mul(score, labels_c)
        n_score_sum = torch.exp(n_score/0.2).sum(dim=1).view(-1, 1)
        n_score_div_exp = 1/ (n_score_sum)
        
        fen = torch.mul(torch.exp(p_score/0.2), n_score_div_exp)
        count = torch.count_nonzero(p_score, dim=1).view(-1,1)
        
        p_score_mean = 1 - torch.abs(p_score - p_score / count)
        fen = torch.mul(fen, p_score_mean)
        fen[p_score == 0] = 1
        
        fen = - torch.log(fen) 
        
        
        loss = 1 / batch_label.shape[0] * fen.sum()
        return loss

    def map_visual_text(self, visual_feats, labels, labels_embd):
        
        mapped_labels_embd = []
        labels == 1  
        for i in range(0, labels.shape[0]):  
            class_embd = labels_embd[labels[i] == 1].mean(dim=0)[None, :]  
            mapped_labels_embd.append(class_embd)  
        mapped_labels_embd = torch.cat(mapped_labels_embd)  
        
        return visual_feats.detach(), mapped_labels_embd.detach()
    
    def sim_score(self, a, b):
        a_norm = a / a.norm(dim=1)[:, None]  
        
        b_norm = b / (1e-6 + b.norm(dim=1))[:, None]  
        score = (torch.mm(a_norm, b_norm.t()))  
        return score  
class NewCrossEntropyLoss3(nn.Module):
    def forward(self, score_f_f, score_f_o, batch_label, device):
        

        p_score = torch.mul(score_f_f, batch_label)
        labels_c = torch.where(batch_label == 0, torch.tensor(1).to(device), torch.tensor(0).to(device))
        n_score = torch.mul(score_f_o, labels_c)
        n_score_sum = torch.exp(n_score).sum(dim=1).view(-1, 1)
        n_score_div_exp = 1/(n_score_sum)
        
        fen = torch.mul(torch.exp(p_score), n_score_div_exp)
        count = torch.count_nonzero(p_score, dim=1).view(-1,1)

        fen[p_score == 0]=1
        
        fen = - torch.log(fen) 
        
        loss = 1 / batch_label.shape[0] * fen.sum()
        return loss

    def map_visual_text(self, visual_feats, labels, labels_embd):
        
        mapped_labels_embd = []
        labels == 1  
        for i in range(0, labels.shape[0]):  
            class_embd = labels_embd[labels[i] == 1].mean(dim=0)[None, :]  
            mapped_labels_embd.append(class_embd)  
        mapped_labels_embd = torch.cat(mapped_labels_embd)  
        
        return visual_feats.detach(), mapped_labels_embd.detach()
    
    def sim_score(self, a, b):
        a_norm = a / a.norm(dim=1)[:, None]  
        
        b_norm = b / (1e-6 + b.norm(dim=1))[:, None]  
        score = (torch.mm(a_norm, b_norm.t()))  
        return score  
class SampleCrossEntropyLoss(nn.Module):
    
    def forward(self, visual_feats, batch_label, class_is_loaded, device):
        
        loss = torch.zeros(1).to(device)

        for i in class_is_loaded:
            p_visual = visual_feats[batch_label[:, i] == 1]
            
            if p_visual.shape[0]!=0:
                fen_zi = self.sim_score(p_visual, p_visual)[0,1:]
                if fen_zi.shape[0]>1:
                    loss += torch.abs(fen_zi-torch.mean(fen_zi)).sum()

        loss = 1 / batch_label.shape[0] * loss
        
        return loss

    def map_visual_text(self, visual_feats, labels, labels_embd):
        
        mapped_labels_embd = []
        labels == 1  
        for i in range(0, labels.shape[0]):  
            class_embd = labels_embd[labels[i] == 1].mean(dim=0)[None, :]  
            mapped_labels_embd.append(class_embd)  
        mapped_labels_embd = torch.cat(mapped_labels_embd)  
        
        return visual_feats.detach(), mapped_labels_embd.detach()
    
    def sim_score(self, a, b):
        a_norm = a / a.norm(dim=1)[:, None]  
        
        b_norm = b / (1e-6 + b.norm(dim=1))[:, None]  
        score = (torch.mm(a_norm, b_norm.t()))  
        return score  
class NewCrossEntropyLoss2(nn.Module):
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    def forward(self, visual_feats, text_feats, batch_label, class_ids_loaded, d_space, device):
        loss_align_one = torch.zeros(1).to(device)
        loss_align_outer = torch.zeros(1).to(device)
        for i in range(0, batch_label.shape[0]):  
            extract = batch_label[i]
            
            
            
            
            
            
            
            

            zero_indices = [j for j in range(len(extract)) if extract[j] == 0]
            one_indices = [j for j in range(len(extract)) if extract[j] != 0]
            numeric_n = list(set(zero_indices) & set(class_ids_loaded))
            numeric_p = list(set(one_indices) & set(class_ids_loaded))
            negative_text_feats = text_feats[numeric_n]
            positive_text_feats = text_feats[numeric_p]
            
            
            
            
            
            
            
            
            ith_visual_feats = visual_feats[i, :].view(1, d_space)
            one_negative_sample = self.sim_score(ith_visual_feats, negative_text_feats)  
            one_positive_sample = self.sim_score(ith_visual_feats, positive_text_feats)  
            
            fen_zi = torch.exp(one_positive_sample)  
            fen_mu = torch.exp(one_negative_sample).sum(1).view(1, -1)  
            if fen_zi.shape[1] > 1:
                
                
                
                
                log_input = fen_zi / fen_mu[:, 0].view(-1, 1)
                loss_align_outer = -torch.log(log_input)
                loss_align_one += (1 / fen_zi.shape[1]) * torch.sum(loss_align_outer, dim=1)
                
            else:
                
                loss_align_one += -torch.log(fen_zi[0] / (fen_mu[0]))
        loss_align = loss_align_one / batch_label.shape[0]
        return loss_align
    
    def sim_score(self, a, b):
        a_norm = a / a.norm(dim=1)[:, None]  
        
        
        b_norm = b / (1e-6 +b.norm(dim=1))[:, None]  
        score = (torch.mm(a_norm, b_norm.t()))  
        return score  















































class ContrastiveLoss(nn.Module):
    def __init__(self, batch_size, device='cuda:0', temperature=0.2):
        super().__init__()
        self.batch_size = batch_size
        self.register_buffer("temperature", torch.tensor(temperature).to(device))  
        self.register_buffer("negatives_mask", (
            torch.eye(batch_size * 2, batch_size * 2, dtype=bool).to(device)).float())  

    def forward(self, emb_i, emb_j):  
        z_i = F.normalize(emb_i, dim=1)  
        z_j = F.normalize(emb_j, dim=1)  

        representations = torch.cat([z_i, z_j], dim=0)  
        similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0),
                                                dim=2)  

        sim_ij = torch.diag(similarity_matrix, self.batch_size)  
        sim_ji = torch.diag(similarity_matrix, -self.batch_size)  
        positives = torch.cat([sim_ij, sim_ji], dim=0)  

        nominator = torch.exp(positives / self.temperature)  
        denominator = self.negatives_mask * torch.exp(similarity_matrix / self.temperature)  

        loss_partial = -torch.log(nominator / torch.sum(denominator, dim=1))  
        loss = torch.sum(loss_partial) / (2 * self.batch_size)
        return loss


class SimCLR(nn.Module):
    def __init__(self, temp=0.2):
        super().__init__()

        self.temp = temp

    def contrastive_loss(self, q, k):
        
        q = nn.functional.normalize(q, dim=1)
        k = nn.functional.normalize(k, dim=1)

        
        k = concat_all_gather(k)
        logits = torch.einsum('nc,mc->nm', [q, k]) / self.temp
        N = logits.shape[0]

        labels = (torch.arange(N, dtype=torch.long) + N * torch.distributed.get_rank()).cuda()
        return nn.CrossEntropyLoss()(logits, labels) * (2 * self.temp)

    def forward(self, student_output, teacher_output, epoch):
        student_out = student_output
        student_out = student_out.chunk(2)

        teacher_out = teacher_output
        teacher_out = teacher_out.detach().chunk(2)

        return self.contrastive_loss(student_out[0], teacher_out[1]) + self.contrastive_loss(student_out[1],
                                                                                             teacher_out[0])


@torch.no_grad()
def concat_all_gather(tensor):
    tensors_gather = [torch.ones_like(tensor)
                      for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output

"""
Author: Yonglong Tian (yonglong@mit.edu)
Date: May 07, 2020
"""

class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        
        mask = mask.repeat(anchor_count, contrast_count)
        
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss