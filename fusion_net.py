from Models.fundus_swin_network import build_model as fundus_build_model
from Models.unetr import UNETR_base_3DNet
import torch
import torch.nn as nn
import ot
import torch.nn.functional as F
import torch
import torch.nn as nn
import ot
import numpy as np
import torch
import ot
import os


class PoE(nn.Module):
    def __init__(self, modality_num=2, sample_num=50, seed=1):
        super(PoE, self).__init__()

        self.sample_num = sample_num
        self.seed = seed

        phi = torch.ones(modality_num, requires_grad=True)
        self.phi = torch.nn.Parameter(phi)

    def forward(self, mu_list, var_list, eps=1e-8):
        t_sum = 0
        mu_t_sum = 0

        alpha = F.softmax(self.phi, dim=0)

        for idx, (mu, var) in enumerate(zip(mu_list, var_list)):
            T = 1 / (var + eps)

            t_sum += alpha[idx] * T
            mu_t_sum += mu * alpha[idx] * T

        mu = mu_t_sum / t_sum
        var = 1 / t_sum

        dim = mu.shape[1]
        batch_size = mu.shape[0]

        eps = self.gaussian_noise(samples=(batch_size, self.sample_num), k=dim,
                                  seed=self.seed)  # eps torch.Size([8, 50, 2])
        eps = eps.unsqueeze(dim=-1).repeat(1, 1, 1, 128)

        # poe_features = torch.unsqueeze(mu, dim=1) + torch.unsqueeze(var, dim=1) * eps   # # torch.unsqueeze(var, dim=1 torch.Size([8, 1, 2, 256])

        poe_features = torch.unsqueeze(mu, dim=1) + torch.unsqueeze(var, dim=1)

        return poe_features

    def gaussian_noise(self, samples, k, seed):
        # works with integers as well as tuples
        if self.training:
            return torch.normal(torch.zeros(*samples, k), torch.ones(*samples, k)).cuda()
        else:
            return torch.normal(torch.zeros(*samples, k), torch.ones(*samples, k),
                                generator=torch.manual_seed(seed)).cuda()  # must be the same seed as the training seed


class EPRL(nn.Module):
    def __init__(self,
                 x_dim,
                 z_dim=256,
                 beta=1e-2,
                 sample_num=50,
                 topk=1,
                 num_classes=3,
                 seed=1,
                 batch_size=16):
        super(EPRL, self).__init__()

        self.beta = beta
        self.sample_num = sample_num
        self.topk = 1
        self.num_classes = num_classes
        self.seed = seed
        self.z_dim = z_dim

        self.encoder = nn.Sequential(
            nn.Linear(x_dim, z_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(z_dim * 2, z_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(z_dim * 2, z_dim),
        )
        self.batch_size = batch_size
        # Decoder: simple logistic regression as in the paper
        self.decoder_logits = nn.Linear(z_dim, num_classes)

        self.mlp_2d = nn.Sequential(nn.ReLU(), nn.Linear(144, num_classes),nn.Dropout(0.2), nn.ReLU())
        self.mlp_3d = nn.Sequential(nn.ReLU(), nn.Linear(216, num_classes), nn.Dropout(0.2), nn.ReLU())

        # Design proxies for histology images
        self.proxies = nn.Parameter(torch.empty([num_classes, z_dim * 2]))  # torch.Size([4, 512])
        torch.nn.init.xavier_uniform_(self.proxies, gain=1.0)
        self.proxies_dict = {"0": 0, "1": 1}

        self.alpha = nn.Parameter(torch.tensor(0.5))

    def gaussian_noise(self, samples, K, seed):
        if self.training:
            return torch.normal(torch.zeros(*samples, K), torch.ones(*samples, K)).cuda()
        else:
            return torch.normal(torch.zeros(*samples, K), torch.ones(*samples, K),
                                generator=torch.manual_seed(seed)).cuda()

    def encoder_result(self, x):
        encoder_output = self.encoder(x)
        return encoder_output

    def encoder_proxies(self):
        mu_proxy = self.proxies[:, :self.z_dim]
        sigma_proxy = torch.nn.functional.softplus(self.proxies[:, self.z_dim:])
        return mu_proxy, sigma_proxy

    def estimate_v(self, z_proxy, epsilon=1e-8):
        var = torch.var(z_proxy, dim=1, unbiased=False)
        v_initial = 2 * var / (var - 1 + epsilon)
        v = torch.clamp(v_initial, min=2)
        return v

    def entropy_regularization(self, logits):
        p = torch.softmax(logits, dim=1)
        log_p = torch.log_softmax(logits, dim=1)
        entropy = -torch.sum(p * log_p, dim=1)
        return entropy.mean()

    def forward(self, x, y=None):
        z = self.encoder_result(x)  # torch.Size([8, 256])

        # Get mu and sigma from proxies
        mu_proxy, sigma_proxy = self.encoder_proxies()  # mu_proxy torch.Size([2, 256])

        # Sample
        eps_proxy = self.gaussian_noise(samples=([self.num_classes, self.sample_num]), K=self.z_dim,
                                        seed=self.seed)

        z_proxy_sample = mu_proxy.unsqueeze(dim=1) + sigma_proxy.unsqueeze(
            dim=1) * eps_proxy

        z_proxy = z_proxy_sample

        # Get attention maps
        z_norm = F.normalize(z, dim=1)  # torch.Size([16, 256])
        z_proxy_norm = F.normalize(z_proxy)  # torch.Size([3, 50, 256])

        if not self.training:
            threshold =0.5
            # Generate pseudo-labels based on highest attention scores
            z_proxy_norm_expanded = z_proxy_norm.unsqueeze(0).expand(1, -1, -1, -1)  # [16, 3, 50, 256]

            att = torch.matmul(z_norm.unsqueeze(1), torch.transpose(z_proxy_norm_expanded, 2, 3))  # [16, 3, 144, 50]
            att = att.permute(0, 2, 1, 3)  # [16, 144, 3, 50]
            att = att.mean(dim=1)


            att_mean = torch.mean(att, dim=2)
            z_mean = torch.mean(z_norm, dim=2)


            pseudo_labels_att = torch.softmax(att_mean, dim=1)
            pseudo_labels_feat = torch.softmax(z_mean, dim=1)
            if pseudo_labels_feat.shape[1] == 144:
                pseudo_labels_feat = self.mlp_2d(pseudo_labels_feat)
            else :
                pseudo_labels_feat = self.mlp_3d(pseudo_labels_feat)

            pseudo_labels_combined = self.alpha * pseudo_labels_att + (1- self.alpha) * pseudo_labels_feat



            confidence, labels = torch.max(pseudo_labels_combined, dim=1)
            mask = confidence > threshold


            if mask.sum().item() == 0:
                mask[confidence.argmax()] = True

            filtered_labels = labels[mask]

            proxy_indices = [self.proxies_dict[str(int(label_item))] for label_item in filtered_labels]
            proxy_indices = torch.tensor(proxy_indices).long().cuda()


            mask = torch.zeros(att.size(0), att.size(1), dtype=torch.bool).cuda()
            mask[torch.arange(att.size(0)), proxy_indices] = True

            att_positive = torch.masked_select(att, mask.unsqueeze(-1)).view(att.size(0), -1)
            att_negative = torch.masked_select(att, ~mask.unsqueeze(-1)).view(att.size(0), -1)

            # print('att_positive', att_positive.shape)
            # print('att_negative', att_negative.shape)

            self_topk = 100
            att_topk_positive, _ = torch.topk(att_positive, self_topk, dim=1)
            att_topk_negative, _ = torch.topk(att_negative, self_topk, dim=1)

            att_positive_mean = torch.mean(att_topk_positive, dim=1)
            att_negative_mean = torch.mean(att_topk_negative, dim=1)

            proxy_loss = torch.mean(torch.exp(-att_positive_mean + att_negative_mean))

            entropy_loss = self.entropy_regularization(pseudo_labels_combined)

            mu_proxy_repeat = mu_proxy.repeat(x.shape[0], 1, 1)
            sigma_proxy_repeat = sigma_proxy.repeat(x.shape[0], 1, 1)
            mu_topk = mu_proxy_repeat
            sigma_topk = sigma_proxy_repeat

            z_topk = z


            return mu_topk, sigma_topk, proxy_loss, z_topk, entropy_loss

        else:
            z_proxy_norm_expanded = z_proxy_norm.unsqueeze(0).expand(self.batch_size, -1, -1, -1)  # [16, 3, 50, 256]

            att = torch.matmul(z_norm.unsqueeze(1), torch.transpose(z_proxy_norm_expanded, 2, 3))  # [16, 3, 144, 50]
            att = att.permute(0, 2, 1, 3)  # [16, 144, 3, 50]
            att = att.mean(dim=1)

            proxy_indices = [self.proxies_dict[str(int(y_item))] for y_item in y]
            proxy_indices = torch.tensor(proxy_indices).long().cuda()

            mask = torch.zeros(att.size(0), att.size(1), dtype=torch.bool).cuda()
            mask[torch.arange(att.size(0)), proxy_indices] = True

            att_positive = torch.masked_select(att, mask.unsqueeze(-1)).view(att.size(0), -1)
            att_negative = torch.masked_select(att, ~mask.unsqueeze(-1)).view(att.size(0), -1)

            self_topk = 100
            att_topk_positive, _ = torch.topk(att_positive, self_topk, dim=1)
            att_topk_negative, _ = torch.topk(att_negative, self_topk, dim=1)

            att_positive_mean = torch.mean(att_topk_positive, dim=1)
            att_negative_mean = torch.mean(att_topk_negative, dim=1)

            proxy_loss = torch.mean(torch.exp(-att_positive_mean + att_negative_mean))

        # Gather mu_proxy and sigma_proxy for each sample
        mu_proxy_repeat = mu_proxy.repeat(x.shape[0], 1, 1)  # torch.Size([batch-size, 2, 256])
        sigma_proxy_repeat = sigma_proxy.repeat(x.shape[0], 1, 1)

        mu_topk = mu_proxy_repeat
        sigma_topk = sigma_proxy_repeat
        att_unbind = torch.cat(torch.unbind(att, dim=2), dim=1)

        z_topk = z

        return mu_topk, sigma_topk, proxy_loss, z_topk


class MIAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super(MIAttention, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)

        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MIAttention_fusion(nn.Module):
    def __init__(self, dim, dim_oct, dim_general, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.,
                 proj_drop=0., dropout=0.1):
        super(MIAttention_fusion, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv_fundus = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.qkv_oct = nn.Linear(dim_oct, dim_oct * 3, bias=qkv_bias)
        self.qkv_general = nn.Linear(dim_general, dim_general * 3, bias=qkv_bias)
        self.norm = nn.LayerNorm(128)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_fundus = nn.Linear(dim, dim)
        self.proj_oct = nn.Linear(dim_oct, dim)
        self.proj_general = nn.Linear(288, 128)
        self.fc = nn.Sequential(nn.ReLU(), nn.Linear(32, 288), nn.ReLU())
        self.proj_drop = nn.Dropout(proj_drop)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_2d, x_3d, x_global):
        if x_2d.dim() == 2:
            x_2d = x_2d.unsqueeze(1)

        if x_3d.dim() == 2:
            x_3d = x_3d.unsqueeze(1)

        B_2d, N_2d, C_2d = x_2d.shape
        B_3d, N_3d, C_3d = x_3d.shape
        B_gen, N_gen, C_gen = x_global.shape

        qkv_2d = self.qkv_fundus(x_2d).reshape(B_2d, N_2d, 3, self.num_heads, C_2d // self.num_heads).permute(2, 0, 3,
                                                                                                              1, 4)
        q_2d, k_2d, v_2d = qkv_2d[0], qkv_2d[1], qkv_2d[2]

        qkv_3d = self.qkv_oct(x_3d).reshape(B_3d, N_3d, 3, self.num_heads, C_3d // self.num_heads).permute(2, 0, 3, 1,
                                                                                                           4)
        q_3d, k_3d, v_3d = qkv_3d[0], qkv_3d[1], qkv_3d[2]

        qkv_general = self.qkv_general(x_global).reshape(B_gen, N_gen, 3, self.num_heads,
                                                         C_gen // self.num_heads).permute(2, 0, 3, 1, 4)
        q_general, k_general, v_general = qkv_general[0], qkv_general[1], qkv_general[2]

        q_general = self.fc(q_general)

        attn_global = (q_general @ torch.cat((k_general, k_3d, k_2d), dim=3).transpose(-2, -1)) * self.scale
        attn_global = attn_global.softmax(dim=-1)
        attn_global = self.attn_drop(attn_global)

        attn_global_x = (attn_global @ torch.cat((v_general, v_2d, v_3d), dim=3)).transpose(1, 2)

        attn_global_x = self.proj_general(attn_global_x)
        x_global = self.norm(self.dropout(attn_global_x))

        return x_global


class SelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(SelfAttention, self).__init__()
        self.attention = MIAttention(embed_dim, num_heads)

    def forward(self, x):
        return self.attention(x)


class SelfAttention_fusion(nn.Module):
    def __init__(self, embed_dim, dim_oct, dim_general, num_heads):
        super(SelfAttention_fusion, self).__init__()
        self.attention = MIAttention_fusion(embed_dim, dim_oct, dim_general, num_heads)

    def forward(self, x, y, z):
        return self.attention(x, y, z)


class CrossAttention(nn.Module):
    def __init__(self, embed_dim, embed_dim_3d, num_heads):
        super(CrossAttention, self).__init__()
        self.attention_2d = MIAttention(embed_dim, num_heads)
        self.attention_3d = MIAttention(embed_dim_3d, num_heads)
        self.fc = nn.Sequential(nn.ReLU(), nn.Linear(768, 1024), nn.ReLU())

    def forward(self, query, key_value):
        att_3d = self.fc(self.attention_3d(key_value))
        out = self.attention_2d(query) + att_3d
        return out


class CrossAttention1(nn.Module):
    def __init__(self, embed_dim, embed_dim_3d, num_heads):
        super(CrossAttention1, self).__init__()
        self.attention_2d = MIAttention(embed_dim, num_heads)
        self.attention_3d = MIAttention(embed_dim_3d, num_heads)
        self.fc = nn.Sequential(nn.ReLU(), nn.Linear(768, 1024), nn.ReLU())

    def forward(self, query, key_value):
        attn_3d = self.fc(self.attention_3d(query))
        out = attn_3d + self.attention_2d(key_value)
        return out


def KL_between_normals(q_distr, p_distr):
    mu_q, sigma_q = q_distr
    mu_p, sigma_p = p_distr
    k = mu_q.size(1)

    mu_diff = mu_p - mu_q
    mu_diff_sq = torch.mul(mu_diff, mu_diff)
    logdet_sigma_q = torch.sum(2 * torch.log(torch.clamp(sigma_q, min=1e-8)), dim=1)
    logdet_sigma_p = torch.sum(2 * torch.log(torch.clamp(sigma_p, min=1e-8)), dim=1)

    fs = torch.sum(torch.div(sigma_q ** 2, sigma_p ** 2), dim=1) + torch.sum(torch.div(mu_diff_sq, sigma_p ** 2), dim=1)
    two_kl = fs - k + logdet_sigma_p - logdet_sigma_q
    return two_kl * 0.5


class PID(nn.Module):
    def __init__(self, embed_dim, embed_dim_3d, dim_general, num_heads, dropout=0.1, dim_pid=256):
        super(PID, self).__init__()

        self.self_attn1 = SelfAttention(embed_dim_3d, num_heads)
        self.self_attn = SelfAttention(embed_dim, num_heads)
        self.fc = nn.Sequential(nn.ReLU(), nn.Linear(768, 1024), nn.ReLU())
        self.dropout = nn.Dropout(dropout)
        # self.norm = nn.LayerNorm(embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
        self.norm1 = nn.LayerNorm(embed_dim_3d)
        self.avgpool = nn.AdaptiveAvgPool1d(1)

        self.general = SelfAttention_fusion(embed_dim, embed_dim, dim_general, num_heads)

    def forward(self, x_2d, x_3d, distribution):

        x_2d_attn = self.self_attn(x_2d)
        x_3d_attn = self.self_attn1(x_3d)
        x_3d_attn = self.fc(x_3d_attn)
        # print('x_2d_out_att ', x_2d_attn.shape)  # x_3d_out torch.Size([8, 1, 1024])
        # print('x_3d_out_att', x_3d_attn.shape)  # x_3d_out torch.Size([8, 1, 1024])

        # global_out = self.general(x_2d, x_3d, distribution)

        x_2d_combined = self.norm(self.dropout(x_2d_attn))
        x_3d_combined = self.norm(self.dropout(x_3d_attn))

        x_3d_combined = self.avgpool(x_3d_attn.transpose(1, 2))
        x_2d_combined = self.avgpool(x_2d_attn.transpose(1, 2))

        # print('x_3d_combined', x_3d_combined.shape)  # torch.Size([16, 256 ,1])
        # print('x_2d_combined', x_2d_combined.shape)  # torch.Size([16, 256 ,1])

        return x_2d_combined, x_3d_combined


import matplotlib.pyplot as plt
from scipy.stats import t as StudentT


def visualize_student_t_distributions(mu_pos, sigma_pos, v_pos, mu_neg, sigma_neg, v_neg, title, filename):
    num_distributions = len(mu_pos)
    num_cols = 4
    num_rows = (num_distributions + num_cols - 1) // num_cols
    x = np.linspace(-0.1, 0.1, 1000)

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(20, 12))
    axes = axes.flatten()

    for i in range(num_distributions):
        # print('v_pos[i]', v_pos[i])
        # print('mu_pos[i]', mu_pos[i])
        # print('sigma_pos[i]', sigma_pos[i])
        y_pos = StudentT.pdf(x, df=v_pos[i], loc=mu_pos[i], scale=sigma_pos[i])
        y_neg = StudentT.pdf(x, df=v_neg[i], loc=mu_neg[i], scale=sigma_neg[i])
        axes[i].plot(x, y_pos, label=f'Positive (v={v_pos[i]:.8f}, loc={mu_pos[i]:.8f}, scale={sigma_pos[i]:.8f})',
                     color='blue')
        axes[i].plot(x, y_neg, label=f'Negative (v={v_neg[i]:.8f}, loc={mu_neg[i]:.8f}, scale={sigma_neg[i]:.8f})',
                     color='red')
        axes[i].set_title(f'Sample {i + 1}')
        axes[i].set_xlabel('x')
        axes[i].set_ylabel('Probability Density')
        axes[i].legend()
        axes[i].grid(True)


    for i in range(num_distributions, num_rows * num_cols):
        fig.delaxes(axes[i])

    fig.suptitle(title)
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)

    plt.savefig(filename, format='pdf')


class MIEstimator(nn.Module):
    def __init__(self, dim=128):
        super(MIEstimator, self).__init__()
        self.dim = dim
        self.mimin_glob = CLUBMean(self.dim * 2, self.dim)  # Can also use CLUBEstimator, but CLUBMean is more stable
        self.mimin = CLUBMean(self.dim, self.dim)

    def forward(self, histology, pathways, global_embed):
        mimin = self.mimin(histology, pathways)
        mimin += self.mimin_glob(torch.cat((histology, pathways), dim=1), global_embed)
        return mimin

    def learning_loss(self, histology, pathways, global_embed):
        mimin_loss = self.mimin.learning_loss(histology, pathways)
        mimin_loss += self.mimin_glob.learning_loss(torch.cat((histology, pathways), dim=1), global_embed).mean()

        return mimin_loss


class CLUBMean(nn.Module):  # Set variance of q(y|x) to 1, logvar = 0. Update 11/26/2022
    def __init__(self, x_dim, y_dim, hidden_size=512):
        # p_mu outputs mean of q(Y|X)
        # print("create CLUB with dim {}, {}, hiddensize {}".format(x_dim, y_dim, hidden_size))

        super(CLUBMean, self).__init__()

        if hidden_size is None:
            self.p_mu = nn.Linear(x_dim, y_dim)
        else:
            self.p_mu = nn.Sequential(nn.Linear(x_dim, int(hidden_size)),
                                      nn.ReLU(),
                                      nn.Linear(int(hidden_size), y_dim))

    def get_mu_logvar(self, x_samples):
        # variance is set to 1, which means logvar=0
        mu = self.p_mu(x_samples)
        return mu, 0

    def forward(self, x_samples, y_samples):

        mu, logvar = self.get_mu_logvar(x_samples)

        # log of conditional probability of positive sample pairs
        positive = - (mu - y_samples) ** 2 / 2.

        prediction_1 = mu.unsqueeze(1)  # shape [nsample,1,dim]
        y_samples_1 = y_samples.unsqueeze(0)  # shape [1,nsample,dim]

        # log of conditional probability of negative sample pairs
        negative = - ((y_samples_1 - prediction_1) ** 2).mean(dim=1) / 2.

        return (positive.sum(dim=-1) - negative.sum(dim=-1)).mean()

    def loglikeli(self, x_samples, y_samples):  # unnormalized loglikelihood
        mu, logvar = self.get_mu_logvar(x_samples)
        # print("mu size:", mu.shape)
        # print("y_samples size:", y_samples.shape)
        return (-(mu - y_samples) ** 2).sum(dim=1).mean(dim=0)

    def learning_loss(self, x_samples, y_samples):
        return - self.loglikeli(x_samples, y_samples)

def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

class AttentionModel(nn.Module):
    def __init__(self, embed_size, num_heads, num_layers):
        super(AttentionModel, self).__init__()

        # Multihead Attention layer
        self.attn = nn.MultiheadAttention(embed_size, num_heads,batch_first=True)




        self.layer_norm = nn.LayerNorm(embed_size)

        self.ffn = nn.Sequential(
            nn.Linear(embed_size, embed_size * 3),
            nn.ReLU(),
            nn.Linear(embed_size * 3, embed_size)
        )
        self.relu = nn.ReLU()

    def forward(self, x,y,z):

        attn_output, attn_weights = self.attn(x, y, z)
        attn_output = x + attn_output
        attn_output = self.layer_norm(attn_output)

        output =  attn_output + self.ffn(attn_output)
        output = self.relu(output)

        return output

class DILR(nn.Module):
    def __init__(self, args,common_ratio = 0.5):
        super().__init__()
        self.args = args
        self.common_ratio = common_ratio
        # backbone

        # deformable attention
        """if args.rda:
            from .dat.dat_blocks import DAttentionBaseline

            self.da1_l3 = DAttentionBaseline(
                q_size=(14,14), kv_size=(14,14), n_heads=8, n_head_channels=128, n_groups=4,
                attn_drop=0, proj_drop=0, stride=2,
                offset_range_factor=-1, use_pe=True, dwc_pe=False,
                no_off=False, fixed_pe=False, ksize=5, log_cpb=False
            )

            self.da1_l4 = DAttentionBaseline(
                q_size=(7,7), kv_size=(7,7), n_heads=16, n_head_channels=128, n_groups=8,
                attn_drop=0, proj_drop=0, stride=1,
                offset_range_factor=-1, use_pe=True, dwc_pe=False,
                no_off=False, fixed_pe=False, ksize=3, log_cpb=False
            )

            self.da2_l3 = DAttentionBaseline(
                q_size=(14,14), kv_size=(14,14), n_heads=8, n_head_channels=128, n_groups=4,
                attn_drop=0, proj_drop=0, stride=2,
                offset_range_factor=-1, use_pe=True, dwc_pe=False,
                no_off=False, fixed_pe=False, ksize=5, log_cpb=False
            )

            self.da2_l4 = DAttentionBaseline(
                q_size=(7,7), kv_size=(7,7), n_heads=16, n_head_channels=128, n_groups=8,
                attn_drop=0, proj_drop=0, stride=1,
                offset_range_factor=-1, use_pe=True, dwc_pe=False,
                no_off=False, fixed_pe=False, ksize=3, log_cpb=False
            )"""

        """# projector
        if args.backbone == 'resnet50':
            sizes = [2048] + list(map(int, args.projector.split('-')))
        elif args.backbone == 'resnet18':
            sizes = [512] + list(map(int, args.projector.split('-')))
        elif args.backbone == 'vits16':
            sizes = [384] + list(map(int, args.projector.split('-')))
        elif 'mit' in args.backbone:
            sizes = [2048] + list(map(int, args.projector.split('-')))"""

        """layers = []
        for i in range(len(sizes) - 2):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=False))
            layers.append(nn.BatchNorm1d(sizes[i + 1]))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(sizes[-2], sizes[-1], bias=False))"""
        self.projector1 = nn.Linear(1024,2048)
        self.projector2 = nn.Linear(768,2048)
        self.self_attn1 = AttentionModel(1024,8,1)
        self.self_attn2 = AttentionModel(1024,8,1)
        self.common_dim = 2048 * self.common_ratio

        self.shared_features_projector = nn.Linear(1024,int(2048 * common_ratio) )
        self.guided_features_projector1 = nn.Linear(1024,int(2048 * common_ratio) )
        self.guided_features_projector2 = nn.Linear(1024,int(2048 * common_ratio) )

        self.cross_attn1 = AttentionModel(1024,8,1)
        self.cross_attn2 = AttentionModel(1024,8,1)


        #self.self_attn1 = nn.selfAttentionModel(1024,8,1)
        #self.self_attn2 = nn.selfAttentionModel(1024,8,1)

        # normalization layer for the representations z1 and z2
        self.bn1 = nn.BatchNorm1d(2048, affine=False)
        self.bn2 = nn.BatchNorm1d(2048, affine=False)

    def bt_loss_cross(self, z1, z2,common_dim):
        # empirical cross-correlation matrix
        c = self.bn1(z1).T @ self.bn2(z2)

        # sum the cross-correlation matrix between all gpus
        c.div_(self.args.batch_size*4)


        dim_c = int(common_dim)
        c_c = c[:dim_c,:dim_c]
        c_u = c[dim_c:,dim_c:]

        on_diag_c = torch.diagonal(c_c).add_(-1).pow_(2).sum()
        off_diag_c = off_diagonal(c_c).pow_(2).sum()

        on_diag_u = torch.diagonal(c_u).pow_(2).sum()
        off_diag_u = off_diagonal(c_u).pow_(2).sum()

        loss_c = on_diag_c + 0.0051 * off_diag_c
        loss_u = on_diag_u + 0.0051 * off_diag_u

        return loss_c,on_diag_c,off_diag_c,loss_u,on_diag_u,off_diag_u


    def bt_loss_single(self, z1, z2):
        # empirical cross-correlation matrix
        c = self.bn(z1).T @ self.bn(z2)

        # sum the cross-correlation matrix between all gpus
        c.div_(self.args.batch_size*4)
        torch.distributed.all_reduce(c)

        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(c).pow_(2).sum()
        loss = on_diag + self.args.lambd * off_diag
        return loss,on_diag,off_diag

    def forward_resnet_da(self, x, backbone, da_l3, da_l4):
        x = backbone.conv1(x)
        x = backbone.bn1(x)
        x = backbone.relu(x)
        x = backbone.maxpool(x)

        x = backbone.layer1(x)
        x = backbone.layer2(x)
        x = backbone.layer3(x)
        x1,pos1,ref1 = da_l3(x)
        x = x + x1
        x = backbone.layer4(x)
        x2,pos2,ref2 = da_l4(x)
        x = x + x2

        x = backbone.avgpool(x)
        x = torch.flatten(x, 1)
        x = backbone.fc(x)

        return x

    def forward(self, y1_2, y2_1, shared_features,funds_guided,octs_guided):
        # Project input features
        y1 = self.projector1(y1_2)  # Shape: [batch, seq_len1, feature_dim]
        y2 = self.projector2(y2_1)  # Shape: [batch, seq_len2, feature_dim]

        # Calculate dimensions for common and unique parts
        feature_dim = y1.size(2)
        common_dim = int(self.common_ratio * feature_dim)
        unique_dim = feature_dim - common_dim

        # Split features into common and unique parts
        y1_unique_part = y1[:, :, :common_dim]
        y1_common_part = y1[:, :, common_dim:]
        y2_unique_part = y2[:, :, :common_dim]
        y2_common_part = y2[:, :, common_dim:]

        funds_guided = self.guided_features_projector1(funds_guided)
        octs_guided = self.guided_features_projector2(octs_guided)
        # Process unique parts with self-attention
        y1_uni = self.self_attn1(funds_guided, y1_unique_part, y1_unique_part)
        y2_uni = self.self_attn2(octs_guided, y2_unique_part, y2_unique_part)

        # Aggregate sequence dimension
        y1_uni = torch.mean(y1_uni, dim=1)  # Shape: [batch, common_dim]
        y2_uni = torch.mean(y2_uni, dim=1)  # Shape: [batch, common_dim]

        # Process common parts with cross-attention using shared features
        shared_features_projected = self.shared_features_projector(shared_features).unsqueeze(1)
        y1_common = self.cross_attn1(shared_features_projected, y1_common_part, y1_common_part).squeeze(1)  # Shape: [batch, unique_dim]
        y2_common = self.cross_attn2(shared_features_projected, y2_common_part, y2_common_part).squeeze(1)  # Shape: [batch, unique_dim]

        # Concatenate common and unique parts for each modality
        y1 = torch.cat((y1_common, y1_uni), dim=1)  # Shape: [batch, feature_dim]
        y2 = torch.cat((y2_common, y2_uni), dim=1)  # Shape: [batch, feature_dim]

        # Calculate loss based on current common dimension ratio
        common_dim_out = int(self.common_ratio * y1.size(1))
        loss12_c, on_diag12_c, off_diag12_c, loss12_u, on_diag12_u, off_diag12_u = self.bt_loss_cross(
            y1, y2, common_dim=common_dim_out
        )
        loss12 = (loss12_c + loss12_u) / 2.0

        # Normalize features
        y1 = self.bn1(y1)
        y2 = self.bn2(y2)

        # Combine features for output
        # Ensure dimensions are consistent by explicitly using common_dim_out
        combined_features = torch.cat((
            y1[:, common_dim_out:],         # Unique features from y1
            y1_common + y2_common,          # Shared common features
            y2[:, common_dim_out:],         # Unique features from y2
        ), dim=1)

        return combined_features, loss12

class MedFusion(nn.Module):
    def __init__(self, classes, modalties, classifiers_dims, args):
        """
        :param classes: Number of classification categories
        :param views: Number of modalties
        :param classifier_dims: Dimension of the classifier
        :param annealing_epoch: KL divergence annealing epoch during training
        """
        super(MedFusion, self).__init__()
        self.modalties = modalties
        self.classes = classes
        self.mode = args.mode
        dropout = 0.25
        self.fundus_embedding_dim = 1024
        self.oct_embedding_dim = 768
        self.dim_general = 256

        self.num_classes = 2
        self.topk_fundus = 1
        self.topk_oct = 1
        self.sample_num = 800
        # self.sample_num = 50
        self.seed = 1
        self.head = 8

        # ---- 2D Transformer Backbone ----
        self.transformer_2DNet = fundus_build_model()  # SWIN-Transformer

        # ---- 3D Transformer Backbone ----
        self.transformer_3DNet = UNETR_base_3DNet(num_classes=self.classes)

        self.fc_fundus = nn.Sequential(nn.ReLU(), nn.Linear(512, 1024), nn.ReLU())

        # ---Evidential
        self.fc = nn.Sequential(nn.ReLU(), nn.Linear(3072, 64), nn.ReLU(),
                                nn.Linear(64, self.classes))
        self.logit_fc = nn.Sequential(nn.ReLU(), nn.Linear(256, 64), nn.ReLU(),
                                    nn.Linear(64, self.classes))

        fundus_encoder_layer = nn.TransformerEncoderLayer(d_model=768, nhead=8, dim_feedforward=512, dropout=dropout,
                                                          activation='relu')
        self.fundus_transformer = nn.TransformerEncoder(fundus_encoder_layer, num_layers=2)

        oct_encoder_layer = nn.TransformerEncoderLayer(d_model=768, nhead=8, dim_feedforward=512, dropout=dropout,
                                                       activation='relu')
        self.oct_transformer = nn.TransformerEncoder(oct_encoder_layer, num_layers=2)

        self.EPRL_fundus = EPRL(self.fundus_embedding_dim, num_classes=self.num_classes,
                              topk=self.topk_fundus, sample_num=self.sample_num, seed=self.seed, batch_size=args.batch_size)

        self.EPRL_oct = EPRL(self.oct_embedding_dim, num_classes=self.num_classes,
                           topk=self.topk_oct, sample_num=self.sample_num, seed=self.seed,batch_size=args.batch_size)

        self.PoE = PoE(modality_num=2, sample_num=800, seed=1)

        self.PID = PID(self.fundus_embedding_dim, self.oct_embedding_dim, self.dim_general, self.head, dropout=0.1,
                       dim_pid=256)

        self.ce_loss = nn.CrossEntropyLoss()

        self.avgpool = nn.AdaptiveAvgPool1d(1)

        self.CLUB = MIEstimator(self.fundus_embedding_dim)

        self.DILR = DILR(args,common_ratio=0.5)

        self.args = args

    def get_KL_loss(self, mu, std):
        '''
        :param mu: [batch_size, dimZ]
        :param std: [batch_size, dimZ]
        :return:
        '''
        # KL divergence between prior and posterior
        prior_z_distr = torch.zeros_like(mu), torch.ones_like(std)
        encoder_z_distr = mu, std

        I_zx_bound = torch.mean(KL_between_normals(encoder_z_distr, prior_z_distr))

        return torch.mean(I_zx_bound)

    def visualize_and_save_distributions(self, mu_mean_positive, sigma_mean_positive, v_mean_positive,
                                         mu_mean_negative, sigma_mean_negative, v_mean_negative, epoch):

        output_dir = 'students_t_distributions/'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)


        filename = os.path.join(output_dir, f'students_t_distributions_epoch_{epoch + 1}.pdf')


        visualize_student_t_distributions(
            mu_mean_positive, sigma_mean_positive, v_mean_positive,
            mu_mean_negative, sigma_mean_negative, v_mean_negative,
            f'Epoch {epoch + 1} Student\'s t Distributions (Positive and Negative)',
            filename
        )

    def compute_loss_test(self, loss1, IB_loss_proxy, proxy_loss_fundus, proxy_loss_oct, mimin_loss, entropy_loss):

        # loss = loss1 + IB_loss_proxy + (proxy_loss_fundus + proxy_loss_oct) * 0.8 + 0.001 * mimin_loss + entropy_loss* 0.001
        loss = loss1 + IB_loss_proxy + (proxy_loss_fundus + proxy_loss_oct) * 0.8 + 0.001 * mimin_loss
        return loss

    def compute_loss_train(self, loss1, IB_loss_proxy, proxy_loss_fundus, proxy_loss_oct, mimin_loss):

        loss = loss1 + IB_loss_proxy + (proxy_loss_fundus + proxy_loss_oct) * 0.3 + 0.001 * mimin_loss
        return loss



    def forward(self, X, y, epoch):
        x, fundus_out = self.transformer_2DNet(X[0])
        x1, oct_out = self.transformer_3DNet(X[1]) # shape [32, 144, 1024]



        #print(f"x shape: {x.shape}")

        # mu_topk_fundus, sigma_topk_fundus, proxy_loss_fundus, z_topk_fundus, v_fundus = self.EPRL_fundus(fundus_out,
        #                                                                                                 y=y)  # mu_top torch.Size([8, 2, 256])
        # mu_topk_oct, sigma_topk_oct, proxy_loss_oct, z_topk_oct, v_oct = self.EPRL_oct(oct_out, y=y)
        if not self.training:
            mu_topk_fundus, sigma_topk_fundus, proxy_loss_fundus, z_topk_fundus, entropy_loss = self.EPRL_fundus(x, y=y)  # mu_top torch.Size([8, 2, 256])
            mu_topk_oct, sigma_topk_oct, proxy_loss_oct, z_topk_oct, entropy_loss  = self.EPRL_oct(x1, y=y)
        else:
            mu_topk_fundus, sigma_topk_fundus, proxy_loss_fundus, z_topk_fundus = self.EPRL_fundus(x, y=y)  # mu_top torch.Size([8, 2, 256])
            mu_topk_oct, sigma_topk_oct, proxy_loss_oct, z_topk_oct = self.EPRL_oct(x1, y=y)

        mu_list = [mu_topk_fundus, mu_topk_oct]
        var_list = [sigma_topk_fundus, sigma_topk_oct]


        eps = self.gaussian_noise(samples=(16, self.sample_num), k=dim,
                                  seed=self.seed)  # eps torch.Size([8, 50, 2])
        fundus_guided = mu_topk_fundus + torch.rand_like(mu_topk_fundus) * sigma_topk_fundus


        oct_guided = mu_topk_oct + torch.rand_like(mu_topk_oct) * sigma_topk_oct

        poe_features = self.PoE(mu_list, var_list)  # poe_features torch.Size([8, 1, 2, 128])
        poe_embed = torch.mean(poe_features, dim=1)
        #print('poe_embed', poe_embed.shape) # shape [32, 2, 256])
        B, N, D = poe_embed.shape
        global_fusion = self.fc_fundus(poe_embed.reshape(B, -1)) # (32,1024)



        combine_features, loss_DILR = self.DILR(x, x1, global_fusion,fundus_guided, oct_guided)

        #mimin_loss = self.CLUB.learning_loss(x_fundus.squeeze(2), x_oct.squeeze(2), global_fusion)
        # print('loss', mimin_loss)

        #combine_features = torch.cat([x_fundus.squeeze(2), global_fusion, x_oct.squeeze(2)], 1)
        #test_features = torch.cat([x_fundus.squeeze(2), x_oct.squeeze(2)], 1)
        #print('combine_features', combine_features.shape) # shape [32, 3072]

        pred = self.fc(combine_features)
        pred = pred[:,:2]
        smoothing=0.1
        with torch.no_grad():

            true_dist = torch.zeros_like(pred).to(pred.device)
            true_dist.fill_(smoothing / (self.num_classes - 1))
            true_dist.scatter_(1, y.unsqueeze(1), 1.0 - smoothing)

        #loss1 = self.ce_loss(pred, y)
        loss1 = torch.sum(-true_dist * F.log_softmax(pred, dim=-1), dim=-1).mean()


        IB_loss_proxy = 0.01 * self.get_KL_loss(mu_topk_fundus, sigma_topk_fundus) + 0.01 * self.get_KL_loss(
            mu_topk_oct, sigma_topk_oct)

        if not self.training:
            loss = self.compute_loss_test(loss1, IB_loss_proxy, proxy_loss_fundus, proxy_loss_oct, loss_DILR, entropy_loss)
        else:
            loss = self.compute_loss_train(loss1, IB_loss_proxy, proxy_loss_fundus, proxy_loss_oct, loss_DILR)

        loss = torch.mean(loss)

        return pred, loss,combine_features


class twoD_transformer(nn.Module):
    def __init__(self, classes, modalties, classifiers_dims, args):
        """
        :param classes: Number of classification categories
        :param views: Number of modalties
        :param classifier_dims: Dimension of the classifier
        :param annealing_epoch: KL divergence annealing epoch during training
        """
        super(twoD_transformer, self).__init__()
        self.modalties = modalties
        self.classes = classes
        self.mode = args.mode
        # ---- 2D Transformer Backbone ----
        self.transformer_2DNet = fundus_build_model()  # SWIN-Transformer

        # ---Evidential
        self.fc = nn.Sequential(nn.ReLU(), nn.Linear(1024, 64), nn.ReLU(),
                                nn.Linear(64, self.classes))

        self.fc_fundus = nn.Sequential(nn.ReLU(), nn.Linear(1024, 768), nn.ReLU())

        self.ce_loss = nn.CrossEntropyLoss()

        self.args = args

    def forward(self, X, y):
        x, backboneout_1 = self.transformer_2DNet(X[0])
        backboneout_1 = self.fc_fundus(backboneout_1)

        pred = self.fc(backboneout_1)

        return pred, backboneout_1


class threeD_transformer(nn.Module):
    def __init__(self, classes, modalties, classifiers_dims, args):
        """
        :param classes: Number of classification categories
        :param views: Number of modalties
        :param classifier_dims: Dimension of the classifier
        :param annealing_epoch: KL divergence annealing epoch during training
        """
        super(threeD_transformer, self).__init__()
        self.modalties = modalties
        self.classes = classes
        self.mode = args.mode

        # ---- 3D Transformer Backbone ----
        # UNETR from MONAI
        self.transformer_3DNet = UNETR_base_3DNet(num_classes=self.classes)

        # ---Evidential
        self.fc = nn.Sequential(nn.ReLU(), nn.Linear(768, 64), nn.ReLU(),
                                nn.Linear(64, self.classes))

        self.args = args

    def forward(self, X, y):
        x, backboneout_2 = self.transformer_3DNet(X[1])

        pred = self.fc(backboneout_2)

        return pred, backboneout_2


