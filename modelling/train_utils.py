import torch
from torch import nn


def weighting_maker(targets, target_type, ignore_value):

    if target_type == 'ignore':
        weighting = (targets[:, 0:1] != ignore_value).type(torch.float)
    else:
        raise ValueError('Unknown target type: {}'.format(target_type))
    return weighting


def ce(logits, targets):
    loss = -targets * torch.log_softmax(logits, dim=1)
    loss = loss.sum(dim=1, keepdim=True)
    return loss


def normalize(loss, weight):
    weight_sum = weight.sum(dim=0)
    num_weight = torch.clamp(weight_sum, 1.)
    loss = (loss * weight).sum(dim=0) / num_weight
    return loss


def b_accu(binary_cls_logits, binary_cls_targets, is_logits=True):
    return (b_pred(binary_cls_logits, is_logits=is_logits) == binary_cls_targets).type(torch.float)


def b_pred(binary_cls_logits, is_logits=True):
    if is_logits:
        return (binary_cls_logits > 0).type(torch.float)
    else:
        return (binary_cls_logits > 0.5).type(torch.float)


def accu(logits, targets):
    max_value, max_cls = pred(logits)
    _, target_cls = targets.max(dim=1, keepdim=True)
    return (max_cls == target_cls).type(torch.float)


def pred(logits):
    max_value, max_cls = logits.max(dim=1, keepdim=True)
    return max_value, max_cls.float()


def target_maker(targets, attn_targets, config):

    end = 0
    cls_targets = list()
    for t_name in config['cls_tasks']:
        start = end
        end += config['cls_tasks'][t_name]
        cls_targets.append(targets[:, start:end])

    end = 0
    attn_targets_out = list()
    for t_name in config['attn_tasks']:
        start = end
        end += config['attn_tasks'][t_name]['num_channels']
        attn_targets_out.append(attn_targets[:, start:end])

    return cls_targets, attn_targets_out


def loss_maker(cls_logits, attn_logits, cls_targets, attn_targets, config):

    loss_cls = list()
    for l, t in zip(cls_logits, cls_targets):
        ls = ce(l, t)
        ls = normalize(ls, weighting_maker(t, target_type='ignore', ignore_value=config['ignore_value']))
        loss_cls.append(ls)

    loss_attn = list()
    version = config['attn_loss_ver']
    for l, t in zip(attn_logits, attn_targets):
        if version == 1:
            ll = torch.softmax(l.reshape(l.shape[0], l.shape[1], -1), dim=2).view(-1, 1)
            tt = nn.functional.interpolate(t, l.shape[2:], mode='nearest').repeat(1, l.shape[1], 1, 1)
            tt = tt.view(-1, 1)
            ls = - tt * torch.log(ll.clamp(min=1e-10))
            ls = normalize(ls, weighting_maker(tt, target_type='ignore', ignore_value=config['ignore_value']))
        loss_attn.append(ls*config['attn_loss_scalar'])
    
    losses = dict()
    tag = 'loss_'
    for c_idx, c_name in enumerate(config['cls_tasks']):
        k = tag + c_name[:min(4, len(c_name))]
        losses[k] = loss_cls[c_idx]

    for c_idx, c_name in enumerate(config['attn_tasks']):
        k = tag + c_name[:min(4, len(c_name))]
        losses[k] = loss_attn[c_idx]

    return losses


def accuracy_maker(cls_logits, attn_logits, cls_targets, attn_targets, config):

    accu_cls = list()
    num_valid_cases_cls = list()
    for l, t in zip(cls_logits, cls_targets):
        au = accu(l, t)
        vc = (t[:, 0:1] != config['ignore_value']).type(torch.float)
        num_valid_cases_cls.append(vc.sum(dim=0))

        au = normalize(au, vc)
        accu_cls.append(au)

    accu_attn = list()
    num_valid_cases_attn = list()
    for l, t in zip(attn_logits, attn_targets):
        t = nn.functional.interpolate(t, l.shape[2:], mode='nearest').repeat(1, l.shape[1], 1, 1)
        l = l.view(-1, 1)
        t = t.view(-1, 1)
        au = b_accu(l, t)
        vc = (t != config['ignore_value']).type(torch.float)
        num_valid_cases_attn.append(vc.sum(dim=0))

        au = normalize(au, vc)
        accu_attn.append(au)
        
    accus = dict()
    num_valid_cases = dict()
    tag = 'accu_'
    tag_vc = 'nvac_'

    for c_idx, c_name in enumerate(config['cls_tasks']):
        k = tag + c_name[:min(4, len(c_name))]
        accus[k] = accu_cls[c_idx]

        k_vc = tag_vc + c_name[:min(4, len(c_name))]
        num_valid_cases[k_vc] = num_valid_cases_cls[c_idx]

    for c_idx, c_name in enumerate(config['attn_tasks']):
        k = tag + c_name[:min(4, len(c_name))]
        accus[k] = accu_attn[c_idx]

        k_vc = tag_vc + c_name[:min(4, len(c_name))]
        num_valid_cases[k_vc] = num_valid_cases_attn[c_idx]

    return accus, num_valid_cases


def prediction_maker(cls_logits, cls_targets,
                     config, is_logits=True):

    pred_cls = list()
    prob_cls = cls_logits
    for l in cls_logits:
        _, p = pred(l)
        pred_cls.append(p)

    preds = dict()
    probs = dict()
    targets = dict()
    tag = 'pred_'
    for c_idx, c_name in enumerate(config['cls_tasks']):
        k = tag + c_name[:min(4, len(c_name))]
        preds[k] = pred_cls[c_idx]

    tag = 'prob_'
    for c_idx, c_name in enumerate(config['cls_tasks']):
        k = tag + c_name[:min(4, len(c_name))]
        probs[k] = prob_cls[c_idx]

    tag = 'tar_'
    for c_idx, c_name in enumerate(config['cls_tasks']):
        k = tag + c_name[:min(4, len(c_name))]
        max_v, max_idx = cls_targets[c_idx].max(dim=1, keepdim=True)
        max_idx[max_v == config['ignore_value']] = config['ignore_value']
        targets[k] = max_idx.float()

    return preds, probs, targets


def map_maker(cls_maps, config):

    tag = 'map_'
    maps = dict()
    for c_idx, c_name in enumerate(config['cls_tasks']):
        k = tag + c_name[:min(4, len(c_name))]
        maps[k] = cls_maps[c_idx]

    return maps
