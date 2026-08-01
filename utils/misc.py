import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


def soft_update(target, source, tau, return_diff=False):
    avg_diff = 0.0
    count = 0
    with torch.no_grad():
        for target_param, param in zip(target.parameters(), source.parameters()):
            if return_diff:
                avg_diff += torch.norm(target_param.data - param.data, p=2).item()
                count += 1
            target_param.data.lerp_(param.data, tau)
    if return_diff:
        return avg_diff / max(count, 1)
    return None


def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


def onehot_from_logits(logits, eps=0.0):
    argmax_acs = (logits == logits.max(1, keepdim=True)[0]).float()
    if eps == 0.0:
        return argmax_acs
    rand_acs = Variable(torch.eye(logits.shape[1], device=logits.device)[[
        np.random.choice(range(logits.shape[1]), size=logits.shape[0])
    ]], requires_grad=False)
    return torch.stack([
        argmax_acs[i] if r > eps else rand_acs[i]
        for i, r in enumerate(torch.rand(logits.shape[0], device=logits.device))
    ])


def sample_gumbel(shape, eps=1e-20, tens_type=torch.FloatTensor):
    U = Variable(tens_type(*shape).uniform_(), requires_grad=False)
    return -torch.log(-torch.log(U + eps) + eps)


def gumbel_softmax_sample(logits, temperature):
    y = logits + sample_gumbel(logits.shape, tens_type=type(logits.data))
    return F.softmax(y / temperature, dim=1)


def gumbel_softmax(logits, temperature=1.0, hard=False):
    y = gumbel_softmax_sample(logits, temperature)
    if hard:
        y_hard = onehot_from_logits(y)
        y = (y_hard - y).detach() + y
    return y
