import torch
import logging
import numpy as np
from torch.utils.data import Subset
from utils.privacy_randomized import RandomizedLabelPrivacy


def compute_randomized_labels_classification(train_loader, estimate_loader, mechanism, prior, num_classes, seed, eps, eps1, sigma, delta):
    '''choose a fitable mechanism, randomize labels'''

    num_same = 0
    x_sets_l = []
    y_sets_l = []
    y_tilde_sets_l = []
    eps_tensor = torch.tensor(eps)

    generator = torch.Generator().manual_seed(seed)
    
    if mechanism == 'rrwithweight': 

        # compute probability of labels
        prior = compute_distribution_with_Laplace(estimate_loader, prior, num_classes, eps1)
        prior = prior.view(-1) 

        # Setting of metrix V
        e = torch.exp(torch.tensor(-1.0 / sigma))
        v_weight = torch.full((num_classes, num_classes), e)
        v_weight.fill_diagonal_(1.0)
        dp_weight = v_weight * prior.view(-1, 1)
        logging.info(f"prior probability: {prior}")
        
        # compute S1_tilde,S2_tilde
        idx_max_value = torch.argmax(dp_weight, dim=0)
        S1 = torch.where(idx_max_value == torch.arange(num_classes))[0]
        S2 = torch.where(idx_max_value != torch.arange(num_classes))[0]
        S1_tilde, S2_tilde = S1, S2
        delta = min(delta, len(S1))
        _, Delta = torch.topk(torch.diagonal(dp_weight), delta)

    elif mechanism == 'rrwithprior':

        # compute probability of labels
        prior = compute_distribution_with_Laplace(estimate_loader, prior, num_classes, eps1)
        # prior = compute_distribution_with_Laplace_infty(estimate_loader, prior, num_classes)
        prior = prior.view(-1) 
        logging.info(f"prior probability:, {prior}")

        # compute the k* of rrwithprior
        idx_sort = torch.argsort(prior, descending=True)
        prior_sorted = prior[idx_sort]
        logging.info(f"sorted prior: {prior_sorted}")

        nusum = torch.arange(len(prior))
        wks = torch.cumsum(prior_sorted, dim=0) / (1 + nusum / torch.exp(eps_tensor))
        optim_k = torch.argmax(wks) + 1
        logging.info(f"objective value: {wks}")

        # compute S1_tilde,S2_tilde
        S1, S2 = idx_sort[:optim_k], idx_sort[optim_k:]
        S1_tilde, S2_tilde = S1, torch.tensor([])
        Delta = S1_tilde
    
    elif mechanism == 'rr':
        # compute S1_tilde,S2_tilde
        S1, S2 = torch.arange(num_classes), torch.tensor([])
        S1_tilde, S2_tilde = torch.arange(num_classes), torch.tensor([])
        Delta = S2_tilde

    # randomize labels
    if isinstance(S1, torch.Tensor):
        S1 = set(S1.flatten().tolist())
    elif isinstance(S1, set):
        pass
    else:
        raise TypeError("S1_tilde must be Tensor or set")

    if isinstance(S2, torch.Tensor):
        S2 = set(S2.flatten().tolist())
    elif isinstance(S2, set):
        pass
    else:
        raise TypeError("S2_tilde must be Tensor or set")
    
    if isinstance(S1_tilde, torch.Tensor):
        S1_tilde = set(S1_tilde.flatten().tolist())
    elif isinstance(S1_tilde, set):
        pass
    else:
        raise TypeError("S1_tilde must be Tensor or set")

    if isinstance(S2_tilde, torch.Tensor):
        S2_tilde = set(S2_tilde.flatten().tolist())
    elif isinstance(S2_tilde, set):
        pass
    else:
        raise TypeError("S2_tilde must be Tensor or set")
    
    if isinstance(Delta, torch.Tensor):
        Delta = set(Delta.flatten().tolist())
    elif isinstance(Delta, set):
        pass
    else:
        raise TypeError("Delta must be Tensor or set")

    
    l1, l2 = len(S1_tilde), len(S2_tilde)
    delta1 = len(Delta)

    tmp = (torch.exp(eps_tensor)+l1-1)*(torch.exp(eps_tensor)+l2-1)-(l1-delta1)*l2
    beta = ((torch.exp(eps_tensor)-1)+delta1*l2/(l1+l2)) / tmp
    gamma = ((torch.exp(eps_tensor)+delta1-1)-delta1*(torch.exp(eps_tensor)+l1-1)/(l1+l2)) / tmp
    sum = (torch.exp(eps_tensor) + l1 -1)*beta + l2*gamma

    num_per_cls = torch.zeros(num_classes)
    num_per_same = torch.zeros(num_classes)
    for i, (image, label) in enumerate(train_loader):
        for i in range(num_classes):
            num_per_cls[i] += ((label == i).sum())
        y_tildes = torch.zeros(len(label), dtype=torch.long)
        for idx in range(len(label)):
            y = label[idx].item()
            proba = torch.zeros(num_classes, dtype=torch.float32)

            if y in S1:
                for j in S1_tilde | S2_tilde:
                    if j in S1_tilde:
                        if j == y:
                            proba[j] = torch.exp(eps_tensor) * beta
                        else:
                            proba[j] = beta
                    elif j in S2_tilde:
                        proba[j] = gamma
            else:
                for j in S1_tilde | S2_tilde:
                    if j in Delta:
                        proba[j] = 1 / (l1 + l2)
                    elif j in S1_tilde - Delta:
                        proba[j] = beta
                    elif j in S2_tilde:
                        if j ==  y:
                            proba[j] = torch.exp(eps_tensor) * gamma
                        else:
                            proba[j] = gamma

            proba = proba / proba.sum()  # Normalize probabilities
            sampled = torch.multinomial(proba, num_samples=1, generator=generator)
            y_tildes[idx] = sampled
            if y == sampled:
                num_same += 1
                num_per_same[y] += 1

        x_sets_l.append(image)
        y_sets_l.append(label)
        y_tilde_sets_l.append(y_tildes)
    logging.info(f"S1_tilde: {S1_tilde} | Delta: {Delta}")
    logging.info(f"number of original label equalling randomized label: {num_same}")
    x_sets_l = torch.cat(x_sets_l, dim=0)
    y_sets_l = torch.cat(y_sets_l, dim=0)
    y_tilde_sets_l = torch.cat(y_tilde_sets_l, dim=0)

    return x_sets_l, y_sets_l, y_tilde_sets_l

def compute_randomized_labels_classification_infty(train_loader, estimate_loader, mechanism, prior, num_classes, seed, eps, eps1, sigma, delta):
    '''choose a fitable mechanism, randomize labels'''

    num_same = 0
    x_sets_l = []
    y_sets_l = []
    y_tilde_sets_l = []

    generator = torch.Generator().manual_seed(seed)
    
    if mechanism == 'rrwithweight': 

        # compute probability of labels
        prior = compute_distribution_with_Laplace_infty(estimate_loader, prior, num_classes)
        prior = prior.view(-1) 

        # Setting of metrix V
        e = torch.exp(torch.tensor(-1.0 / sigma))
        v_weight = torch.full((num_classes, num_classes), e)
        v_weight.fill_diagonal_(1.0)
        dp_weight = v_weight * prior.view(-1, 1)
        logging.info(f"prior probability: {prior}")
        
        # compute S1_tilde,S2_tilde
        _, Delta = torch.topk(torch.diagonal(dp_weight), delta)
        idx_max_value = torch.argmax(dp_weight, dim=0)
        S1 = torch.where(idx_max_value == torch.arange(num_classes))[0]
        S2 = torch.where(idx_max_value != torch.arange(num_classes))[0]
        S1_tilde, S2_tilde = S1, S2
    
    elif mechanism == 'rr' or 'rrwithprior':
        # compute S1_tilde,S2_tilde
        S1, S2 = torch.arange(num_classes), torch.tensor([])
        S1_tilde, S2_tilde = torch.arange(num_classes), torch.tensor([])
        Delta = S2_tilde

    # randomize labels
    if isinstance(S1, torch.Tensor):
        S1 = set(S1.flatten().tolist())
    elif isinstance(S1, set):
        pass
    else:
        raise TypeError("S1_tilde must be Tensor or set")

    if isinstance(S2, torch.Tensor):
        S2 = set(S2.flatten().tolist())
    elif isinstance(S2, set):
        pass
    else:
        raise TypeError("S2_tilde must be Tensor or set")
    
    if isinstance(S1_tilde, torch.Tensor):
        S1_tilde = set(S1_tilde.flatten().tolist())
    elif isinstance(S1_tilde, set):
        pass
    else:
        raise TypeError("S1_tilde must be Tensor or set")

    if isinstance(S2_tilde, torch.Tensor):
        S2_tilde = set(S2_tilde.flatten().tolist())
    elif isinstance(S2_tilde, set):
        pass
    else:
        raise TypeError("S2_tilde must be Tensor or set")
    
    if isinstance(Delta, torch.Tensor):
        Delta = set(Delta.flatten().tolist())
    elif isinstance(Delta, set):
        pass
    else:
        raise TypeError("Delta must be Tensor or set")

    
    l1, l2 = len(S1_tilde), len(S2_tilde)

    e_beta = 1
    e_gamma = 1 - delta / (l1 + l2)

    for i, (image, label) in enumerate(train_loader):
        y_tildes = torch.zeros(len(label), dtype=torch.long)
        for idx in range(len(label)):
            y = label[idx].item()
            proba = torch.zeros(num_classes, dtype=torch.float32)

            if y in S1:
                for j in S1_tilde | S2_tilde:
                    if j in S1_tilde:
                        if j == y:
                            proba[j] = e_beta
                        else:
                            proba[j] = 0
                    elif j in S2_tilde:
                        proba[j] = 0
            else:
                for j in S1_tilde | S2_tilde:
                    if j in Delta:
                        proba[j] = 1 / (l1 + l2)
                    elif j in S1_tilde - Delta:
                        proba[j] = 0
                    elif j in S2_tilde:
                        if j ==  y:
                            proba[j] = e_gamma
                        else:
                            proba[j] = 0

            proba = proba / proba.sum()  # Normalize probabilities
            sampled = torch.multinomial(proba, num_samples=1, generator=generator)
            y_tildes[idx] = sampled
            if y == sampled:
                num_same += 1

        x_sets_l.append(image)
        y_sets_l.append(label)
        y_tilde_sets_l.append(y_tildes)
    logging.info(f"S1_tilde: {S1_tilde} | Delta: {Delta}")
    logging.info(f"number of original label equalling randomized label: {num_same}")
    x_sets_l = torch.cat(x_sets_l, dim=0)
    y_sets_l = torch.cat(y_sets_l, dim=0)
    y_tilde_sets_l = torch.cat(y_tilde_sets_l, dim=0)

    return x_sets_l, y_sets_l, y_tilde_sets_l

def compute_distribution_with_Laplace(estimate_loader, prior, num_classes, eps1):
    if prior is not None:
        prob = torch.from_numpy(prior)
    else:
        y_sets_l = []
        for i, (x, y) in enumerate(estimate_loader):
            y_sets_l.append(y)
        y_sets_l = torch.cat(y_sets_l, dim=0)    

        prob = torch.zeros(num_classes,1)

        for class_idx in range(num_classes):
            print(torch.sum(y_sets_l == class_idx))
            prob[class_idx] = torch.sum(y_sets_l == class_idx)
            
        sensitivity = 2 #torch.max(prob) - torch.min(prob)

        prob = torch.max(prob + torch.distributions.laplace.Laplace(0.0, sensitivity/eps1).sample((num_classes,1)), torch.zeros(num_classes,1))

        prob = prob / torch.sum(prob)

    return prob

def compute_distribution_with_Laplace_infty(estimate_loader, prior, num_classes):
    if prior is not None:
        prob = torch.from_numpy(prior)
    else:
        y_sets_l = []
        for i, (x, y) in enumerate(estimate_loader):
            y_sets_l.append(y)
        y_sets_l = torch.cat(y_sets_l, dim=0)    

        prob = torch.zeros(num_classes,1)

        for class_idx in range(num_classes):
            prob[class_idx] = torch.sum(y_sets_l == class_idx)

        prob = prob / torch.sum(prob)

    return prob
