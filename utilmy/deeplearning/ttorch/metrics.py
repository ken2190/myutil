# -*- coding: utf-8 -*-

import numpy as np
import torch 


##############################################################################
def test_metrics():
    from utilmy.deeplearning.ttorch import metrics

    model  = torch.hub.load('pytorch/vision:v0.10.0', 'alexnet', pretrained = True)
    data   = torch.rand(64, 3, 224, 224)
    output = model(data)
    labels = torch.randint(1000, (64,))#random labels 
    acc    = metrics.torch_metric_accuracy(output = output, labels = labels) 



    # -*- coding: utf-8 -*-
    import torch
    from utilmy.deeplearning.ttorch import metrics

    x1 = torch.rand(100,)
    x2 = torch.rand(100,)
    r = metrics.torch_pearson_coeff(x1, x2)

    x = torch.rand(100, 30)
    r_pairs = metrics.torch_pearson_coeff_pairs(x)



def test2():
    from utilmy.deeplearning.ttorch import metrics
    model = torch.hub.load('pytorch/vision:v0.10.0', 'alexnet', pretrained = True)

    data = torch.rand(64, 3, 224, 224)
    output = model(data)
    # This is just an example where class coded by 999 has more occurences
    # No train test splits are applied to lead to the overrepresentation of class 999 
    p = [(1-0.05)/1000]*999
    p.append(1-sum(p))
    labels = np.random.choice(list(range(1000)), 
                            size = (10000,), 
                            p = p)#imbalanced 1000-class labels
    labels = torch.Tensor(labels).long()
    weight, label_weight = metrics.torch_class_weights(labels)
    loss = torch.nn.CrossEntropyLoss(weight = weight)
    l = loss(output, labels[:64])




##############################################################################
def torch_pearson_coeff(x1, x2):
    '''Computes pearson correlation coefficient between two 1D tensors
    with torch
    
    Input
    -----
    x1: 1D torch.Tensor of shape (N,)
    
    x2: 1D torch.Tensor of shape (N,)
    
    Output
    ------
    r: scalar pearson correllation coefficient 
    '''
    cos = torch.nn.CosineSimilarity(dim = 0, eps = 1e-6)
    r = cos(x1 - x1.mean(dim = 0, keepdim = True), 
            x2 - x2.mean(dim = 0, keepdim = True))
    
    return r


def torch_pearson_coeff_pairs(x): 
    '''Computes pearson correlation coefficient across 
    the 1st dimension of a 2D tensor  
    
    Input
    -----
    x: 2D torch.Tensor of shape (N,M)
       correlation coefficients will be computed between 
       all unique pairs across the first dimension
       x[1,M] x[2,M], ...x[i,M] x[j,M], for unique pairs (i,j)

    Output
    ------
    r: list of tuples such that r[n][0] scalar denoting the 
       pearson correllation coefficient of the pair of tensors with idx in 
       tuple r[n][1] 
    '''
    from itertools import combinations 
    all_un_pair_comb = [comb for comb in combinations(list(range(x.shape[0])), 2)]
    r = []
    for aupc in all_un_pair_comb:
        current_r = torch_pearson_coeff(x[aupc[0], :], x[aupc[1], :])    
        r.append((current_r, (aupc[0], aupc[1])))
    
    return r


def torch_metric_accuracy(output = None, labels = None):
    ''' Classification accuracy calculation as acc = (TP + TN) / nr total pred
    
    Input
    -----
    output: torch.Tensor of size (N,M) where N are the observations and 
        M the classes. Values must be such that highest values denote the 
        most probable class prediction.
    
    labels: torch.Tensor tensor of size (N,) of int denoting for each of the N
        observations the class that it belongs to, thus int must be in the 
        range 0 to M-1
    
    Output
    ------
    acc: float, accuracy of the predictions    
    '''
    _ , predicted = torch.max(output.data, 1)
    total = labels.size(0)
    correct = (predicted == labels).sum().item()
    acc = 100*(correct/total)

    return acc 


def torch_class_weights(labels):
    '''Compute class weights for imbalanced classes
    
    Input
    -----
    labels: torch.Tensor of shape (N,) of int ranging from 0,1,..C-1 where
        C is the number of classes
    
    Output
    ------
    weights: torch.Tensor of shape (C,) where C is the number of classes 
        with the weights of each class based on the occurence of each class
        NOTE: computed as weights_c = min(occurence) / occurence_c
        for class c
    
    labels_weights: dict, with keys the unique int for each class and values
        the weight assigned to each class based on the occurence of each class    
    '''
    labels_unique = torch.unique(labels)
    occurence = [len(torch.where(lu == labels)[0]) for lu in labels_unique]
    weights = [min(occurence) / o for o in occurence]
    labels_weights = {lu.item():w for lu,w in zip(labels_unique, weights)}
    weights = torch.Tensor(weights)
       
    return weights, labels_weights


def torch_effective_dim(X, center = True):
    '''Compute the effective dimension based on the eigenvalues of X
    
    Input
    -----
    X: tensor of shape (N,M) where N the samples and M the features
    
    center: bool, default True, indicating if X should be centered or not
    
    Output
    ------
    ed: effective dimension of X
    '''
    pca = torch.pca_lowrank(X, 
                            q = min(X.shape), 
                            center = center)
    eigenvalues = pca[1]
    eigenvalues = torch.pow(eigenvalues, 2) / (X.shape[0] - 1)
    li = eigenvalues /torch.sum(eigenvalues)
    ed = 1 / torch.sum(torch.pow(li, 2))
    
    return ed




###############################################################################################################
if __name__ == "__main__":
    import fire
    fire.Fire()


