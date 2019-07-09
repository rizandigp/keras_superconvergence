'''Cyclical learning rate policies.'''

import math


def triangular(x, slant=0.5):
    '''Triangular learning rate. Linear increase for the first 1-slant then linear decrease.'''
    x,_ = math.modf(x)
    if x<(1.0-slant):
        return x/(1.0-slant)
    else:
        return 1.0 - (x-(1.0-slant))/slant

def linear_then_cosine(x, slant=0.5):
    '''Learning rate that increases linearly then decreases with cosine annealing.'''
    x,_ = math.modf(x)
    if x<(1.0-slant):
        return x/(1.0-slant)
    else:
        return 0.5 + 0.5*math.cos((x-(1.0-slant))/slant*math.pi)

def cosine_with_restarts(x):
    '''Does cosine annealing and restarts every cycle
    Ref: https://arxiv.org/pdf/1608.03983.pdf
    '''
    x,_ = math.modf(x)
    return 0.5 + math.cos(x*math.pi)/2.0

def sine(x, slant=0.5):
    '''Slanted sine wave.'''
    x,_ = math.modf(x)
    if x<(1.0-slant):
        x_ = 0.5*x/(1.0-slant)
    else:
        x_ = 0.5 + 0.5*(x-(1.0-slant))/slant
    return 0.5 + 0.5*math.cos((x_-0.5)*math.pi*2.0)