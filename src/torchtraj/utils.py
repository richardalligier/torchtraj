from collections import namedtuple
import torch
from . import named
import numbers
from inspect import isfunction,ismethod

WPTS ="wpts"
T = "t"
PROJ = "proj"
XY = "xy"


def repeat_on_new_axis(lparams,ntimes,name):
    l = [named.repeat(named.unsqueeze(x,0,name),(ntimes,)+(1,) * len(x.shape)) for x in lparams]
    return l


def vheading(theta):
    cx = torch.cos(theta)
    cy = torch.sin(theta)
    return torch.cat([named.unsqueeze(vd,-1,XY) for vd in (cx,cy)],-1)

def compute_vxy(v,theta):
    vh = vheading(theta)
    return v.align_as(vh)*vh


def apply_mask(res,mask):
    return res * mask.align_as(res)


def clone(o):
    # print(type(o),isfunction(o),ismethod(o))
    if isinstance(o,dict):
        return {k:clone(v) for k,v in o.items()}
    elif isinstance(o,list):
        return [clone(v) for v in o]
    elif isinstance(o,tuple):
        return tuple(clone(v) for v in o)
    elif isinstance(o,numbers.Number) or isinstance(o,bool) or isinstance(o,str) or o is None or isfunction(o) or ismethod(o):
        return o
    else:
        return o.clone()

def to(o,device):
    if isinstance(o,dict):
        return {k:to(v,device) for k,v in o.items()}
    elif isinstance(o,list):
        return [to(v,device) for v in o]
    elif isinstance(o,tuple):
        return tuple(to(v,device) for v in o)
    elif isinstance(o,numbers.Number) or isinstance(o,bool) or isinstance(o,str) or o is None or isfunction(o) or ismethod(o):
        return o
    else:
        return o.to(device)
