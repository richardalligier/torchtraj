from collections import namedtuple
import torch
from . import named

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
