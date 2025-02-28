from . import named
from .utils import WPTS, XY
import pandas as pd
import torch
import numpy as np
def vheading(theta):
    cx = torch.cos(theta)
    cy = torch.sin(theta)
    return torch.cat([named.unsqueeze(vd,-1,XY) for vd in (cx,cy)],-1)

def repeat_on_new_axis(lparams,ntimes,name):
    l = [named.repeat(named.unsqueeze(x,0,name),(ntimes,)+(1,) * len(x.shape)) for x in lparams]
    return l


def stack_lflights(lflights,batchname):
    t = type(lflights[0])
    for f in lflights:
        assert(t==type(f))
    d = {k:named.stack([f.dictparams()[k] for f in lflights],batchname) for k in lflights[0].dictparams()}
    return lflights[0].from_argsdict(d)

def cat_lflights(lflights,dim=0):
    t = type(lflights[0])
    for f in lflights:
        assert(t==type(f))
    d = {k:named.cat([f.dictparams()[k] for f in lflights],dim=dim) for k in lflights[0].dictparams()}
    return lflights[0].from_argsdict(d)




class Flights:
    ncoords = 2
    def __init__(self,xy0,v,theta,duration,turn_rate):
        assert(v.min()>0)
        self.xy0 = xy0
        self.v = v
        self.theta = theta
        self.duration = duration
        self.turn_rate = turn_rate
        self.check_names()
        # self.xy0 = xy0.rename(None)
        # self.v = v.rename(None)
        # self.theta = theta.rename(None)
        # self.duration = duration.rename(None)
        # self.turn_rate = turn_rate.rename(None)

    def dictparams(self):
        return {"xy0":self.xy0,"v":self.v,"theta":self.theta,"duration":self.duration,"turn_rate":self.turn_rate}

    def check_names(f):
        # print("enterchecknames")
        # print(f)
        assert(f.duration.names[-1]==WPTS)
        assert(f.xy0.names[-1]==XY)
        assert(f.theta.names[:-1]==f.xy0.names[:-1])
        assert(f.v.names==f.duration.names)
        assert(f.theta.names == f.duration.names)
        assert(f.turn_rate.names==f.duration.names)
        # print("endchecknames")
        # assert(f.theta.names == f.duration.names)
        # assert(f.turn_rate.names==f.duration.names)
    @property
    def names(self):
        return self.v.names[:-1]
    @classmethod
    def new(cls,xy0,v,theta,duration,turn_rate):
        return cls(xy0,v,theta,duration,turn_rate)
    @classmethod
    def _meanv(cls,v):
        return v
    def meanv(self):
        return self._meanv(self.v)
    def nwpts(self):
        return self.duration.shape[-1]
    def total_duration(self):
        return self.duration.sum(axis=-1)
    def device(self):
        return self.xy0.device
    def dtype(self):
        return self.xy0.dtype
    def speed_at_turns(self):
        return self.v[...,:-1]
    def __str__(self):
        return "\n".join(k+":"+str(v)+str(v.shape) for k,v in sorted(self.dictparams().items()))

    def segdist(self,clipped_t,duration):
        return self.v.align_as(clipped_t) * clipped_t
    def to_dataframe(self):
        def torchtodict(t,name):
            assert(len(t.shape)==2)
            names = "_".join(t.names)
            for i in range(t.shape[1]):
                yield f"{name}_{str(i)}_{names}",t[:,i].numpy()
        return pd.DataFrame({k:v for n,t in self.dictparams().items() for k,v in torchtodict(t,n)})
    @classmethod
    def from_dataframe(cls,df,suffix=""):
        lnames = ["xy0","v","theta","duration","turn_rate"]
        def split(name,var):
            if var[:len(name)]==name:
                s = var.split("_")
                vname = "_".join(var.split("_")[:-3])
                if vname == name:
                    i = int(s[-3])
                    names = tuple(s[-2:])
                    # print(names)
                    return (vname,i,names)
            return None
        dv = {k:sorted([x for x in (split(k,v[:len(v)-len(suffix)]) for v in list(df) if v.endswith(suffix)) if x is not None]) for k in lnames}
        # print(dv)
        for vis in dv.values():
            nv = len(vis)
            assert(max(i for _,i,_ in vis)==nv-1)
            assert(min(i for _,i,_ in vis)==0)
        def to_tensor(vs):
            return torch.tensor(np.array([df[f"{v}_{i}_{'_'.join(names)}{suffix}"].values for v,i,names in vs]))
        d = {k:torch.transpose(to_tensor(vs),0,1).rename(*vs[0][-1]) for k,vs in dv.items()}
        return cls.from_argsdict(d)
    def compute_wpts(self):
        dist = self.meanv() * self.duration
        c = vheading(self.theta)
        xy = c * dist.align_as(c)
        return self.xy0.align_as(xy) + torch.cumsum(xy,axis = xy.names.index(WPTS))
    @classmethod
    def cpu(cls, flights):
        return cls(**{k:v.cpu() for k,v in flights.dictparams().items()})
    @classmethod
    def clone(cls, flights):
        return cls(**{k:v.clone() for k,v in flights.dictparams().items()})
    def to(self,device=None,dtype=None):
        return self.dmap(self,f=lambda v:v.to(device=device,dtype=dtype))
    @classmethod
    def dmap(cls, flights,f):
        return cls(**{k:f(v) for k,v in flights.dictparams().items()})
    @classmethod
    def from_argslist(cls,l):
        return cls(*l)
    @classmethod
    def from_argsdict(cls,d):
        return cls(**d)
    @classmethod
    def from_wpts(cls,xy0,v,turn_rate,wpts):
        # print(xy0.names,wpts.names)
        assert(wpts.names[-1]==XY)
        assert(wpts.names[-2]==WPTS)
        meanv = cls._meanv(v)
        xy0w = xy0.align_as(wpts)
        # print(wpts.shape,xy0w.shape)
        # print(xy0w.expand(wpts.shape).shape)
        xy0wshape = list(wpts.shape)
        iwpts = xy0w.names.index(WPTS)
        xy0wshape[iwpts] = 1
        xy0wpts = torch.cat((xy0w.expand(xy0wshape),wpts),axis=iwpts)
        # print(xy0wpts)
        # print(xy0wpts.shape)
        # raise Exception
        dxy = xy0wpts[...,1:,:]-xy0wpts[...,:-1,:]
        theta = torch.atan2(dxy[...,1],dxy[...,0])
        # print(theta)
        # print(dxy[...,1].shape,meanv.shape)
        duration = torch.hypot(dxy[...,1],dxy[...,0])/meanv#.expand(dxy.shape)
        # print(duration)
        return cls(xy0,v,theta,duration,turn_rate)


class FlightsWithAcc(Flights):
    def repeat_on_new_axis(self,ntimes,name):
        return FlightsWithAcc(*super().repeat_on_new_axis(ntimes,name).listparams())
    # def meanv(self):
    #     assert(self.v.names[-1]==WPTS)
    #     v = named.pad(self.v,(0,1),'replicate')
    #     return (v[...,:-1] + v[...,1:]) * 0.5
    @classmethod
    def _meanv(cls,v):
        assert(v.names[-1]==WPTS)
        meanv = (v[...,:-1] + v[...,1:]) * 0.5
        meanv = torch.cat((meanv,v[...,-1:]),axis=-1)
        return meanv
    # def meanv(self):
    #     assert(self.v.names[-1]==WPTS)
    #     meanv = (self.v[...,:-1] + self.v[...,1:]) * 0.5
    #     meanv = torch.cat((meanv,self.v[...,-1:]),axis=-1)
    #     return meanv
    def speed_at_turns(self):
        return self.v[...,1:]
    def segdist(self,clipped_t,duration):
        # print(self.v.names, duration.names)
        acc = named.pad(torch.diff(self.v,axis=-1),(0,1),'constant',0)/duration.align_as(self.v)
        return clipped_t * (self.v.align_as(clipped_t) + acc.align_as(clipped_t) * 0.5 * clipped_t)
    # @classmethod
    # def from_Flights(self,f):

