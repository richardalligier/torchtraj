from . import named
from .utils import WPTS, XY, vheading, T, repeat_on_new_axis
import pandas as pd
import torch
import numpy as np
from . import traj
from . import uncertainty


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
    maxnwpts = max(f.nwpts() for f in lflights)
    lflights = [f.pad_wpts_at_end(maxnwpts-f.nwpts()) for f in lflights]
    d = {k:named.cat([f.dictparams()[k] for f in lflights],dim=dim) for k in lflights[0].dictparams()}
    return lflights[0].from_argsdict(d)


def named_cat_lflights(lflights,dimname):
    t = type(lflights[0])
    for f in lflights:
        assert(t==type(f))
    maxnwpts = max(f.nwpts() for f in lflights)
    # res = []
    # for f in lflights:
    #     print(maxnwpts,f.nwpts())
    #     print(f.duration.names,f.duration.shape)
    #     res.append(f.pad_wpts_at_end(maxnwpts-f.nwpts()))
    # lflights = res
    lflights = [f.pad_wpts_at_end(maxnwpts-f.nwpts()) for f in lflights]
    d = {k:named.cat([f.dictparams()[k] for f in lflights],dim=v.names.index(dimname)) for k,v in lflights[0].dictparams().items()}
    return lflights[0].from_argsdict(d)


def get_tstart(tend):
    return torch.cat([torch.zeros_like(tend[...,:1]),tend[...,:-1]],axis=-1)

class Flights:
    # ncoords = 2
    def __init__(self,xy0,v,theta,duration,turn_rate):
        self.xy0 = xy0
        self.v = v
        self.theta = theta
        self.duration = duration
        self.turn_rate = turn_rate
        self.check_names()
        assert(v.min()>0)

        # self.xy0 = xy0.rename(None)
        # self.v = v.rename(None)
        # self.theta = theta.rename(None)
        # self.duration = duration.rename(None)
        # self.turn_rate = turn_rate.rename(None)

    def dictparams(self):
        return {"xy0":self.xy0,"v":self.v,"theta":self.theta,"duration":self.duration,"turn_rate":self.turn_rate}
    def serialize(self):
        d = self.dictparams()
        return self.deserialize,{k: named.serialize(v) for k,v in d.items()}
    @classmethod
    def deserialize(cls,d):
        res = {k:v[0](v[1]) for k,v in d.items()}
        return cls(**res)

    def check_names(f):
        # print("enterchecknames")
        # print(f)
        assert(f.duration.names[-1]==WPTS)
        assert(f.xy0.names[-1]==XY)
        # print(f.v.names,f.duration.names)
        assert(f.v.names==f.duration.names)
        assert(f.turn_rate.names==f.duration.names)
        assert(f.theta.names[:-1]==f.xy0.names[:-1])
        assert(f.theta.names == f.duration.names)
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

    def pad_wpts_at_end(self,npad):
        d = {k:v.clone() for k,v in self.dictparams().items()}
        # print(self.turn_rate)
        # print(self.turn_rate.shape)
        if npad == 0:
            return self.from_argsdict(d)
        assert(self.turn_rate.shape[-1]==1)
        durationbeforesplit=self.duration.cumsum(axis=-1)[...,-1]
        for v in ["v","theta"]:
            d[v]=named.pad(d[v],(0,npad),mode="replicate")
        lastduration = self.duration[...,-1]#.min().item()
        step_t = torch.trunc(lastduration * 0.9 / npad * 1024.) / 1024.
        assert( step_t.min()>0.)
        added_t = step_t * npad
        d["duration"][...,-1] = d["duration"][...,-1] - added_t
        for v in ["duration"]:
            # print("duration",npad)
            # print("orig",d[v].names,d[v].shape)
            d[v]=named.pad(d[v],(0,npad),mode="constant",value=0.)#added_t / npad * 0.5)
            # print("pade",d[v].names,d[v].shape)
            # raise Exception
            # print("step",step_t.names,step_t.shape)
            d[v][...,-npad:] = step_t.rename(None).unsqueeze(-1)
        durationaftersplit=d[v].cumsum(axis=-1)[...,-1]
        # print(self.duration.cumsum(axis=-1))
        # print(d["duration"].cumsum(axis=-1))
        diff = (durationaftersplit.rename(None)-durationbeforesplit.rename(None)).abs().max().item()
        assert(diff>=0.)
        assert (diff==0.)
        return self.from_argsdict(d)

    def add_wpt_at_t(self,t:torch.tensor):
        assert(t.min()>0.)
        assert(T not in t.names)
        assert(WPTS in t.names)
        for x in t.names:
            assert x in self.duration.names
        tend = torch.cumsum(self.duration,axis=-1)
        assert(tend.names[-1]==WPTS)
        tstart = get_tstart(tend)
        assert((tend-tstart).min()>0.)
        t = t.align_as(tend)
        # print(f"{t.shape=} {t.names=} {tend.shape=} {tend.names=}")
        tshape = tuple(named.broadcastshapes(t.shape,tend.shape))
        tshape = tshape[:-1]+(1,)
        t = t.broadcast_to(tshape).clone()
        # print(f"{t.names=}")
        merged = torch.cat([t,tend],axis=-1)
        newtend = named.sort(merged,dim=WPTS)
        # raise Exception
        def separate(tend):
            tstart = get_tstart(tend)
            duration = tend-tstart
            if duration.min()>0.:
                return tend
            else:
                dmin = duration.rename(None)[(duration > 0.).rename(None)].min()
                # print(f"{dmin=}")
                # raise Exception
                tend = tend - dmin * 0.5 * (duration==0.)
                res = named.sort(tend,dim=WPTS)
                return separate(res)
        newtend  = separate(newtend).align_as(tend)
        newtstart = get_tstart(newtend)
        d = {}
        # xy = traj.generate(self, newtend.rename(**{WPTS:T})).rename(**{T:WPTS})
        iwpts = self.which_seg_at_t(newtstart.rename(**{WPTS:T}))#.rename(**{T:WPTS})
        d["duration"] = (newtend - newtstart)#.align_as(self.duration)
        d["v"] = uncertainty.gather_wpts(self.v,iwpts).rename(**{T:WPTS}).align_as(d["duration"])
        d["theta"] = uncertainty.gather_wpts(self.theta,iwpts).rename(**{T:WPTS}).align_as(d["duration"])

        assert(self.turn_rate.shape[-1]==1)
        d["turn_rate"] = self.turn_rate.clone().align_as(d["duration"])
        d["xy0"] = self.xy0.clone()
        assert(d["duration"].min()>0.)
        return self.new(**d)

    def _wpts_at_t(self,t:torch.tensor):
        names = named.mergenames((self.duration.names,t.names))
        dates = self.duration.cumsum(axis=-1)
        # print(dates)
        # print(t)
        mask = (t.align_to(*names) == dates.align_to(*names)).align_to(...,WPTS)
        indexes = torch.arange(start=1,end=1+dates.shape[-1],device=mask.device).rename(WPTS)
        res = (indexes * mask).sum(axis=-1)
        return res

    def wpts_at_t(self,t:torch.tensor):
        res = self._wpts_at_t(t)
        # print(res)
        assert(res.min()>0)
        return res

    def is_wpts_at_t(self,t:torch.tensor):
        return self._wpts_at_t(t) > 0

    def which_seg_at_t(self,t):
        names = named.mergenames((self.duration.names,t.names))
        tend = torch.cumsum(self.duration,axis=-1)
        tstart = get_tstart(tend)
        _,iwpts =torch.max((t.align_to(*names)<tend.align_to(*names)).rename(None),dim=-1)
        return iwpts.rename(*names[:-1])
        # v = uncertainty.gather_wpts(self.v,iwpts)
        # print(v)

    def shift_xy0(self,t):
        # print(t,type(t))
        assert(isinstance(t,float))
        assert(t>=0.)
        dparam = {k:v.clone() for k,v in self.dictparams().items()}
        if t == 0.:
            return self.new(**dparam)
        assert(dparam["turn_rate"].shape[-1]==1)
        for k in ["theta","duration","v"]:
            assert(dparam[k].names[-1]==WPTS)
            dparam[k]=named.cat([dparam[k][...,:1],dparam[k]],dim=-1)
        v0 = vheading(self.theta[...,0])
        dparam["xy0"] = self.xy0 -  t * self.v[...,0].align_as(self.xy0) * v0
        dparam["duration"][...,0] = t
        return self.new(**dparam)

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
    # @classmethod
    # def cpu(cls, flights):
    #     return cls(**{k:v.cpu() for k,v in flights.dictparams().items()})
    # @classmethod
    # def clone(cls, flights):
    #     return cls(**{k:v.clone() for k,v in flights.dictparams().items()})
    # def to(self,device=None,dtype=None):
    #     return self.dmap(self,f=lambda v:v.to(device=device,dtype=dtype))
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

def add_tensors_operations(class_type):
    def f(opname):
        def f(self,*args,**kwargs):
            return type(self)(**{k:getattr(v,opname)(*args,**kwargs) for k, v in self.dictparams().items()})
        return f
    for opname in ["cpu","to","clone"]:
        setattr(class_type, opname, f(opname))

add_tensors_operations(Flights)

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
    def v_at_t(self,t):
        raise NotImplemented
    def segdist(self,clipped_t,duration):
        # print(self.v.names, duration.names)
        acc = named.pad(torch.diff(self.v,axis=-1),(0,1),'constant',0)/duration.align_as(self.v)
        return clipped_t * (self.v.align_as(clipped_t) + acc.align_as(clipped_t) * 0.5 * clipped_t)
    # @classmethod
    # def from_Flights(self,f):

