import pandas as pd
import matplotlib.pyplot as plt
import time
import torch
from utils import WPTS, T, XY
from flights import Flights,FlightsWithAcc
import numpy as np
from traj import CannotTurnError
# df = pd.read_parquet("AA39115843.parquet")
# print(list(df))

class Scaler:
    def __init__(self):
        self.fs = None
        self.xymin = None
        self.tmax = None

    def scale(self,fmodel,trajreal,t):
        # Sortie: fmodel scale, traj scale, t scale
        namesbatch = tuple()
        names = namesbatch + (WPTS,)
        # print(trajreal.names)
        # assert(len(trajreal.shape)==2)
        xmin , xmax = trajreal[...,0].min().item(),trajreal[...,0].max().item()
        ymin , ymax = trajreal[...,1].min().item(),trajreal[...,1].max().item()
        dscale = max(ymax-ymin,xmax-xmin)
        xymin = torch.tensor([xmin,ymin],dtype=trajreal.dtype,device=trajreal.device)
        fs = 1 / dscale
        #print(fs,xymin,((trajreal - xymin) * fs)[...,0].min(),((trajreal - xymin) * fs)[...,0].max(),((trajreal - xymin) * fs)[...,1].min(),((trajreal - xymin) * fs)[...,1].max())
        #xymin , xymax = trajreal.min().item(),trajreal.max().item()
        #fs = 1 / (xymax - xymin)
        #print(fs,xymin,((trajreal - xymin) * fs)[...,0].min(),((trajreal - xymin) * fs)[...,0].max(),((trajreal - xymin) * fs)[...,1].min(),((trajreal - xymin) * fs)[...,1].max())
        #raise Exception
        tmax = torch.max(t)
        new_fmodel = FlightsWithAcc((fmodel.xy0  - xymin) * fs,fmodel.v  * fs* tmax,fmodel.theta,fmodel.duration / tmax,fmodel.turn_rate*tmax)
        new_trajreal = (trajreal - xymin) * fs
        self.fs = fs
        self.xymin = xymin
        self.tmax = tmax
        return (new_fmodel,new_trajreal, t / tmax)

    def unscale(self,fmodel):
        new_fmodel = FlightsWithAcc(fmodel.xy0 / self.fs + self.xymin,fmodel.v / self.fs / self.tmax,fmodel.theta,fmodel.duration * self.tmax ,fmodel.turn_rate / self.tmax)
        return new_fmodel


def randomfpl(device,n, nwpts = 4):
    namesbatch = ("batch",)
    names = namesbatch + (WPTS,)
    shape = (n, nwpts)
    shapeturn_rate = (n,nwpts-1)
    xy0 = torch.zeros(shape[:-1]+(2,),names=namesbatch+(XY,),device=device)
    v = 10+torch.rand(shape,names=names,device=device)*10
    dtheta = (torch.rand(shape,names=names,device=device)-0.5)*(0.5*torch.pi)
    theta = torch.cumsum(dtheta,axis=-1)
    duration = 200/nwpts * torch.ones_like(v)
    turn_rate = torch.ones(shapeturn_rate,names=names,device=device)*1e0
    return FlightsWithAcc(xy0,v,theta,duration,turn_rate)


# def constraint_duration(dres):
#     dres["duration"]


def scaled_fit(fmodel,trajreal,tmodel,gen,niter,minturn_rate,cst_duration,initial_duration,min_duration):
    d = {"xy0":fmodel.xy0}
    dopti = { "duration":fmodel.duration,"v":fmodel.v,"theta":fmodel.theta,"turn_rate":fmodel.turn_rate,}
    dnoname = {k:v.clone().rename(None) for k,v in dopti.items()}
    # if cst_duration is not None:
    #     cst_duration = cst_duration
    #     initial_duration = dnoname["duration"].clone()
    for x in dnoname.values():
        x.requires_grad=True
    lr = 1e-4
    optimizer = torch.optim.Adam([
        {'params':dnoname['duration'],'lr':lr},
        {'params':dnoname['v'],'lr':lr * 1e2},# * 1e2},
        {'params':dnoname['theta'],'lr':lr * 1e2},
        {'params':dnoname['turn_rate'],'lr':lr*1e3},
    ],lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,verbose=True,patience=1000,factor=0.9)
    memlowest = None
    losslowest = None
    # assert(t.diff().min()>0)
    lastok = minturn_rate
    def direction(xy):
        dimt = xy.names.index(T)
        dimxy = xy.names.index(XY)
        dxy = torch.diff(xy,dim=dimt).rename(None)
        # assert((dxy!=0.).all())
        return dxy/(torch.linalg.vector_norm(dxy,dim=dimxy,keepdim=True)+1e-20)
    nbtrajs = trajreal.shape[0]
    for i in range(niter):
        dres = d.copy()
        for k,v in dnoname.items():
            dres[k]=v.rename(*dopti[k].names)
        dres["v"] = torch.abs(dres["v"])
        if cst_duration is not None:
            dres["duration"]=initial_duration+torch.clip(dres["duration"]-initial_duration,min=-cst_duration,max=cst_duration)
        dres["duration"] = torch.clip(dres["duration"],min=min_duration)
        dres["turn_rate"] = torch.maximum(dres["turn_rate"].rename(None),minturn_rate).rename(*dres["turn_rate"].names)
        # dres["turn_rate"] = torch.clip(dres["turn_rate"].rename(None),min=minturn_rate).rename(*dres["turn_rate"].names)
        optimizer.zero_grad()
        # dres["turn_rate"] = torch.clip(dres["turn_rate"],min=minturn_rate*scaler.tmax)
        # print((dres["turn_rate"]/scaler.tmax).min())
        # print(dres["v"].min())# = torch.abs(dres["v"])
        f = FlightsWithAcc(**dres)
        # newok = dres["turn_rate"].rename(None)/scaler.tmax
        newok = minturn_rate
        try:
            out = gen(f,tmodel)
            # print(tmodel)
            # print(trajreal)
            # print(out)
            # print(direction(out))
        except CannotTurnError as e:
            # print(e.args)
            # print(lastok,newok)
            mask = torch.logical_or(e.args[0][...,1:],e.args[0][...,:-1]).rename(None)
            lastok[mask]=lastok[mask] * 2
            # print(lastok.max()/scaler.tmax)#,lastok>0.)
            # lastok = lastok*2
            return scaled_fit(fmodel,trajreal,tmodel,gen,niter,minturn_rate,cst_duration,initial_duration,min_duration)#fit(fmodel_noscale,trajreal_noscale,tmodel,gen,niter,minturn_rate=lastok/scaler.tmax,cst_duration=None if cst_duration is None else cst_duration*scaler.tmax)
        lastok = newok
        # print(out.shape)
        # raise Exception
        # outdxy = torch.diff(out,dim=0)
        # outdxy =
        # direrr = ((torch.diff(out,dim=0)-torch.diff(trajreal,dim=0))**2).mean()
        # print(f)
        # print(t)
        # print(t.shape)
        # print(nbtrajs)
        direrr = nbtrajs*((direction(out)-direction(trajreal))**2).mean()
        # print(out.names)
        # print(trajreal.names)
        poserr = nbtrajs*((out-trajreal)**2).rename(None).mean()
        loss = poserr#*(1+100.*direrr)
        assert(not torch.isnan(loss).any())
        # raise Exception
        #print(loss)
        if losslowest is None or losslowest>loss.item():
            memlowest={k:v.clone().detach() for k,v in dres.items()}
            losslowest=loss.item()
        # print(loss.item(),losslowest)
        loss.backward()
        optimizer.step()
        scheduler.step(loss.item())
    # print("unscaledlosslowest",losslowest/scaler.fs**2)
    f = FlightsWithAcc(**memlowest)
    f = f.dmap(f,lambda v:v.detach())
    return f

#sans t [0,1], couplefit: lr  = 1e-4; 'lr' = lr * 1e
def fit(fmodel_noscale,trajreal_noscale,tmodel,gen,niter,minturn_rate=None,cst_duration=None):#,dlr,factor,patience):
    print("minturn_rate",None if minturn_rate is None else minturn_rate.max())
    # Sortie : f opti unscale
    scaler = Scaler()
    (fmodel,trajreal,t)  = scaler.scale(fmodel_noscale,trajreal_noscale,tmodel)
    if minturn_rate == None:
        minturn_rate = 1e-3*torch.ones_like(fmodel_noscale.turn_rate).rename(None)
    minturn_rate = minturn_rate*scaler.tmax#*torch.ones_like(fmodel_noscale.turn_rate).rename(None)
    d = {"xy0":fmodel.xy0}
    dopti = { "duration":fmodel.duration,"v":fmodel.v,"theta":fmodel.theta,"turn_rate":fmodel.turn_rate,}
    dnoname = {k:v.clone().rename(None) for k,v in dopti.items()}
    if cst_duration is not None:
        cst_duration = cst_duration / scaler.tmax
    initial_duration = dnoname["duration"].clone()
    min_duration = 1e-1 / scaler.tmax
    f = scaled_fit(fmodel,trajreal,t,gen,niter,minturn_rate,cst_duration,initial_duration,min_duration)
    # f = FlightsWithAcc(**memlowest)
    # f = f.dmap(f,lambda v:v.detach())
    # print(minturn_rate)
    return scaler.unscale(f)#scaler.unscale(f)#f#.dmap(f,lambda v:v.detach().cpu())

def main():
    import traj
    torch.random.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    freal = randomfpl(device,1)
    t = torch.linspace(0,200,1000,device=device).rename(T)
    trajreal = traj.generate(freal,  t)
    trajreal = trajreal + 3 * torch.randn_like(trajreal)
    fmodel = randomfpl(device,1)
    fopti = fit(fmodel,trajreal,t,traj.generate,1000)
    with torch.no_grad():
        xy = traj.generate(fmodel,t)#.cpu(
    plt.scatter(xy[0,:,0].cpu().numpy(),xy[0,:,1].cpu().numpy(),color="red")
    with torch.no_grad():
        xy = traj.generate(fopti,t)#.cpu(
    print(((trajreal-xy)**2).mean())
    print(fmodel.duration)
    print(fmodel.theta)
    print(fopti.duration)
    print(freal.duration)
    plt.scatter(xy[0,:,0].cpu().numpy(),xy[0,:,1].cpu().numpy(),color="blue")
    plt.scatter(trajreal[0,:,0].cpu().numpy(),trajreal[0,:,1].cpu().numpy(),color="green")
    plt.show()

def testdims():
    from functorch.dim import dims
    a=torch.rand((2,)*5)
    j = dims(a.dim())
    ac = a[*j]
    print(ac.dims)
    print(ac)
    print((ac.order(*ac.dims)==a).all())

    raise Exception
    input = torch.rand(2, 3, 224, 224)
    v = dims(input.dim())
    input.requires_grad=True
    optimizer = torch.optim.SGD([input],lr=1e-3)
    for i in range(0):
        optimizer.zero_grad()
        input_fc=input[*v]
        a=input_fc.unsqueeze(-1)

        print(b)
        print(j.size,k.size)
        raise Exception
        loss = torch.abs(input_fc).sum(dim=v)#.order()#.order(v[0])
        print(loss.item())
        loss.backward()
        optimizer.step()
    print(input)
def debug():
    from torchtraj import traj
    d ={
        "duration":torch.tensor([4.5320e-02, 5.5332e-01, 4.0021e-01, 1.5870e-04], names=('wpts',)),
        "theta": torch.tensor([1.9015, 2.1331, 1.6562, 1.6538], names=('wpts',)),
        "turn_rate": torch.tensor([1., 1., 1.], names=('wpts',)),
        "v":torch.tensor([1.0966, 1.2372, 1.1277, 1.0303], names=('wpts',)),
        "xy0":torch.tensor([0.3641, 0.0000], names=('xy',)),
    }
    f = FlightsWithAcc.from_argsdict(d)
    t = torch.linspace(0,1,1314).rename(T)
    print(f)
    xy = traj.generate(f,t)
    dxy=torch.diff(xy,dim=0).rename(None)
    print(dxy)
if __name__ == '__main__':
    main()

