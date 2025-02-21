import torch
from . import named
from .utils import WPTS, T, XY
from .flights import vheading,repeat_on_new_axis
import itertools
import matplotlib.pyplot as plt

DT0 = "dt0"
DT1 = "dt1"
DANGLE = "dangle"
DSPEED = "dspeed"
PARAMS = "params"

def zero_pad(dxy,whichtomove,nwpts):
    iwpts = dxy.names.index(WPTS)
    dims = list(dxy.shape)
    dims[iwpts]=nwpts
    res = torch.zeros(dims,device=dxy.device,dtype=dxy.dtype)
    slicewpt = (slice(None),)* iwpts + (slice(*whichtomove),)
    res[slicewpt]=dxy.rename(None)
    return res.rename(*dxy.names)

def processnames(dtnames,xnames):
    return dtnames+tuple(x for x in xnames if x not in dtnames)

def applydxy(f,paramnames,dxy):
    wpts = f.compute_wpts()
    # print(dxy.names)
    wptsfinalname = processnames(paramnames,wpts.names)
    # print(dxy)
    wpts = wpts.align_to(*wptsfinalname) + dxy.align_to(*wptsfinalname)#.align_as(wpts)
    # for i in range(wpts.shape[0]):
    #     plt.scatter(wpts[i,0,:,0].cpu(),wpts[i,0,:,1].cpu())
    # plt.axis('equal')
    # plt.show()
    dparams = {"xy0":f.xy0, "meanv":f.meanv(), "v":f.v, "turn_rate":f.turn_rate}
    dparams = {k:x.align_to(*processnames(paramnames,x.names)) for k,x in dparams.items()}
    dparams["wpts"]=wpts
    # newf = f.from_wpts(**dparams)
    return dparams

def adddt(dt,whichwpts,whichtomove,f):#xy0,v,theta,duration,turn_rate)
    vhead = vheading(f.theta)
    meanv = f.meanv()
    assert(vhead.names[-2]==WPTS)
    dxy = vhead[...,whichwpts:whichwpts+1,:] * meanv.align_as(vhead)[...,whichwpts:whichwpts+1,:]
    # print(f.duration)
    newnames = processnames(dt.names, dxy.names)
    assert((f.duration.align_to(*newnames)+dt.align_to(*newnames)).min()>=0.)
    dxy = dxy.align_to(*newnames) * dt.align_to(*newnames)
    dxy = zero_pad(dxy,whichtomove,f.nwpts())
    # wpts = f.compute_wpts()
    dparams = applydxy(f,dt.names,dxy)
    return f.from_wpts(**dparams)


def adddtwithoutchangingotherdates(dt,whichwpts,whichtomove,f):#xy0,v,theta,duration,turn_rate)
    vhead = vheading(f.theta)
    meanv = f.meanv()
    assert(vhead.names[-2]==WPTS)
    dxy = vhead[...,whichwpts:whichwpts+1,:] * meanv.align_as(vhead)[...,whichwpts:whichwpts+1,:]
    backdxy = vhead[...,whichwpts+1:whichwpts+2,:] * meanv.align_as(vhead)[...,whichwpts+1:whichwpts+2,:]
    newnames = processnames(dt.names, dxy.names)
    dxy = dxy.align_to(*newnames) * dt.align_to(*newnames)
    backdxy = -backdxy.align_to(*newnames) * dt.align_to(*newnames)
    dxy = zero_pad(dxy,whichtomove,f.nwpts())
    backdxy = zero_pad(backdxy,(whichtomove[0]+1,whichtomove[1]),f.nwpts())
    wpts = f.compute_wpts()
    return f.from_wpts(**applydxy(f,dt.names,dxy+backdxy))

# def adddtwithoutchangingotherdates(dt,whichwpts,whichtomove,f):#xy0,v,theta,duration,turn_rate)
#     vhead = vheading(f.theta)
#     meanv = f.meanv()
#     assert(vhead.names[-2]==WPTS)
#     dxy = vhead[...,whichwpts:whichwpts+1,:] * meanv.align_as(vhead)[...,whichwpts:whichwpts+1,:]
#     newnames = processnames(dt.names, dxy.names)
#     dxy = dxy.align_to(*newnames) * dt.align_to(*newnames)
#     dxy = zero_pad(dxy,whichtomove,f.nwpts())
#     backdxy = vhead[...,whichwpts+1:,:] * meanv.align_as(vhead)[...,whichwpts+1:,:]
#     backdt = f.duration[whichwpts+1:]-dt
#     backdxy = -backdxy.align_to(*newnames) * backdt.align_to(*newnames)
#     backdxy = zero_pad(backdxy,(whichtomove[0]+1,whichtomove[1]),f.nwpts())
#     wpts = f.compute_wpts()
#     return f.from_wpts(**applydxy(f,dt.names,dxy+backdxy))


import operator

def changespeed(dspeed,whichtomove,f):
    assert(f.v.names[-1] == WPTS)
    nwpts = f.v.shape[-1]
    wpts = f.compute_wpts()
    meanv = f.meanv()
    vhead = vheading(f.theta)
    velocity = vhead* meanv.align_as(vhead)
    # dxy = vhead * meanv.align_as(vhead) * f.duration
    # newnames = processnames(dspeed.names, dxy.names)
    # dxy = dxy.align_to(newnames) * dspeed.align_to(newnames)
    # dxy = extend(dxy,whichtomove,f.nwpts())
    # print(processnames(dspeed.names, f.v.names),f.v.names)
    # args = named.align(f.v,dspeed,processnames(dspeed.names, f.v.names))
    # print(args)
    # print(operator.mul(*args))
    # raise Exception
    # newnames = processnames(dspeed.names, f.v.names)
    def opalign(op,x,y):
        newnames = processnames(x.names,y.names)
        args = tuple(w.align_to(*newnames) for w in (x,y))
        return op(*args)
    newmeanv = opalign(operator.mul,dspeed-1,velocity)
    dxy = opalign(operator.mul,newmeanv,f.duration)
    assert(dxy.names[-2] == WPTS)
    # print(dxy.names)
    # raise Exception
    dxy = torch.cumsum(dxy,axis=-2)
    dxy = zero_pad(dxy[..., slice(*whichtomove),:], whichtomove,nwpts)
    dparams=applydxy(f, dspeed.names, dxy)
    for k in ("v","meanv"):
        dparams[k]= dparams[k] * dspeed.align_as(dparams[k])
        # dparams["v"]= dparams["v"] * dspeed.align_as(dparams["v"])
    return f.from_wpts(**dparams)
    # newmeanv = meanv.align_to(nev) * dspeed.align_to(newfv)
    # lrparams = [f.xy0, f.meanv(), f.v, f.turn_rate]
    # lrparams = [x.align_to(*processnames(dt.names,x.names)) for x in lrparams]
    # for x in lrparams+[wpts]:
    #     print(x.names)
    # newnames = processnames(dspeed.names,f.v.names)
    # newv = f.v.align_to(*newnames) * dspeed.align_to(*newnames)
    # newduration = f.duration.align_to(*newnames) / dspeed.align_to(*newnames)
    # lrparams = [f.xy0, newv, f.theta, newduration, f.turn_rate]
    # lrparams = [x.align_to(*processnames(dspeed.names,x.names)) for x in lrparams]
    # return f.from_argslist(lrparams)


# def addspeed(dspeed,whichtomove,f):
#     assert(f.v.names[-1] == WPTS)
#     assert(whichtomove.dim()==1)
#     newnames = processnames(dspeed.names,f.v.names)
#     newv = f.v.align_to(*newnames) * dspeed.align_to(*newnames)
#     newduration = f.duration.align_to(*newnames) / dspeed.align_to(*newnames)
#     lrparams = [f.xy0, newv, f.theta, newduration, f.turn_rate]
#     lrparams = [x.align_to(*processnames(dspeed.names,x.names)) for x in lrparams]
#     return f.from_argslist(lrparams)


def addangle(dtheta,whichwpts,whichtomove,f):
    assert(f.theta.names[-1]==WPTS)
    newnames = processnames(dtheta.names,f.theta.names)
    newtheta = f.theta.align_to(*newnames) + dtheta.align_to(*newnames)
    newvheading = vheading(newtheta)
    dxytheta = newvheading - vheading(f.theta).align_as(newvheading)
    newnames = processnames(dtheta.names,dxytheta.names)
    dxy = dxytheta.align_to(*newnames) * f.meanv().align_to(*newnames) * f.duration.align_to(*newnames)
    assert(dxy.names[-2]==WPTS)
    dxy = zero_pad(dxy[...,whichwpts:whichwpts+1,:],whichtomove,f.nwpts())
    # raise Exception
    dparams=applydxy(f, dtheta.names, dxy)
    return f.from_wpts(**dparams)

def addlongitudinaldspeed(dspeed,f):
    newnames = processnames(dspeed.names,f.v.names)
    newv = f.v.align_to(*newnames) * dspeed.align_to(*newnames)
    newnames = processnames(dspeed.names,f.duration.names)
    newduration = f.duration.align_to(*newnames) / dspeed.align_to(*newnames)
    dparams = f.dictparams()
    dparams = {k:x.align_to(*processnames(dspeed.names,x.names)) for k,x in dparams.items()}
    dparams["v"] = newv
    dparams["duration"] = newduration
    return f.from_argsdict(dparams)


def plot(xy):
    # t = torch.arange(xy[0,:,1].numpy().shape[0])
    # plt.plot(t,xy[0,:,0].numpy())
    # plt.scatter(t,xy[0,:,0].numpy())
    indices = itertools.product(*[ list(range(n))for n in xy.shape[:-2]])
    for ij in indices:
        ijx = ij + (slice(None), 0)
        ijy = ij + (slice(None), 1)
        plt.scatter(xy[*ijx].cpu().numpy(),xy[*ijy].cpu().numpy())
if __name__ == '__main__':

    import time
    import sys
    import traj
    from utils import WPTS, XY,T
    from flights import Flights, FlightsWithAcc


        # for i in range(xy.shape[0]):
        #     plt.plot(xy[i,0,:,0].cpu().numpy(),xy[i,0,:,1].cpu().numpy())
        # plt.scatter(xy[0,:,0].numpy(),xy[0,:,1].numpy())
    def main():
        import numpy as np
        torch.random.manual_seed(0)
        device = torch.device("cpu")#"cuda" if torch.cuda.is_available() else "cpu")
        print(device)
        n = 1
        nwpts = 4
        namesbatch = ("batch",)
        names = namesbatch + (WPTS,)
        shape = (n, nwpts)
        shapeturn_rate = (n,nwpts-1)
        xy0 = torch.randn(shape[:-1]+(2,),names=namesbatch+(XY,),device=device) * 100
        v = 10+torch.rand(shape,names=names,device=device)*10
        dtheta = (torch.rand(shape,names=names,device=device)-0.5)*(1.5*torch.pi)
        theta = torch.cumsum(dtheta,axis=-1)
        duration = 20 + torch.rand(shape,names=names,device=device)*30
        duration[:,1]=20
        turn_rate = torch.ones(shapeturn_rate,names=names,device=device)*1e0
        # turn_rate = torch.ones(1,names=(WPTS,),device=device)*1e0
        f = FlightsWithAcc(xy0,v,theta,duration,turn_rate)
        # acceleration = torch.ones(shape,names=names,device=device) * 1e12
        def adddts(f):
            dt0 = torch.tensor([[0.,-10.]],names=(PARAMS,DT0,),device=device)
            dt1 = torch.tensor([[0.,-10.]],names=(PARAMS,DT1,),device=device)
            # f = adddt(DT0,dt0,f)
            whichtomovedt0 = (0,2)#torch.tensor([True,True,False,False])#,names=(WPTS,))
            f = adddt(dt0,0,whichtomovedt0,f)
            whichtomovedt1 = (1,2)#torch.tensor([False,True,False,False])#,names=(WPTS,))
            f = adddt(dt1,1,whichtomovedt1,f)
            return f
        def adddtswithoutchangingt1(f):
            dt0 = torch.tensor([[0.,10.]],names=(PARAMS,DT0,),device=device)
            dt1 = torch.tensor([[0.,10.]],names=(PARAMS,DT1,),device=device)
            # f = adddt(DT0,dt0,f)
            whichtomovedt0 = (0,2)#torch.tensor([True,True,False,False])#,names=(WPTS,))
            f = adddtwithoutchangingotherdates(dt0,0,whichtomovedt0,f)
            # whichtomovedt1 = (1,2)#torch.tensor([False,True,False,False])#,names=(WPTS,))
            # f = adddt(dt1,1,whichtomovedt1,f)
            return f
        def adddangles(f):
            dtheta0 = torch.tensor([[-np.radians(3),np.radians(3)]],names=(PARAMS,DANGLE,),device=device)
            # f = adddt(DT0,dt0,f)
            whichtomove = (1,2)#torch.tensor([False,True,False,False])#,names=(WPTS,))
            f = addangle(dtheta0,1,whichtomove,f)
            return f
        def adddspeeds(f):
            dspeed = torch.tensor([[1.1,0.9]],names=(PARAMS,DSPEED,),device=device)
            whichtomove = (0,2)#torch.tensor([True,True,True,True])
            f = changespeed(dspeed,whichtomove,f)
            # whichtomove = torch.tensor([True,True,False,False])
            # f = adddt(dt,whichtomove,f)
            return f
        def addlongspeeds(f):
            dspeed = torch.tensor([[1.1,1.]],names=(PARAMS,DSPEED,),device=device)
            f = addlongitudinaldspeed(dspeed,f)
            # whichtomove = torch.tensor([True,True,False,False])
            # f = adddt(dt,whichtomove,f)
            return f
        # f = adddspeeds(f)
        f = adddts(f)
        print(f.meanv())
        # f = adddtswithoutchangingt1(f)
        print(f.meanv())
        # f = adddangles(f)
        # f = addlongspeeds(f)
        t = torch.linspace(0,100,100,device=device).rename(T)
        t0 = time.perf_counter()
        xy = traj.generate(f,  t)#.cpu()
        # if torch.cuda.is_available():
        #     torch.cuda.synchronize()
        # print(time.perf_counter()-t0)
        # print(xy)
        # print(xy.shape)

        
        # print(xy.shape,sys.getsizeof(xy.storage())/1024/1024)
        # print(torch.cuda.max_memory_allocated()/1024/1024)
        # print(xy)
        # print(xy.shape,xy.names)
        plot(xy)
        plt.axis('equal')
        plt.show()
    main()


