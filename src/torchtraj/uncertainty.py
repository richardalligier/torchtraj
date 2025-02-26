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




def zero_pad_old(dxy,whichtomove,nwpts):
    iwpts = dxy.names.index(WPTS)
    dims = list(dxy.shape)
    dims[iwpts]=nwpts
    res = torch.zeros(dims,device=dxy.device,dtype=dxy.dtype)
    slicewpt = (slice(None),)* iwpts + (slice(*whichtomove),)
    res[slicewpt]=dxy.rename(None)
    return res.rename(*dxy.names)


def processnames(dtnames,xnames):
    return dtnames+tuple(x for x in xnames if x not in dtnames)

# def adddt_old(dt,whichwpts,whichtomove,f):#xy0,v,theta,duration,turn_rate)
#     vhead = vheading(f.theta)
#     meanv = f.meanv()
#     assert(vhead.names[-2]==WPTS)
#     dxy = vhead[...,whichwpts:whichwpts+1,:] * meanv.align_as(vhead)[...,whichwpts:whichwpts+1,:]
#     # print(f.duration)
#     newnames = processnames(dt.names, dxy.names)
#     assert((f.duration.align_to(*newnames)+dt.align_to(*newnames)).min()>=0.)
#     dxy = dxy.align_to(*newnames) * dt.align_to(*newnames)
#     dxy = zero_pad_old(dxy,whichtomove,f.nwpts())
#     # wpts = f.compute_wpts()
#     dparams = applydxy(f,dxy)
#     return f.from_wpts(**dparams)




# zero_pad on [:wps_start+1] and [wps_end:]
def zero_pad(dxy,wpts_start,wpts_end):
    bshape = named.broadcastshapes(dxy.shape,wpts_start.align_as(dxy).shape)
    bshape = named.broadcastshapes(bshape,wpts_end.align_as(dxy).shape)
    # print(bshape)
    # raise Exception
    dxy = dxy.broadcast_to(bshape).clone()
    # dxy = dxy.clone()
    iwpts = dxy.names.index(WPTS)
    nwpts = dxy.shape[iwpts]
    i = torch.arange(nwpts,device=dxy.device).rename(WPTS)#.align_as(dxy)
    # print(i.names,i.shape)
    # print(wpts_start.names,wpts_start.shape)
    # print(dxy.names,dxy.shape)
    # print(i.align_as(dxy).names,i.align_as(dxy).shape)
    # print(wpts_start.align_as(dxy).names,wpts_start.align_as(dxy).shape)
    # print((i.align_as(dxy) <= wpts_start.align_as(dxy)).names, (i.align_as(dxy) <= wpts_start.align_as(dxy)).shape)
    mask_start = (i.align_as(dxy) <= wpts_start.align_as(dxy)).broadcast_to(dxy.shape)
    mask_turn = (i.align_as(dxy) >= wpts_end.align_as(dxy)).broadcast_to(dxy.shape)
    dxynames = dxy.names
    dxy = dxy.rename(None)
    dxy[mask_start.rename(None)]=0
    # dxy = torch.cumsum(dxy,axis=-2)
    # dxy[...,wpts_turn:,:]=0
    dxy[mask_turn.rename(None)]=0
    return dxy.rename(*dxynames)

def applydxy(f,dxy):
    wpts = f.compute_wpts()
    assert(dxy.names[-1]==XY)
    assert(dxy.names[-2]==WPTS)
    basename = dxy.names[:-2]
    dparams = {
        "wpts": wpts.align_as(dxy) + dxy,
        "xy0":f.xy0.align_to(*basename,XY),
        "v":f.v.align_to(*basename,WPTS),
        "turn_rate":f.turn_rate.align_to(*basename,WPTS)
    }
    return dparams


def adddt(dt,wpts_start,wpts_end,f):
    '''
    wpts_start>0
    wpts_start-1 unchanged
    wpts_start is the shifted wpts
    wpts_end is the last wpts to be shifted
    wpts_end+1 unchanged
    '''
    argstocheck = (dt,wpts_start,wpts_end)
    basename = compute_basename(f,*argstocheck)
    vhead = vheading(f.theta)
    meanv = f.meanv()
    assert(vhead.names[-2]==WPTS)
    assert(wpts_start.min()>0)
    wpts_start = wpts_start - 1
    dxy = (vhead * meanv.align_to(...,XY)).align_to(*basename,WPTS,XY)
    dxy = dxy * dt.align_as(dxy)
    dxy = zero_pad(dxy,wpts_start-1,wpts_start+1)
    # print(dxy)
    # raise Exception
    dxy = torch.cumsum(dxy,axis=-2)
    dxy = zero_pad(dxy,wpts_start-1,wpts_end)
    dparams = applydxy(f,dxy)
    return f.from_wpts(**dparams)



def changespeed(dspeed,wpts_start,wpts_end,f):
    '''
    wpts_start>0
    wpts_start-1 unchanged
    wpts_start is the shifted wpts
    wpts_end is the last wpts to be shifted
    wpts_end+1 unchanged
    '''
    assert(f.v.names[-1] == WPTS)
    argstocheck = (dspeed,wpts_start,wpts_end)
    basename = compute_basename(f,*argstocheck)
    nwpts = f.v.shape[-1]
    wpts = f.compute_wpts()
    meanv = f.meanv()
    vhead = vheading(f.theta).align_to(*basename,WPTS,XY)
    velocity = vhead * meanv.align_as(vhead)
    wpts_start = wpts_start - 1
    newmeanv = (dspeed.align_as(velocity)-1)*velocity
    dxy = newmeanv * f.duration.align_as(newmeanv)
    assert(dxy.names[-2] == WPTS)
    dxy = zero_pad(dxy,wpts_start-1,wpts_start+1)
    dxy = torch.cumsum(dxy,axis=-2)
    dxy = zero_pad(dxy,wpts_start-1,wpts_end)
    dspeed = zero_pad(dspeed.align_to(*basename,WPTS),wpts_start-1,wpts_end)
    dparams=applydxy(f, dxy)
    dparams["v"]=f.v.align_to(*basename,WPTS)*(1+dspeed.align_to(*basename,WPTS))#.align_as(*-1,wpts_start-1,wpts_end)

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

def changespeed_old(dspeed,whichtomove,f):
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


def addangle_old(dtheta,whichwpts,whichtomove,f):
    assert(f.theta.names[-1]==WPTS)
    newnames = processnames(dtheta.names,f.theta.names)
    newtheta = f.theta.align_to(*newnames) + dtheta.align_to(*newnames)
    newvheading = vheading(newtheta)
    dxytheta = newvheading - vheading(f.theta).align_as(newvheading)
    newnames = processnames(dtheta.names,dxytheta.names)
    dxy = dxytheta.align_to(*newnames) * f.meanv().align_to(*newnames) * f.duration.align_to(*newnames)
    assert(dxy.names[-2]==WPTS)
    dxy = zero_pad_old(dxy[...,whichwpts:whichwpts+1,:],whichtomove,f.nwpts())
    # raise Exception
    dparams=applydxy(f, dtheta.names, dxy)
    return f.from_wpts(**dparams)


def gather_wpts(t,iwpts):
    assert(WPTS not in iwpts.names)
    assert(WPTS in t.names)
    # for x in iwpts.names:
    #     assert(x in t.names)
    assert(iwpts.min()>=0)
    gathered_t = named.gather(t,WPTS,iwpts.align_to(*(iwpts.names+(WPTS,))))
    return gathered_t.align_to(...,WPTS).squeeze(-1)


def check_same(a,b):
    assert(b.shape==a.shape)
    assert(b.names==a.names)


def check_param_names(v):
    assert(XY not in v.names)
    assert(WPTS not in v.names)
    assert(T not in v.names)
    assert(None not in v.names)

def compute_basename(f,*args):# merge all but XY, WPTS and T axis
    for v in args:
        check_param_names(v)
    newnames = named.mergenames((f.theta.names[:-1],)+ tuple(x.names for x in args))
    return tuple(sorted(newnames))

def addangle(dtheta,wpts_start,wpts_turn,wpts_rejoin,f):
    '''
    wpts_start>=0
    wpts_start unchanged
    wpts_rejoin unchanged
    '''
    argstocheck = (dtheta,wpts_start,wpts_turn,wpts_rejoin)
    basename = compute_basename(f,*argstocheck)#named.mergenames((f.theta.names[:-1],)+ tuple(x.names for x in argstocheck))

    wpts_turn = wpts_turn -1
    wpts_rejoin = wpts_rejoin -1
    wpts_start = wpts_start -1
    newf = addangle_no_rejoin(dtheta,wpts_start,wpts_rejoin,f)

    def compute_theta_turn_rejoin(f1,f2):
        turn = gather_wpts(f1.compute_wpts(),wpts_turn)
        rejoin = gather_wpts(f2.compute_wpts(),wpts_rejoin)
        diff = (rejoin-turn).align_to(...,XY)
        return torch.atan2(diff[...,1],diff[...,0]).align_to(*basename)
    newtheta = compute_theta_turn_rejoin(newf,newf)
    print(f"{newtheta.names=}  {newtheta.shape=}")
    # raise Exception
    # print(f"{newtheta.names=} {newtheta.shape=}")
    # print(f"{dtheta.names=} {dtheta.shape=}")
    oldtheta = compute_theta_turn_rejoin(f,f) + dtheta.align_to(*basename)

    # newf = addangle_no_rejoin(dtheta,wpts_start,wpts_rejoin,f)
    # print(newtheta)
    # print(oldtheta)
    # print(newtheta-oldtheta)
    return addangle_no_rejoin(newtheta-oldtheta,wpts_turn,wpts_rejoin,newf)

# def make_consistent(dparams):
#     names = set(named.mergenames(tuple(x.names for x in dparams.values())))
#     names = tuple(names.difference({XY,WPTS}))
#     print(names)
#     print(dparams)
#     for  v in ["wpts"]:
#         dparams[v] = dparams[v].align_to(*names,WPTS,XY)
#     for v in ["v","turn_rate",]:
#         dparams[v] = dparams[v].align_to(*names,WPTS)
#     print(f'{dparams["xy0"].names=}')
#     for v in ["xy0",]:
#         dparams[v] = dparams[v].align_to(*names,XY)
#     return dparams


def addangle_no_rejoin(dtheta,wpts_start,wpts_turn,f):
    basename = compute_basename(f,dtheta,wpts_start,wpts_turn)
    newtheta = f.theta.align_to(*basename,WPTS) + dtheta.align_to(*basename,WPTS)
    newvheading = vheading(newtheta)
    dxytheta = newvheading - vheading(f.theta).align_as(newvheading)
    assert(dxytheta.names[-2]==WPTS)
    assert(dxytheta.names[-1]==XY)
    dxy = dxytheta * f.meanv().align_as(dxytheta) * f.duration.align_as(dxytheta)
    # print(dxy.shape,)
    # raise Exception
    dxy = zero_pad(dxy,wpts_start,wpts_turn)
    dxy = torch.cumsum(dxy,axis=-2)
    dxy = zero_pad(dxy,wpts_start,wpts_turn)
    dparams=applydxy(f, dxy)
    # dparams=make_consistent(applydxy(f, dtheta.names, dxy))
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


