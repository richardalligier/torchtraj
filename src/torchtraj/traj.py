__all__ = ["generate"]

import torch
import sys

from . import named
from .utils import WPTS,T,XY
from .flights import vheading


# THETA = "theta"
# SPEED = "speed"
# DURATION = "duration"
# TURN_RATE = "turn_rate"

#from functorch.dim import dims


class CannotTurnError(Exception):
    pass

eps=1e-12

# fpl = (SPEED,THETA,DURATION,TURN_RATE)
def distance_between_turn_and_wpt(dtheta, radius):
    return torch.tan(torch.abs(dtheta)*0.5) * radius

def time_between_endseg_to_beginturn(distance,meanv):
    return named.pad(distance/ meanv[...,:-1],(0,1),'constant',0)
def time_between_endturn_to_beginseg(distance,meanv):
    return named.pad(distance/ meanv[...,1:],(1,0),'constant',0)
def segment_xy(f, t, tstart, v, theta, duration):
    # duration = named.unsqueeze(duration,-1,T)#.align_as(used_t)
    tstart = named.unsqueeze(tstart,-1,T)
    clipped_t = t.align_as(tstart)-tstart
    # vx = v * torch.cos(theta)
    # vy = v * torch.sin(theta)
    # cx = torch.cos(theta)
    # cy = torch.sin(theta)
    c = vheading(theta)#torch.cat([named.unsqueeze(vd,-1,XY) for vd in (cx,cy)],-1)
    # vxy = torch.cat([named.unsqueeze(vd,-1,XY) for vd in (vx,vy)],-1)
    # vxy = named.unsqueeze(vxy,-2,T)
    # print(clipped_t.shape,duration.shape,tstart.shape)
    # duration = named.unsqueeze(duration,-1,T)#.align_as(used_t)
    clipped_t.clip_(min=torch.zeros_like(clipped_t),max=named.unsqueeze(duration,-1,T))#duration)
    named.unsqueeze_(clipped_t,-1,XY)
    # xy = v.align_as(clipped_t) * c.align_as(clipped_t) * clipped_t #named.unsqueeze(clipped_t,-1,XY)
    xy = f.segdist(clipped_t,duration) * c.align_as(clipped_t)
    # print(xy.shape,sys.getsizeof(xy.storage())/1024/1024)
    # print(clipped_t.shape,sys.getsizeof(clipped_t.storage())/1024/1024)
    return torch.sum(xy,axis=xy.names.index(WPTS))

# class SegmentWithAcc(Segment):
#     @staticmethod
#     def speed_at_turns(v):
#         return v[...,1:]
#     @staticmethod
#     def meanv(v):
#         assert(v.names[-1]==WPTS)
#         v = named.pad(v,(0,1),'replicate')
#         return (v[...,:-1] + v[...,1:]) * 0.5
#     # @staticmethod
#     # def time_between_endseg_to_beginturn(distance,meanv):
#     #     return named.pad(distance/ meanv[...,:-1],(0,1),'constant',0)
#     # @staticmethod
#     # def time_between_endturn_to_beginseg(distance,meanv):
#     #     return named.pad(distance/ meanv[...,1:],(1,0),'constant',0)
#     @staticmethod
#     def xy(t, tstart, v,theta, duration):
#         assert(v.names[-1]==WPTS)
#         tstart = named.unsqueeze(tstart,-1,T)
#         v0 = v[...,0]
#         acc = named.pad(torch.diff(v,axis=-1),(0,1),'constant',0)/duration
#         # print(acc)
#         duration = named.unsqueeze(duration,-1,T)#.align_as(used_t)
#         # vstart = v.clone()
#         # vend = v.clone()
#         # vend[...,:-1]=vend[...,1:]
#         clipped_t = t.align_as(tstart)-tstart
#         # print(clipped_t.shape,duration.shape,tstart.shape)
#         # duration = named.unsqueeze(duration,-1,T)#.align_as(used_t)
#         clipped_t.clip_(min=torch.zeros_like(duration),max=duration)
#         print("clipped_t.names",clipped_t.names)
#         # named.unsqueeze_(clipped_t,-1,XY)
#         print(clipped_t.shape,clipped_t.names)
#         print(duration.shape,duration.names)
#         cx = torch.cos(theta)
#         cy = torch.sin(theta)
#         c = torch.cat([named.unsqueeze(vd,-1,XY) for vd in (cx,cy)],-1)
#         named.unsqueeze_(clipped_t,-1,XY)
#         print("clipped_t.names",clipped_t.names)
#         xy = c.align_as(clipped_t) * clipped_t * (v.align_as(clipped_t) + acc.align_as(clipped_t) * 0.5 * clipped_t)
#         return torch.sum(xy,axis=xy.names.index(WPTS))

def segment_xy_lowmemory(f, t, tstart, v, theta, duration):#t, tstart, vxy, duration):
    assert(duration.names[-1]==WPTS)
    assert(tstart.names[-1]==WPTS)
    vxy = v.align_as() * vheading(theta)
    duration = named.unsqueeze(duration,-1,T)#.align_as(used_t)
    tstart = named.unsqueeze(tstart,-1,T)
    def clipped_t_wpts(i):
        # print(t.shape,tstart.shape)
        tstarti = tstart[...,i,:]
        a = t.align_as(tstarti)-tstarti
        v = duration[...,i,:]
        a.clip_(min=torch.zeros_like(v),max=v)
        named.unsqueeze_(a,-1,XY)
        return a
    # clipped_t = t.align_as(tstart)-tstart[...,:,i]
    # print(clipped_t.shape,duration.shape,tstart.shape)
    # duration = named.unsqueeze(duration,-1,T)#.align_as(used_t)
    # vxy = named.unsqueeze(vxy,-2,T)
    # named.unsqueeze_(clipped_t,-1,XY)
    # xy = clipped_t
    # xy.mul_(vxy)
    print("vxy.names",vxy.names)
    xy = vxy[...,0,:,:] * clipped_t_wpts(0)#named.unsqueeze(clipped_t,-1,XY)
    for i in range(1,duration.shape[-2]):
        xy.add_(vxy[...,i,:,:] * clipped_t_wpts(i))
    print(xy.shape,sys.getsizeof(xy.storage())/1024/1024)
    # print(clipped_t.shape,sys.getsizeof(clipped_t.storage())/1024/1024)
    return xy #torch.sum(xy,axis=xy.names.index(WPTS))

def generate(f,t):
    assert(t.min()>=0)
    return generate_gen(f,t,segment_xy)
def generate_lowmem(f,t):
    return generate_gen(f,t,segment_xy_lowmemory)

def generate_gen(f,t,seg_xy):
    # assert(len(t.shape)==1)
    # wptsd = dims(1)
    # td = dims(1)
    # xyd = dims(1)
    # t = t[td]
    # duration = f.duration[...,wptsd]
    # theta = f.theta[...,wptsd]
    # v = f.v[...,wptsd]
    # turn_rate = f.turn_rate[...,wptsd]
    # # acceleration = acceleration.align_as(duration)
    # xy0 = f.xy0[xyd]

    duration = f.duration.align_to(...,WPTS)
    theta = f.theta.align_as(duration)
    v = f.v.align_as(duration)
    turn_rate = f.turn_rate.align_as(duration)
    # acceleration = acceleration.align_as(duration)
    xy0names = duration.names[:-1]+(XY,)
    xy0 = f.xy0.align_to(*xy0names)


    tstart = torch.nn.functional.pad(duration[...,:-1].rename(None), (1,0), 'constant', 0)
    tstart = torch.cumsum(tstart, axis=-1).rename(*duration.names)
    # tstart = named.unsqueeze(tstart,-1,T)
    # duration = named.unsqueeze(duration,-1,T)#.align_as(used_t)
    xy = seg_xy(f,t,tstart,v,theta,duration)
    # torch.cuda.empty_cache()
    xy.add_(xy0.align_as(xy))
    return  xy

def normalizediffangle(dangle):
    pi2 = 2*torch.pi
    return dangle + pi2 * (dangle < -torch.pi) - pi2 * (dangle >= torch.pi)


def turns(t,tstart,radius,theta,dtheta,duration,turn_rate):
    duration = named.unsqueeze(duration,-1,T)#.align_as(used_t)
    tstart = named.unsqueeze(tstart,-1,T)
    clipped_t = t.align_as(tstart)-tstart
    clipped_t.clip_(min=torch.zeros_like(duration),max=duration)
    signdtheta = torch.sign(dtheta)
    thetacenter = theta - torch.pi * 0.5 * signdtheta
    thetat = thetacenter.align_as(clipped_t) + (turn_rate* signdtheta).align_as(clipped_t) * clipped_t
    # print(thetacenter)
    x = radius.align_as(thetat) * (torch.cos(thetat) - torch.cos(thetacenter).align_as(thetat))
    y = radius.align_as(thetat) * (torch.sin(thetat) - torch.sin(thetacenter).align_as(thetat))
    xy = torch.cat([named.unsqueeze(vd,-1,XY) for vd in (x,y)],-1)
    # print("turnxy")
    # print(xy[0,:,19:25,0])
    return torch.sum(xy,axis=xy.names.index(WPTS))

# def turns_lowmemory(t,tstart,radius,theta,dtheta,duration,turn_rate):
#     duration = named.unsqueeze(duration,-1,T)#.align_as(used_t)
#     tstart = named.unsqueeze(tstart,-1,T)
#     clipped_t = t.align_as(tstart)-tstart
#     clipped_t.clip_(min=torch.zeros_like(duration),max=duration)
#     signdtheta = torch.sign(dtheta)
#     thetacenter = theta - torch.pi * 0.5 * signdtheta
#     thetat = thetacenter.align_as(clipped_t) + (turn_rate * signdtheta).align_as(clipped_t) * clipped_t
#     x = radius.align_as(thetat) * (torch.cos(thetat) - torch.cos(thetacenter).align_as(thetat))
#     y = radius.align_as(thetat) * (torch.sin(thetat) - torch.sin(thetacenter).align_as(thetat))
#     xy = torch.cat([named.unsqueeze(vd,-1,XY) for vd in (x,y)],-1)
#     return torch.sum(xy,axis=xy.names.index(WPTS))

# @torch.compile
def generatewithturn(f,t):
    # assert(len(t.shape)==1)
    duration = f.duration.align_to(...,WPTS)
    theta = normalizediffangle(f.theta).align_as(duration)
    v = f.v.align_as(duration)
    turn_rate = f.turn_rate.align_as(duration)
    # acceleration = acceleration.align_as(duration)
    xy0names = duration.names[:-1]+(XY,)
    xy0 = f.xy0.align_to(*xy0names)

    tstart = torch.nn.functional.pad(duration[...,:-1].rename(None),(1,0),'constant',0)
    tstart = torch.cumsum(tstart,axis=-1).rename(*duration.names)
    # tstart.cumsum_(axis=-1).rename_(*duration.names)
    # vx = v * torch.cos(theta)
    # vy = v * torch.sin(theta)
    # vxy = torch.cat([named.unsqueeze(vd,-1,XY) for vd in (vx,vy)],-1)
    # vxy = named.unsqueeze(vxy,-2,T)
    radius = f.speed_at_turns() / turn_rate
    dtheta = torch.diff(theta.rename(None)).rename(*theta.names)# torch.nn.functional.pad(,(1,0),'constant',0).rename(*theta.names)
    dtheta = normalizediffangle(dtheta)
    dturn_wpt = distance_between_turn_and_wpt(dtheta,radius)#torch.tan(torch.abs(dtheta)*0.5) * radius
    # print("dtheta",dtheta)
    # print("decal",decal)
    # print("radius",radius)
    # raise Exception
    meanv = f.meanv()
    durationsegs = duration - time_between_endseg_to_beginturn(dturn_wpt,meanv) - time_between_endturn_to_beginseg(dturn_wpt,meanv)
    # print(durationsegs.min())
    durationsegsmin = durationsegs.min()
    if durationsegsmin<0:
        raise CannotTurnError(durationsegs<0.)
    durationturns = torch.abs(dtheta) / turn_rate
    assert(durationturns.min()>=0)
    tstartsegs = named.pad(torch.cumsum(durationsegs[...,:-1]+durationturns,axis=-1),(1,0),'constant',0)#tstart - named.pad(decal,(1,0),'constant',0) / v + named.pad(durationturns,(1,0),'constant',0)
    # raise Exception
    tstartturns = (tstartsegs + durationsegs)[...,:-1]
    # print(tstartsegs)
    # # print((tstartsegs+durationsegs))
    # print(tstartturns)
    # print(tstartturns+durationturns)

    # assert(durationsegs.min()>=0)
    # assert(tstartsegs.min()>=0)
    xy = segment_xy(f,t,tstartsegs,v,theta,durationsegs)
    # print(xysegments)
    # print("theta",theta)
    xy.add_(turns(t,tstartturns,radius,theta[...,:-1],dtheta,durationturns,turn_rate))
    xy.add_(xy0.align_as(xy))
    # return xyturns
    # torch.cuda.empty_cache()
    return  xy#0.align_as(xysegments) + xysegments + xyturns


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import time
    import sys
    from flights import Flights,FlightsWithAcc
    import numpy as np
    def plot(xy):
        t = torch.arange(xy[0,:,1].cpu().numpy().shape[0]-1)
        # plt.plot(t,np.diff(xy[0,:,0].cpu().numpy()))
        # plt.scatter(t,np.diff(xy[0,:,0].cpu().numpy()))

        # tf = torch.arange(xy[0,:,1].cpu().numpy().shape[0])
        # plt.plot(tf,(xy[0,:,0].cpu().numpy()))
        # plt.scatter(tf,(xy[0,:,0].cpu().numpy()))

        # plt.plot(t,np.diff(xy[0,:,1].cpu().numpy()))
        # plt.scatter(t,np.diff(xy[0,:,1].cpu().numpy()))
        # dx = np.diff(xy[0,:,0].cpu().numpy())
        # dy = np.diff(xy[0,:,1].cpu().numpy())
        # plt.plot(t,np.hypot(dy,dx))
        # plt.scatter(t,np.hypot(dy,dx))

        plt.plot(xy[0,:,0].cpu().numpy(),xy[0,:,1].cpu().numpy())
        plt.scatter(xy[0,:,0].cpu().numpy(),xy[0,:,1].cpu().numpy())
        plt.axis('equal')
    def main():
        torch.random.manual_seed(3)
        device = torch.device("cpu")#"cuda" if torch.cuda.is_available() else "cpu")
        print(device)
        n = 30000
        nwpts = 10
        BATCH = "batch"
        namesbatch = (BATCH,)
        names = namesbatch + (WPTS,)
        shape = (n, nwpts)
        shapeturn_rate = (n,nwpts-1)
        xy0 = torch.zeros(shape[:-1]+(2,),names=namesbatch+(XY,),device=device)
        v = 10+torch.rand(shape,names=names,device=device)*10
        dtheta = (torch.rand(shape,names=names,device=device)-0.5)*(1.5*torch.pi)
        theta = torch.cumsum(dtheta,axis=-1)
        duration = 20+torch.rand(shape,names=names,device=device)*30
        # turn_rate = torch.ones(shapeturn_rate,names=names,device=device)*1e0
        turn_rate = torch.ones(duration.shape[:-1]+(nwpts-1,),names=names,device=device)*1e0
        # acceleration = torch.ones(shape,names=names,device=device) * 1e12
        t = torch.stack([torch.linspace(0,200,1000,device=device),]*n).rename(BATCH,T)
        # t = torch.linspace(0,200,1000,device=device).rename(T)
        # print(t)
        # raise Exception
        f = Flights(xy0,v,theta,duration,turn_rate)
        for _ in range(2):
            t0 = time.perf_counter()
            xys = generate(f,  t)#.cpu()
            torch.cuda.synchronize()
            print(time.perf_counter()-t0)
            t0 = time.perf_counter()
            xy = generatewithturn(f,  t)#.cpu()
            torch.cuda.synchronize()
            print(time.perf_counter()-t0)
            print(xys.dtype)
        torch.cuda.empty_cache()
        print(xy.shape,sys.getsizeof(xy.storage())/1024/1024)
        print(torch.cuda.max_memory_allocated()/1024/1024)

        print(xy)
        tother = torch.linspace(0,200,1000,device=device).rename(T)[400:-100]
        xyo = generate(f,tother)
        # print(xy.shape,xy.names)
        plot(xy)
        plot(xys)
        #plot(xyo)

        plt.show()
    # torch.cuda.memory._record_memory_history()
    # def heart():
    #     device = torch.device("cpu")#cuda" if torch.cuda.is_available() else "cpu")
    #     print(device)
    #     nwpts = 2
    #     namesbatch = ("batch",)
    #     names = namesbatch + (WPTS,)
    #     c = np.radians(45)
    #     b = 30
    #     d = 8.85
    #     duration = torch.tensor([[b,d,d,d,d,d,d,d,d,b]],names=names,device=device)
    #     dtheta = torch.tensor([[c,-c,-c,-c,-c,-c+4*c-c,-c,-c,-c,-c]],names=names,device=device)
    #     # duration = torch.cat((duration,torch.flip(duration.rename(None),dims=(-1,))),dim=-1)
    #     # dtheta =  torch.cat((dtheta,-torch.flip(dtheta.rename(None),dims=(-1,))),dim=-1)
    #     n = 1
    #     nwpts = dtheta.shape[-1]
    #     shape = (n, nwpts)
    #     shapeturn_rate = (n, nwpts-1)
    #     theta = torch.pi/2+torch.cumsum(dtheta,axis=-1)
    #     xy0 = torch.zeros(shape[:-1]+(2,),names=namesbatch+(XY,),device=device)
    #     v = 10+torch.ones(shape,names=names,device=device)*10
    #     turn_rate = torch.ones(shapeturn_rate,names=names,device=device)*1e-1
    #     turn_rate[0,turn_rate.shape[-1]//2]=2e-1
    #     turn_rate[0,turn_rate.shape[-1]//2-1]=2e-1
    #     turn_rate[0,turn_rate.shape[-1]//2+1]=2e-1
    #     # acceleration = torch.ones(shape,names=names,device=device) * 1e12
    #     t = torch.linspace(0,200,1000,device=device).rename(T)
    #     f = Flights(xy0,v,theta,duration,turn_rate)
    #     for _ in range(2):
    #         t0 = time.perf_counter()
    #         xys = generate(SegmentWithAcc,f,  t)#.cpu()
    #         torch.cuda.synchronize()
    #         print(time.perf_counter()-t0)
    #         t0 = time.perf_counter()
    #         xy = generatewithturn(SegmentWithAcc,f,  t)#.cpu()
    #         torch.cuda.synchronize()
    #         print(time.perf_counter()-t0)
    #         print(xys.dtype)
    #     torch.cuda.empty_cache()
    #     print(xy.shape,sys.getsizeof(xy.storage())/1024/1024)
    #     print(torch.cuda.max_memory_allocated()/1024/1024)
    #     print(xy)
    #     # print(xy.shape,xy.names)
    #     plot(xy)
    #     plot(xys)
    #     plt.show()
    with torch.no_grad():
        main()
    # torch.cuda.memory._dump_snapshot("my_snapshot.pickle")


