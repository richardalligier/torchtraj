import torch
import named
from qhull import XY
# THETA = "theta"
# SPEED = "speed"
# DURATION = "duration"
# TURN_RATE = "turn_rate"
WPTS ="wpts"
T = "t"
eps=1e-12

# fpl = (SPEED,THETA,DURATION,TURN_RATE)

def values_at_t(values, difft, dvalues_dt):
    assert(values.names[-1]==WPTS)
    assert(difft.names[-2]==WPTS)
    assert(dvalues_dt.names[-1]==WPTS)
    dvalues = torch.nn.functional.pad(torch.diff(values.rename(None)),(1,0),'constant',0)
    # print("values.names",values.names)
    dvalues=dvalues.rename(*values.names)
    # print("dvalues.names",dvalues.names)
    duration = torch.abs(dvalues) / dvalues_dt
    assert duration.min() >= 0.
    valuesact = torch.clamp(difft / named.unsqueeze((duration+eps),-1,T), min=0, max=1)
    # print(valuesact.names)
    # print(dvalues.names)
    # print("values",values)
    res = dvalues.align_as(valuesact) * valuesact
    # print("res",res.names)
    # print("res",res)
    return values[...,:1].rename(**{WPTS:T}) + torch.sum(res ,axis=-2)

def generate(xy0,v,theta,duration,turn_rate,acceleration,t):
    assert(len(t.shape)==1)
    duration=duration.align_to(...,WPTS)
    theta = theta.align_as(duration)
    v = v.align_as(duration)
    turn_rate = turn_rate.align_as(duration)
    acceleration = acceleration.align_as(duration)
    xy0names = duration.names[:-1]+(XY,)
    xy0 = xy0.align_to(*xy0names)

    tturn = torch.cumsum(duration,axis=-1)
    # print(tturn.shape,tturn.names)
    # print(t.shape)
    difft = t-named.unsqueeze(tturn,-1, T)
    # print(difft.names)
    # print(tturn)
    # print(difft)
    thetat = values_at_t(theta, difft, turn_rate)
    # print(theta)
    # print(thetat)
    # raise Exception
    vt =  values_at_t(v, difft, acceleration)

    # dv = torch.nn.functional.pad(torch.diff(theta),(1,0),'constant',0)
    # print(v)
    # print(vt)
    vx = vt * torch.cos(thetat)
    vy = vt * torch.sin(thetat)
    xy = torch.cat([named.unsqueeze(vd,-1,XY) for vd in (vx,vy)],-1)
    return  torch.cumsum(xy, axis=-2)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import time
    def main():
        torch.random.manual_seed(1)
        n = 10000
        namesbatch = ("batch",)
        names = namesbatch + (WPTS,)
        shape = (n, 3)
        xy0 = torch.zeros(shape[:-1]+(2,),names=namesbatch+(XY,))
        v = torch.rand(shape,names=names)
        theta = torch.rand(shape,names=names)*10
        duration = torch.rand(shape,names=names)*10
        turn_rate = torch.ones(shape,names=names)*1e1
        acceleration = torch.ones(shape,names=names) * 1e12
        t = torch.arange(100)
        t0 = time.perf_counter()
        xy = generate(xy0, v, theta, duration, turn_rate, acceleration, t)
        print(time.perf_counter()-t0)
        print(xy)
        print(xy.shape,xy.names)
        plt.scatter(xy[0,:,0],xy[0,:,1])
        plt.show()
    main()


