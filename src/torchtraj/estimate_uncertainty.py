from torchtraj.utils import WPTS, XY,T
from torchtraj.flights import Flights, FlightsWithAcc
from torchtraj.uncertainty import PARAMS, DT0, DT1, DANGLE, DSPEED, adddt_translate,adddt_rotate, addangle,  addlongitudinaldspeed
from torchtraj.uncertainty import changespeed_old as changespeed
import torch
# from torchtraj.qhull import QhullDist
from torchtraj import uncertainty, named, traj

import matplotlib.pyplot as plt


def generatehulls_maneuvered(flights, t, dt0, dt1, dangle, dspeed):
    flights = adddt_translate(dt0,0,(0,2),flights)
    flights = adddt_rotate(dt1,1,(1,2),flights)
    flights = addangle(dangle,1,(1,2),flights)
    flights = changespeed(dspeed,(0,2),flights)
    return traj.generate(flights, t)

def generatehulls_notmaneuvered(flights, t, dspeed):
    flights = addlongitudinaldspeed(dspeed,flights)
    return traj.generate(flights, t)


def compute_distance(qhulldist, generatehull1, generatehull2, dt0, dt1, dangle, dspeed,capmem=None):
    xy1 = generatehull1(dt0, dt1, dangle, dspeed)
    # print("xy1.shape",xy1.shape)
    xy2 = generatehull2(dt0, dt1, dangle, dspeed)
    # print("xy2.shape",xy2.shape)
    return named.amin(qhulldist.dist(xy1,xy2,dimsInSet=(DT0,DT1,DANGLE,DSPEED),capmem=capmem),dim=(T,))


def create_random(n,device,nwpts=4):
    namesbatch = ("batch",)
    names = namesbatch + (WPTS,)
    shape = (n, nwpts)
    shapeturn_rate = (n, nwpts-1)
    xy0 = torch.randn(shape[:-1]+(2,),names=namesbatch+(XY,),device=device) * 100
    v = 10 + torch.rand(shape,names=names,device=device)*10
    dtheta = (torch.rand(shape,names=names,device=device)-0.5)*(1.5*torch.pi)
    theta = torch.cumsum(dtheta,axis=-1)
    duration = 20 + torch.rand(shape,names=names,device=device)*30
    turn_rate = torch.ones(shapeturn_rate,names=names,device=device)*1e0
    return FlightsWithAcc(xy0,v,theta,duration,turn_rate)

def main():
    import numpy as np
    from torchtraj.qhull import QhullDist
    torch.random.manual_seed(0)
    n=10
    device = torch.device("cpu")#"cuda" if torch.cuda.is_available() else "cpu")
    flights = create_random(n,device)
    flights_others = create_random(n,device)
    t = torch.linspace(0.,100.,int(101),device = flights.device(),dtype=flights.dtype()).rename(T)
    xy_others = traj.generate(flights_others, t)

    l = []
    ##### Both maneuvered, both have uncertainty ####
    def gh1(dt0, dt1, dangle, dspeed):
        return generatehulls_maneuvered(flights,t,dt0, dt1, dangle, dspeed)

    def gh2(dt0, dt1, dangle, dspeed):
        return generatehulls_maneuvered(flights_others,t,dt0, dt1, dangle, dspeed)
    l.append(("both with maneuver uncertainty",gh1,gh2))
    ##### Only one maneuvered, both have uncertainty ####
    def gh1(dt0, dt1, dangle, dspeed):
        return generatehulls_maneuvered(flights,t,dt0, dt1, dangle, dspeed)

    def gh2(dt0, dt1, dangle, dspeed):
        return generatehulls_notmaneuvered(flights_others,t,dspeed)
    l.append(("only one with maneuver uncertainty, the other longitudinal speed uncertainty",gh1,gh2))
    ##### Only one maneuvered, one have uncertainty, the other is just a serie of points ####
    def gh1(dt0, dt1, dangle, dspeed):
        return generatehulls_maneuvered(flights,t,dt0, dt1, dangle, dspeed)

    def gh2(dt0, dt1, dangle, dspeed):
        return xy_others
    l.append(("only one with maneuver uncertainty, the other is a series of points",gh1,gh2))
    #### params ###
    ea = 3
    et0 = 10.
    et1 = 10.
    espeed = 0.1
    dt0 = torch.tensor([[0.,et0],[0.,0.]],names=(PARAMS,DT0,),device=device)
    dt1 = torch.tensor([[0.,et1],[0.,0.]],names=(PARAMS,DT1,),device=device)
    dangle = torch.tensor([[-np.radians(ea),np.radians(ea)],[0.,0.]],names=(PARAMS,DANGLE,),device=device)
    dspeed = torch.tensor([[1.+espeed,1.-espeed],[1.,1.]],names=(PARAMS,DSPEED,),device=device)
    #### initialise distance computers ###
    qhulldist = QhullDist(device,n=180)

    ### computing distances for all cases###
    for (method, gh1, gh2) in l:
        print(method)
        print(compute_distance(qhulldist, gh1, gh2, dt0, dt1, dangle, dspeed))

if __name__ == '__main__':
    main()
