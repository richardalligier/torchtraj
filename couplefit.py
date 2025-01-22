import pandas as pd
import matplotlib.pyplot as plt
import tqdm
import datetime
from os import listdir
from os.path import isfile, join

from utils import WPTS, T, XY
from flights import FlightsWithAcc
import torch
import fit
import numpy as np
import traj
import math
import pytz

INFOLDER = "conflicts"
OUTFOLDER = "modeledconflicts"


def yourmodel(xy0,v,theta,duration,nwpts):
    namesbatch = tuple()
    names = namesbatch + (WPTS,)
    shape = (nwpts,)
    shapeturn_rate = (nwpts-1,)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    xy0 = torch.tensor(xy0,device=device,names=namesbatch+(XY,))
    v = torch.tensor(v,device=device,names=names)
    theta = torch.tensor(theta,device=device,names=names)
    duration = torch.tensor(duration,device=device,names=names)
    turn_rate = torch.ones(shapeturn_rate,names=names,device=device)*1e0
    t = torch.tensor(list(range(int(sum(duration))+1)),device=device,names=(T,))
    return(t,FlightsWithAcc(xy0,v,theta,duration,turn_rate),device)


# def initfmodel(flight,device,dtype,nwpts=4):
#     namesbatch = tuple()
#     names = namesbatch + (WPTS,)
#     shape = (nwpts,)
#     shapeturn_rate = (nwpts-1,)
#     xy0 = torch.tensor([float(x) for x in [flight.longitude.values[0],flight.latitude.values[0]]],device=device,names=namesbatch+(XY,),dtype=dtype)
#     v = torch.tensor([flight.groundspeed.values[int(i)] for i in np.linspace(0,flight.shape[0]-1,nwpts)],device=device,names=names,dtype=dtype)/3600/60*1.1
#     dy0 = flight.latitude.values[1]-flight.latitude.values[0]
#     dx0 = flight.longitude.values[1]-flight.longitude.values[0]
#     dy = flight.latitude.values[-1]-flight.latitude.values[0]
#     dx = flight.longitude.values[-1]-flight.longitude.values[0]
#     theta = np.arctan2(dy,dx)
#     theta = theta * torch.ones_like(v)
#     theta[0] = np.arctan2(dy0,dx0)
#     duration = (flight.shape[0]-61)/(nwpts-1) * torch.ones_like(v)
#     duration[0] = 60
#     turn_rate = torch.ones(shapeturn_rate,names=names,device=device,dtype=dtype)*1e0
#     return FlightsWithAcc(xy0,v,theta,duration,turn_rate)

def initfmodel(flight,device,dtype,iwpts,xstr,ystr):
    nwpts = len(iwpts)
    # print(iwpts)
    # print("xstr",xstr)
    xt = flight[xstr].values
    yt = flight[ystr].values
    t = ((flight["timestamp"]-flight["timestamp"].iloc[0])/pd.to_timedelta(1,unit='s')).values#.dt-flight["timestamp"].values[0].dt
    # print(type(t))
    # print(dir(t))
    # raise Exception
    def getspeed(i):
        if i == 0:
            i=1
        return math.hypot(xt[i]-xt[i-1],yt[i]-yt[i-1])/(t[i]-t[i-1])
    def gettheta(i):
        if i == 0:
            i=1
        return math.atan2(yt[i]-yt[i-1],xt[i]-xt[i-1])
    namesbatch = tuple()
    names = namesbatch + (WPTS,)
    shape = (nwpts,)
    shapeturn_rate = (nwpts-1,)
    xy0 = torch.tensor(np.array([xt[0],yt[0]]),device=device,names=namesbatch+(XY,),dtype=dtype)
    # v = torch.tensor([flight.groundspeed.values[int(i)] for i in np.linspace(0,flight.shape[0]-1,nwpts)],device=device,names=names,dtype=dtype)/3600/60*1.1
    v = torch.tensor([getspeed((i+j)//2) for i,j in zip(iwpts[1:],iwpts[:-1])]+[getspeed(iwpts[-1])],device=device,names=names,dtype=dtype)
    theta = torch.tensor([gettheta((i+j)//2) for i,j in zip(iwpts[1:],iwpts[:-1])]+[gettheta(iwpts[-1])],device=device,names=names,dtype=dtype)
    duration = torch.tensor(np.diff(t[iwpts]).tolist()+[1e3],device=device,names=names,dtype=dtype)
    # duration[0] = 60
    turn_rate = torch.ones(shapeturn_rate,names=names,device=device,dtype=dtype)*1e-2
    return FlightsWithAcc(xy0,v,theta,duration,turn_rate)



def extract_maneuver_iwpts(flight,linecouple):
    # print(linecouple)
    # print(type(linecouple.start))
    wptsdates = [str(x) for x in [linecouple.selected_start,linecouple.start,linecouple.stop,linecouple.selected_end]]
    # print(dates)
    dates = [str(x.replace(tzinfo=pytz.UTC)) for x in flight.timestamp.values.tolist()]
    # print(dates)
    try:
        iwpts = [dates.index(t) for t in wptsdates]
    except ValueError:
        return None
    # assert(dates.index(linecouple.selected_start)==0)
    return iwpts

def extract_nonmaneuver_iwpts(flight,nwpts):
    # assert(dates.index(linecouple.selected_start)==0)
    return list(np.linspace(0,flight.shape[0]-1,nwpts,dtype=int))



def merge_pad_first_dim(lt,batchname):
    maxlt = max(t.shape[0] for t in lt)
    newnames = (batchname,)+lt[0].names
    newshape =(maxlt,)+lt[0].shape[1:]
    res = torch.stack([t[0].rename(None)*torch.ones(newshape,dtype=t.dtype,device=t.device) for t in lt]).rename(*newnames)
    for i,t in enumerate(lt):
        # assert(len(t.shape)==1)
        res[i,:t.shape[0]]=t
    return res

# def unify_trajs(ltraj,batchname):
#     maxltraj = max(traj.shape[0] for traj in ltraj)
#     res = torch.stack([torch.full((maxlt,),t.max().item(),dtype=t.dtype,device=t.device) for t in lt]).rename(batchname,lt[0].names[0])
#     for i,t in enumerate(lt):
#         assert(len(t.shape)==1)
#         res[i,:t.shape[0]]=t
#     return res


#iwpts = extractdate(flight,linecouple)
def fitflight(gen,fmodel,trajreal,t,niter,cst_duration=None):
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # fmodel = initfmodel(flight,device,dtype,iwpts,x,y)
#     return fitflightfromfmodel(flight,device,dtype,fmodel,x,y)
#     # print(type(flight.timestamp.values.tolist()[0]))
#     # print((flight.timestamp.values.tolist()[0]))
#     # np.linspace(0,flight.shape[0]-1,nwpts)
# def fitflightfromfmodel(flight,device,dtype,fmodel,x,y):
    # print(fmodel.theta)
    # print(fmodel.duration)
    # print(fmodel.v)
    # print(flight.dropna(subset=[x,y]).timestamp.min())
    # print(flight.dropna(subset=[x,y]).timestamp.max())
    # raise Exception
    # trajreal = torch.tensor([flight[x].values,flight[y].values],device=device,dtype=dtype)
    # trajreal = trajreal.transpose(0,1)#.rename()
    # print(trajreal.shape)
    fopti = fit.fit(fmodel,trajreal,t,gen,niter,cst_duration=cst_duration)
    # t=t.cpu()
    # trajreal = trajreal.cpu()
    xy = gen(fopti,t)
    # print(t)
    # print("mean",((trajreal-xy)**2).mean())
    # print("max",((trajreal-xy)**2).max())
    err = (trajreal-xy)#[:,:1524]
    # print(err.shape)
    # raise Exception
    # print(fopti.duration[3])
    # print(fopti.v[3])
    # print(fopti.theta[3])
    # print(fopti.turn_rate[3])
    # imax = 841
    # print(xy[3,imax])
    # print(trajreal[3,imax])
    # raise Exception
    # print(xy)
    # print(trajreal)
    dimstoreduce=tuple(range(len(xy.shape)))[1:]
    rmse = torch.sqrt((err**2).mean(dim=dimstoreduce)).cpu()
    maxabs = err.abs().rename(None).amax(dim=dimstoreduce).cpu()
    # print(maxabs)
    # raise Exception
    return (fopti.dmap(fopti,lambda v:v.cpu()),xy.cpu(),rmse,maxabs)
# def fmodel_from_deviation(flight,line,device,dtype):
#     namesbatch = tuple()
#     names = namesbatch + (WPTS,)
#     shape = (nwpts,)
#     shapeturn_rate = (nwpts-1,)
#     xy0 = torch.tensor([float(x) for x in [flight.longitude.values[0],flight.latitude.values[0]]],device=device,names=namesbatch+(XY,),dtype=dtype)
#     v = torch.tensor([flight.groundspeed.values[int(i)] for i in np.linspace(0,flight.shape[0]-1,nwpts)],device=device,names=names,dtype=dtype)/3600/60*1.1
#     dy0 = flight.latitude.values[1]-flight.latitude.values[0]
#     dx0 = flight.longitude.values[1]-flight.longitude.values[0]
#     dy = flight.latitude.values[-1]-flight.latitude.values[0]
#     dx = flight.longitude.values[-1]-flight.longitude.values[0]
#     theta = np.arctan2(dy,dx)
#     theta = theta * torch.ones_like(v)
#     theta[0] = np.arctan2(dy0,dx0)
#     duration = (flight.shape[0]-61)/(nwpts-1) * torch.ones_like(v)
#     duration[0] = 60
#     turn_rate = torch.ones(shapeturn_rate,names=names,device=device,dtype=dtype)*1e0
#     return FlightsWithAcc(xy0,v,theta,duration,turn_rate)




# def fitflight(flight,nwpts):
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     dtype = torch.float32
#     fmodel = initfmodel(flight,device,dtype,nwpts)
#     trajreal = torch.tensor([flight.longitude.values,flight.latitude.values],device=device,dtype=dtype)
#     trajreal = trajreal.transpose(0,1)#.rename()
#     t = torch.tensor(list(range(flight.shape[0])),device=device,names=(T,))
#     fopti = fit.fit(fmodel,trajreal,t)
#     xy = traj.generate(fopti,t)
#     return (fopti,xy,((trajreal-xy)**2).mean())

def main(infolder,outfolder):
    onlyfiles = [f for f in listdir(infolder) if isfile(join(infolder, f))]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for i,fname in enumerate(onlyfiles):
        # if i!=15:
        #     continue
        fname = join(infolder,fname)
        df = pd.read_parquet(fname)
        print(df.flight.flight_id.unique())
        # if df.flight.flight_id.values[0] != "AA38888789":
        #     continue
        print(i, fname)
        print(df.flight.timestamp.diff().min(),df.flight.timestamp.diff().max())
        # raise Exception
        fmodel = initfmodel(df.flight,device)
        trajreal = torch.tensor([df.flight.longitude.values,df.flight.latitude.values],device=device)
        trajreal = trajreal.transpose(0,1)#.rename()
        t = torch.tensor(list(range(df.flight.shape[0])),device=device,names=(T,))
        fopti = fit.fit(fmodel,trajreal,t)
        print(fmodel.duration)
        print(fmodel.theta)
        print(fopti.duration)
        print(fopti.theta)
        print(df.flight.flight_id[0], df.other.flight_id[0], df.flight.timestamp[0])
        with torch.no_grad():
            xy = traj.generate(fmodel,t)#.cpu(
        plt.scatter(xy[:,0].cpu().numpy(),xy[:,1].cpu().numpy(),color="red")
        with torch.no_grad():
            xy = traj.generate(fopti,t)#.cpu(
        print(((trajreal-xy)**2).mean())
        plt.scatter(xy[:,0].cpu().numpy(),xy[:,1].cpu().numpy(),color="blue")
        plt.scatter(trajreal[:,0].cpu().numpy(),trajreal[:,1].cpu().numpy(),color="green")
        plt.axis('equal')
        plt.show()

if __name__ == '__main__':
    main(INFOLDER,OUTFOLDER)
