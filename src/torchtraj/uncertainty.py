import torch
from . import named,flights
from .utils import WPTS, T, XY,vheading,repeat_on_new_axis,compute_vxy
import itertools
import matplotlib.pyplot as plt
import operator as op

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
    iwpts = dxy.names.index(WPTS)
    bshape = named.broadcastshapes(dxy.shape,wpts_start.align_as(dxy).shape)
    bshape = named.broadcastshapes(bshape,wpts_end.align_as(dxy).shape)
    # print(bshape)
    # raise Exception
    print(iwpts,dxy.shape)
    assert(dxy.shape[iwpts]>1)
    dxy = dxy.broadcast_to(bshape).clone()
    # dxy = dxy.clone()
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


def adddt_translate(dt,wpts_start,wpts_end,f):
    '''
    wpts_start>0
    wpts_start-1 unchanged
    wpts_start is shifted
    this shift is propagated to others till wpts_end-1
    wpts_end-1 is the last wpts to be shifted
    wpts_end unchanged
    '''
    wpts_start = wpts_start - 1
    wpts_end = wpts_end - 1
    argstocheck = (dt,wpts_start,wpts_end)
    basename = compute_basename(f,*argstocheck)
    vhead = vheading(f.theta)
    meanv = f.meanv()
    assert(vhead.names[-2]==WPTS)
    assert(wpts_start.min()>0)
    duration_at_wpt = gather_wpts(f.duration,wpts_start).align_to(*basename)
    newduration_at_wpt = duration_at_wpt+dt.align_to(*basename)
    newduration_at_wpt = newduration_at_wpt.clip(min=0.)
    actual_dt = newduration_at_wpt - duration_at_wpt
    remaining_dt = dt.align_to(*basename) - actual_dt
    dxy = (vhead * meanv.align_to(...,XY)).align_to(*basename,WPTS,XY)
    dxy = dxy * actual_dt.align_as(dxy)
    dxy = zero_pad(dxy,wpts_start-1,wpts_start+1)
    dxy = torch.cumsum(dxy,axis=-2)
    dxy = zero_pad(dxy,wpts_start-1,wpts_end)
    dparams = applydxy(f,dxy)
    result = f.from_wpts(**dparams)
    if remaining_dt.max()==0.:
        return result
    else:
        return adddt_translate(remaining_dt,wpts_start,wpts_end+1,result)


# def _contract(f,fref,wpts_start,wpts_end):
#     '''
#     wpts_start>0
#     wpts_start unchanged
#     this shift is propagated to others till wpts_end-1
#     wpts_end-1 is the last wpts to be shifted
#     wpts_end unchanged
#     '''
#     assert(wpts_start.min()>0)
#     wpts_start = wpts_start-1
#     wpts_end = wpts_end-1
#     argstocheck = (fref,wpts_start,wpts_end)
#     basename = compute_basename(f,*argstocheck)
#     assert((wpts_start.align_to(*basename)<wpts_end.align_to(*basename)).rename(None).all())
#     wpts = f.compute_wpts()
#     beacon = gather_wpts(wpts,wpts_end-1)
#     return _contract_beacon(f,fref,wpts_start,wpts_end,beacon)



def _contract(f,fref,wpts_start,wpts_end,beacon=None):
    '''
    wpts_start>0
    wpts_start unchanged
    this shift is propagated to others till wpts_end-1
    wpts_end-1 is the last wpts to be shifted
    wpts_end unchanged
    '''
    assert(wpts_start.min()>0)
    wpts_start = wpts_start-1
    wpts_end = wpts_end-1
    argstocheck = (fref,wpts_start,wpts_end)
    basename = compute_basename(f,*argstocheck)
    assert((wpts_start.align_to(*basename)<wpts_end.align_to(*basename)).rename(None).all())
    # print(f"{basename=}")
    wpts = f.compute_wpts()
    if beacon is None:
        beacon = gather_wpts(wpts,wpts_end-1)
    distf = compute_distance_start_beacon(wpts,wpts_start,beacon)
    wptsref = fref.compute_wpts()
    distfref = compute_distance_start_beacon(wptsref,wpts_start,beacon)
    ratio = distf.align_to(*basename) / distfref.align_to(*basename)
    print(f"{ratio=}")
    xy = wpts
    xy_zero = gather_wpts(xy,wpts_start)
    # print(f"{ratio=}")
    # print(f"{xy_zero=}")
    dxy = (xy - xy_zero.align_as(xy))*(ratio.align_as(xy)-1)
    dxy = zero_pad(dxy,wpts_start,wpts_end)
    dparams = applydxy(f,dxy)
    return f.from_wpts(**dparams)

# def compute_theta_start_end(wpts,wpts_start,wpts_end):
#     '''
#     compute angle to fly from start to end
#     '''
#     beacon = gather_wpts(wpts,wpts_end-1)
#     return compute_theta_start_beacon(wpts,wpts_start,beacon)

def compute_theta_start_beacon(wpts,wpts_start,beacon):
    turn = gather_wpts(wpts,wpts_start-1)
    diff = op.sub(*named.align_common(beacon,turn)).align_to(...,XY)
    return torch.atan2(diff[...,1],diff[...,0])


def compute_distance_start_end(wpts,wpts_start,wpts_end):
    '''
    compute angle to fly from start to end
    '''
    beacon = gather_wpts(wpts,wpts_end)
    return compute_distance_start_beacon(wpts,wpts_start,beacon)

def compute_distance_start_beacon(wpts,wpts_start,beacon):
    '''
    compute angle to fly from start to end
    '''
    turn = gather_wpts(wpts,wpts_start)
    diff = op.sub(*named.align_common(beacon,turn)).align_to(...,XY)**2
    return torch.sqrt(torch.sum(diff,axis=-1))


def adddt_rotate(dt,wpts_start,wpts_turn,wpts_end,f, contract=True,beacon=None):
    '''
    wpts_start>0
    wpts_start-1 unchanged
    wpts_start to wpts_turn is shifted
    from this shift, a rotation is computed from wpts_turn to wpts_end , and is applied
    wpts_end is the last wpts to be shifted
    wpts_end+1 unchanged
    '''
    argstocheck = (dt,wpts_start,wpts_end)
    basename = compute_basename(f,*argstocheck)
    newf = adddt_translate(dt,wpts_start,wpts_end+1,f)
    newwpts = newf.compute_wpts()
    wpts = f.compute_wpts()
    if beacon is None:
        beacon = gather_wpts(wpts,wpts_end-1)
        print(f"{beacon=}")
    newtheta = compute_theta_start_beacon(newwpts,wpts_turn,beacon).align_to(*basename)
    print(f"{newtheta.names=}  {newtheta.shape=}")
    oldtheta = compute_theta_start_beacon(wpts,wpts_turn,beacon).align_to(*basename) #+ dtheta.align_to(*basename)
    newf = rotate_wpts(newtheta-oldtheta,wpts_turn,wpts_end+1,newf)
    return _contract(newf,f,wpts_turn,wpts_end+1,beacon=beacon) if contract else newf


def change_longitudinal_speed(dspeed,wpts_start,wpts_rejoin,f):
    '''
    wpts_start unchanged
    wpts_start+1 is the shifted due to speed
    wpts_rejoin unchanged
    speed on segments between wpts_start to wpts_rejoin is changed
    '''
    assert(f.v.names[-1] == WPTS)
    argstocheck = (dspeed,wpts_rejoin,wpts_start)
    basename = compute_basename(f,*argstocheck)
    dparams = {}
    dspeed = dspeed.align_to(*basename,WPTS)
    bshape = list(dspeed.shape)
    bshape[-1] = f.nwpts()
    dspeed = dspeed.broadcast_to(bshape)
    # print(wpts_rejoin.names,dspeed.names)
    dspeed = 1 + zero_pad(dspeed-1,wpts_start-1,wpts_rejoin)
    # print(dspeed)
    # raise Exception
    dparams["v"] =  op.mul(*named.align_common(f.v,dspeed)).align_to(...,WPTS)
    dparams["turn_rate"] = f.turn_rate.clone().align_as(dparams["v"])
    # print(dparams["v"].names)
    # print(f.compute_wpts().names)
    dparams["wpts"] = f.compute_wpts().align_to(*dparams["v"].names,XY)
    dparams["xy0"] = f.xy0.clone().align_to(*dparams["v"].names[:-1],XY)
    dparams["turn_rate"] = f.turn_rate.clone().align_as(dparams["v"])
    return f.from_wpts(**dparams)

def changespeed(dspeed,wpts_start,wpts_turn,wpts_rejoin,f):
    '''
    wpts_start unchanged
    wpts_start+1 is the shifted due to speed
    wpts_turn is the shifted due to speed
    points between wpts_turn excluded to wpts_rejoin excluded are only "rotated"
    wpts_end unchanged
    speed on segments between wpts_start to wpts_rejoin is changed
    '''
    assert(f.v.names[-1] == WPTS)
    argstocheck = (dspeed,wpts_start,wpts_turn,wpts_rejoin)
    basename = compute_basename(f,*argstocheck)
    nwpts = f.v.shape[-1]
    wpts = f.compute_wpts()
    meanv = f.meanv()
    vhead = vheading(f.theta).align_to(*basename,WPTS,XY)
    velocity = vhead * meanv.align_as(vhead)
    newmeanv = (dspeed.align_as(velocity)-1)*velocity
    dxy = newmeanv * f.duration.align_as(newmeanv)
    assert(dxy.names[-2] == WPTS)
    print(f"{dxy.names=} {dxy.shape=}")
    print(dxy)
    dxy = zero_pad(dxy,wpts_start-1,wpts_turn)
    dxy = torch.cumsum(dxy,axis=-2)
    dxy = zero_pad(dxy,wpts_start-1,wpts_rejoin)
    print(dxy)
    # raise Exception
    dspeed = dspeed.align_to(*basename,WPTS)
    dspeeds = list(dspeed.shape)
    dspeeds[-1] = f.nwpts()
    dspeed = zero_pad(dspeed.broadcast_to(dspeeds)-1,wpts_start-1,wpts_rejoin)
    # print(dspeed)
    # raise Exception
    dparams = applydxy(f, dxy)
    dparams["v"] = f.v.align_to(*basename,WPTS)*(1+dspeed)#.align_as(*-1,wpts_start-1,wpts_end)
    return f.from_wpts(**dparams)

def get_dates(twpts,mask_climb):
    start_cmb = mask_climb.clone()
    start_cmb[...,1:] = torch.logical_and(mask_climb[...,1:],torch.logical_not(mask_climb[...,:-1]))#torch.logical_and(mask_cruise[...,:-1],mask_climb[...,1:])
    #start_cmb = named.cat([start_cmb,torch.ones_like(start_cmb[...,:1])],dim=-1)
    start_dates = start_cmb*twpts[...,:-1].align_as(start_cmb)
    # print(mask_climb[45:])
    # print(start_cmb[45:])
    # print(start_dates[45:])
    # raise Exception
    end_cmb = mask_climb.clone()
    end_cmb[...,:-1] = torch.logical_and(mask_climb[...,:-1],torch.logical_not(mask_climb[...,1:]))
    end_dates = end_cmb*twpts[...,1:].align_as(end_cmb)
    end_bfill = named.backward_fill(end_dates,end_dates==0,dim=WPTS)
    start_ffill = named.forward_fill(start_dates,start_dates==0,dim=WPTS)
    duration = (end_bfill-start_ffill)*mask_climb
    return start_dates,end_dates,duration

def get_durations(duration,mask_cruise,mask_climb):
    twpts = flights.compute_twpts_with_wpts0(duration).align_to(...,WPTS)#[...,:-1]
    start_cmb,end_cmb,dur_cmb = get_dates(twpts,mask_climb)
    start_crs,end_crs,dur_crs = get_dates(twpts,mask_cruise)
    print(mask_cruise[45:])
    print(start_crs[45:])
    print(end_crs[45:])
    print(dur_crs[45:])
    print(start_cmb[45:])
    print(end_cmb[45:])
    print(dur_cmb[45:])
    return dur_crs,dur_cmb
    # raise Exception
    # ffill_end = named.forward_fill(end_dates,end_dates==0,dim=WPTS)
    # bfill_start = named.backward_fill(start_dates,start_dates==0,dim=WPTS)
    # duration_cruise = (bfill_start-ffill_end) * torch.logical_not(mask_climb)
    # bfill_end = named.backward_fill(end_dates,end_dates==0,dim=WPTS)
    # ffill_start = named.forward_fill(start_dates,start_dates==0,dim=WPTS)
    # duration_climb = (bfill_end-ffill_start) * mask_climb
    # return duration_climb


# def change_vertical_speed(dvspeed,tmin,tmax,f,thresh_rocd=200/60,iz=1):
#     assert(iz==0 or iz==1)
#     it = 1-iz
#     assert(f.v.names[-1] == WPTS)
#     argstocheck = (dvspeed,)
#     basename = compute_basename(f,*argstocheck)
#     dparams = f.dictparams()
#     dvspeed = dvspeed.align_to(*basename,WPTS)
#     twpts0 = f.compute_twpts_with_wpts0().align_to(...,WPTS)
#     twpts = twpts0[...,:-1]
#     # print(twpts)
#     # raise Exception
#     v = dparams["v"].align_as(dvspeed) * torch.ones_like(dvspeed)
#     vxy = v.align_to(*basename,WPTS,XY) * vheading(dparams["theta"]).align_to(*basename,WPTS,XY)
#     maskz = torch.abs(vxy[...,iz]) > thresh_rocd
#     # raise Exception
#     maskt =tmin.align_as(maskz)<=twpts.align_as(maskz)
#     maskt = torch.logical_and(maskt,twpts.align_as(maskz)<tmax.align_as(maskz))
#     mask_climb = torch.logical_and(maskt,maskz)
#     scalez = named.where(mask_climb, dvspeed.align_to(*basename,WPTS),torch.ones_like(dvspeed))
#     vxy[...,iz] = vxy[...,iz] * scalez
#     dparams["v"] = torch.hypot(vxy[...,0],vxy[...,1])
#     dparams["theta"] = torch.atan2(vxy[...,1],vxy[...,0])
#     duration_cruise,duration_climb = get_durations(dparams["duration"].align_to(*basename,WPTS),mask_climb)
#     dparams["duration"] = dparams["duration"].align_to(*basename,WPTS) / scalez
#     mduration_cruise,mduration_climb = get_durations(dparams["duration"].align_to(*basename,WPTS),mask_climb)
#     modif_climb = mduration_climb - duration_climb
#     modif_cruise = (duration_cruise - named.forward_fill(modif_climb,torch.logical_not(mask_climb),dim=WPTS))*(~mask_climb)
#     scale_c = named.where(mask_climb,torch.ones_like(modif_cruise),modif_cruise / duration_cruise)
#     dparams["duration"]=dparams["duration"]*scale_c
#     print(mduration_climb[45:])
#     print(duration_climb[45:])
#     print(mduration_cruise[45:])
#     print(duration_cruise[45:])
#     print(modif_cruise[45:])
#     # raise Exception
#     for k in ["turn_rate","theta"]:
#         dparams[k]=dparams[k].align_to(*basename,WPTS)
#     for k in ["xy0"]:
#         dparams[k]=dparams[k].align_to(*basename,XY)
#     # raise Exception
#     return f.from_argsdict(dparams)

def scale_vspeed(dparams,scalez,iz):
    vxy = compute_vxy(v=dparams["v"],theta=dparams["theta"])
    vxy[...,iz] = vxy[...,iz] * scalez
    dparams["v"] = torch.hypot(vxy[...,0],vxy[...,1])
    dparams["theta"] = torch.atan2(vxy[...,1],vxy[...,0])
    dparams["duration"] = dparams["duration"].align_as(scalez)/scalez

def change_vertical_speed_fwd(dvspeed,tmin,tmax,f,thresh_rocd=200/60,iz=1):
    assert(iz==0 or iz==1)
    it = 1-iz
    assert(f.v.names[-1] == WPTS)
    argstocheck = (dvspeed,)
    basename = compute_basename(f,*argstocheck)
    dparams = f.dictparams()
    dvspeed = dvspeed.align_to(*basename,WPTS)
    twpts0 = f.compute_twpts_with_wpts0().align_to(...,WPTS)
    twpts = twpts0[...,:-1]
    # print(twpts)
    # raise Exception
    dparams["v"] = dparams["v"].align_as(dvspeed) * torch.ones_like(dvspeed)
    dparams["theta"]= dparams["theta"].align_to(*basename,WPTS)
    vxy = compute_vxy(v=dparams["v"],theta=dparams["theta"])
    maskz = torch.abs(vxy[...,iz]) > thresh_rocd
    # raise Exception
    maskt =tmin.align_as(maskz)<=twpts.align_as(maskz)
    maskt = torch.logical_and(maskt,twpts.align_as(maskz)<tmax.align_as(maskz))
    mask_climb = torch.logical_and(maskt,maskz)
    mask_cruise = torch.logical_and(maskt,~maskz)
    scalez = named.where(mask_climb, dvspeed.align_to(*basename,WPTS),torch.ones_like(dvspeed))
    duration_cruise,duration_climb = get_durations(dparams["duration"].align_to(*basename,WPTS),mask_cruise,mask_climb)
    scale_vspeed(dparams,scalez,iz)
    # vxy[...,iz] = vxy[...,iz] * scalez
    # dparams["v"] = torch.hypot(vxy[...,0],vxy[...,1])
    # dparams["theta"] = torch.atan2(vxy[...,1],vxy[...,0])
    # dparams["duration"] = dparams["duration"].align_to(*basename,WPTS) / scalez
    mduration_cruise,mduration_climb = get_durations(dparams["duration"].align_to(*basename,WPTS),mask_cruise,mask_climb)
    modif_climb = mduration_climb - duration_climb
    modif_cruise = ( duration_cruise- named.backward_fill(modif_climb,mask_cruise,dim=WPTS))#*mask_cruise
    scale_c = named.where(mask_cruise, duration_cruise/modif_cruise ,torch.ones_like(modif_cruise))
    scale_vspeed(dparams,scale_c,iz)
    # print(modif_climb[45:])
    # print(modif_cruise[45:])
    # print(scale_c[45:])
    print(modif_climb[43:44])
    print(modif_cruise[43:44])
    print(mask_cruise[43:44])
    print(mask_climb[43:44])
    print(scale_c[43:44])
    # raise Exception
    for k in ["turn_rate","theta"]:
        dparams[k]=dparams[k].align_to(*basename,WPTS)
    for k in ["xy0"]:
        dparams[k]=dparams[k].align_to(*basename,XY)
    # raise Exception
    return f.from_argsdict(dparams)

# def change_vertical_speed_bwd(dvspeed,tmin,tmax,f,thresh_rocd=200/60,iz=1):
#     assert(iz==0 or iz==1)
#     it = 1-iz
#     assert(f.v.names[-1] == WPTS)
#     argstocheck = (dvspeed,)
#     basename = compute_basename(f,*argstocheck)
#     dparams = f.dictparams()
#     dvspeed = dvspeed.align_to(*basename,WPTS)
#     twpts0 = f.compute_twpts_with_wpts0().align_to(...,WPTS)
#     twpts = twpts0[...,:-1]
#     # print(twpts)
#     # raise Exception
#     v = dparams["v"].align_as(dvspeed) * torch.ones_like(dvspeed)
#     vxy = v.align_to(*basename,WPTS,XY) * vheading(dparams["theta"]).align_to(*basename,WPTS,XY)
#     maskz = torch.abs(vxy[...,iz]) > thresh_rocd
#     # raise Exception
#     maskt =tmin.align_as(maskz)<=twpts.align_as(maskz)
#     maskt = torch.logical_and(maskt,twpts.align_as(maskz)<tmax.align_as(maskz))
#     mask_climb = torch.logical_and(maskt,maskz)
#     mask_cruise = torch.logical_and(maskt,~maskz)
#     scalez = named.where(mask_climb, dvspeed.align_to(*basename,WPTS),torch.ones_like(dvspeed))
#     vxy[...,iz] = vxy[...,iz] * scalez
#     dparams["v"] = torch.hypot(vxy[...,0],vxy[...,1])
#     dparams["theta"] = torch.atan2(vxy[...,1],vxy[...,0])
#     duration_cruise,duration_climb = get_durations(dparams["duration"].align_to(*basename,WPTS),mask_climb)
#     dparams["duration"] = dparams["duration"].align_to(*basename,WPTS) / scalez
#     mduration_cruise,mduration_climb = get_durations(dparams["duration"].align_to(*basename,WPTS),mask_climb)
#     modif_climb = mduration_climb - duration_climb
#     modif_cruise = (duration_cruise - named.backward_fill(modif_climb,mask_climb,dim=WPTS))*mask_cruise
#     scale_c = named.where(mask_cruise,modif_cruise / duration_cruise,torch.ones_like(modif_cruise))
#     dparams["duration"]=dparams["duration"]*scale_c
#     # print(mduration_climb[45:])
#     # print(duration_climb[45:])
#     # print(mduration_cruise[45:])
#     # print(duration_cruise[45:])
#     # print(modif_cruise[45:])
#     # # raise Exception
#     for k in ["turn_rate","theta"]:
#         dparams[k]=dparams[k].align_to(*basename,WPTS)
#     for k in ["xy0"]:
#         dparams[k]=dparams[k].align_to(*basename,XY)
#     # raise Exception
#     return f.from_argsdict(dparams)



#change wpts de fin
def changespeed_rotate(dspeed,wpts_start,wpts_turn,wpts_rejoin,f,beacon=None,contract=True):
    '''
    wpts_start unchanged
    wpts_start+1 is the shifted due to speed
    wpts_turn is the shifted due to speed
    points between wpts_turn excluded to wpts_rejoin excluded are only "rotated"
    wpts_end unchanged
    speed on segments between wpts_start to wpts_rejoin is changed
    '''
    assert(f.v.names[-1] == WPTS)
    argstocheck = (dspeed,wpts_start,wpts_turn,wpts_rejoin)
    basename = compute_basename(f,*argstocheck)
    newf = changespeed(dspeed,wpts_start,wpts_turn,wpts_rejoin,f)
    newwpts = newf.compute_wpts()
    wpts = f.compute_wpts()
    if beacon is None:
        beacon = gather_wpts(wpts,wpts_rejoin-1)
        print(f"{beacon=}")
    newtheta = compute_theta_start_beacon(newwpts,wpts_turn,beacon).align_to(*basename)
    print(f"{newtheta.names=}  {newtheta.shape=}")
    oldtheta = compute_theta_start_beacon(wpts,wpts_turn,beacon).align_to(*basename) #+ dtheta.align_to(*basename)
    newf = rotate_wpts(newtheta-oldtheta,wpts_turn,wpts_rejoin+1,newf)
    return _contract(newf,f,wpts_turn,wpts_rejoin+1,beacon=beacon) if contract else newf

def shift_t_zero(f,tshift):
    assert(tshift.min()>=0.)



def addangle(dtheta,wpts_start,wpts_turn,wpts_rejoin,f,beacon=None, contract=True):
    '''
    wpts_start>=0
    wpts_start unchanged
    wpts_start+1 is the shifted due to angle
    wpts_turn is the shifted due to angle
    points between wpts_turn excluded to wpts_end included are only "rotated"
    wpts_end+1 unchanged
    if beacon is None, beacon is wpts[wpts_end], and consequently wpts_end is unchanged
    '''
    argstocheck = (dtheta,wpts_start,wpts_turn,wpts_rejoin)
    basename = compute_basename(f,*argstocheck)#named.mergenames((f.theta.names[:-1],)+ tuple(x.names for x in argstocheck))
    wpts_rejoin = wpts_rejoin
    newf = rotate_wpts(dtheta,wpts_start,wpts_rejoin+1,f)
    newwpts = newf.compute_wpts()
    wpts = f.compute_wpts()
    if beacon is None:
        beacon = gather_wpts(wpts,wpts_rejoin-1)
        print(f"compute beacon {beacon}")
    newtheta = compute_theta_start_beacon(newwpts,wpts_turn,beacon=beacon).align_to(*basename)
    print(f"{newtheta.names=}  {newtheta.shape=}")
    oldtheta = compute_theta_start_beacon(wpts,wpts_turn,beacon=beacon).align_to(*basename) + dtheta.align_to(*basename)
    newf = rotate_wpts(newtheta-oldtheta,wpts_turn,wpts_rejoin+1,newf)
    return _contract(newf,f,wpts_turn,wpts_rejoin+1,beacon=beacon) if contract else newf


#def addangle_to(dtheta,wpts_start,wpts_turn,wpts_rejoin,f, contract=True):


# def compute_rotate_to_rejoin(fref,fmodified,wpts_turn,wpts_rejoin):
#     def compute_theta_turn_rejoin(f1,f2):
#         turn = gather_wpts(f1.compute_wpts(),wpts_turn-1)
#         rejoin = gather_wpts(f2.compute_wpts(),wpts_rejoin-1)
#         diff = (rejoin-turn).align_to(...,XY)
#         return torch.atan2(diff[...,1],diff[...,0]).align_to(*basename)
#     newtheta = compute_theta_turn_rejoin(fmodified,fmodified)
#     print(f"{newtheta.names=}  {newtheta.shape=}")
#     oldtheta = compute_theta_turn_rejoin(fref,fref) + dtheta.align_to(*basename)

def rotate_wpts(dtheta,wpts_start,wpts_end,f):
    '''
    wpts_start>=0
    wpts_start unchanged
    wpts_start+1 is the shifted due to angle
    points between wpts_start excluded to wpts_end excluded are only "rotated"
    wpts_end unchanged
    '''
    print(f"{wpts_start=} {wpts_end=}")
    basename = compute_basename(f,dtheta,wpts_start,wpts_end)
    newtheta = f.theta.align_to(*basename,WPTS) + dtheta.align_to(*basename,WPTS)
    newvheading = vheading(newtheta)
    dxytheta = newvheading - vheading(f.theta).align_as(newvheading)
    assert(dxytheta.names[-2]==WPTS)
    assert(dxytheta.names[-1]==XY)
    dxy = dxytheta * f.meanv().align_as(dxytheta) * f.duration.align_as(dxytheta)
    dxy = zero_pad(dxy,wpts_start-1,wpts_end-1)
    dxy = torch.cumsum(dxy,axis=-2)
    dxy = zero_pad(dxy,wpts_start-1,wpts_end-1)
    dparams=applydxy(f, dxy)
    # dparams=make_consistent(applydxy(f, dtheta.names, dxy))
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


