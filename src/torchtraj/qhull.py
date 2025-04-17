import torch
from . import named

from .utils import PROJ, XY



def proj(phi):
    s = torch.sin(phi)
    c = torch.cos(phi)
    p = torch.stack([c, s],axis=-1)#.refine_names(...,"xy")
    return p


class QhullDist:
    def __init__(self, device, n = 180):
        self.projtheta = proj(torch.linspace(0, torch.pi, n,device=device))
    def projection(self, xy):
        self.projtheta = self.projtheta.to(xy.dtype)
        names = xy.names
        newnames = (names[:-1]+(PROJ,))
        # print(self.projtheta.dtype,xy.dtype)
        res = torch.inner(xy.rename(None), self.projtheta)
        return res.rename(*newnames)
    def dist(self, xy1, xy2, dimsInSet:tuple[str], capmem=None):
        if capmem is None:
            return self.distone(xy1, xy2, dimsInSet)
        else:
            candidatesplit = set(xy1.names).intersection(xy2.names).difference(set(dimsInSet).union({XY}))
            csplitn = [(s,name,i) for i,(s,name) in enumerate(zip(xy1.shape,xy1.names)) if name in candidatesplit]
            nb,chosen,ichosen = max(csplitn)
            ichosen2 = xy2.names.index(chosen)
            n_per_chosen = (max(xy1.numel(),xy2.numel())*self.projtheta.shape[0])//nb
            chunk_size = min(max(capmem//n_per_chosen,1),nb)
            # print(chunk_size,xy1.numel(),nb)
            assert(xy1.shape[ichosen] == xy2.shape[ichosen2])
            if nb==1 or nb == chunk_size:
                return self.dist(xy1,xy2,dimsInSet,capmem=None)
            else:
                l=[]
                # print(chunk_size,chosen)
                # print(len(torch.split(xy1,chunk_size,dim=ichosen)))
                # if chosen != "t":
                #     raise Exception
                for xy1s,xy2s in zip(torch.split(xy1,chunk_size,dim=ichosen),torch.split(xy2,chunk_size,dim=ichosen2)):
                    # print(xy1s.numel())
                    l.append(self.dist(xy1s,xy2s,dimsInSet,capmem=capmem))
                icat = l[0].names.index(chosen)
                return torch.cat(l,dim=icat)
            # print(l)
            # print(nb,chosen,ichosen)

    def distone(self, xy1, xy2, dimsInSet:tuple[str]):
        def replacedims(xy, suffix):
            newnames = []
            replaced = []
            for name in xy.names:
                if name in dimsInSet:
                    newname = name+suffix
                    assert(newname not in xy.names)
                    replaced.append(newname)
                else:
                    newname = name
                newnames.append(newname)
            return xy.rename(*newnames),tuple(replaced)
        xy1,replaced1 = replacedims(xy1, "1")
        xy2,replaced2 = replacedims(xy2, "2")
        # dimsInSet = replaced1 + replaced2
        # def support(proj,dims):
        #     return amaxnames(proj,dim=dims)#/np.sqrt(2.)
        xy1 = xy1.refine_names(...,XY)
        xy2 = xy2.refine_names(...,XY)
        p1 = self.projection(xy1)
        p2 = self.projection(xy2)#.align_as(p1)
        p1low = named.nanamin(p1,dim=replaced1) if len(replaced1) > 0 else p1
        p1high = named.nanamax(p1,dim=replaced1) if len(replaced1) > 0 else p1
        p2low = named.nanamin(p2,dim=replaced2) if len(replaced2) > 0 else p2
        p2high = named.nanamax(p2,dim=replaced2) if len(replaced2) > 0 else p2
        d1 = p2low - p1high
        d2 = p1low - p2high
        assert d1.names==d2.names
        d = torch.maximum(d1.rename(None),d2.rename(None))
        return torch.amax(d,dim=-1).refine_names(*d1.names[:-1])#*outnames1[:-1])#*d1.names[:-1])#amaxnames(d1, dim=()


# class QhullDistLowMem:
#     def __init__(self, device, n = 180):
#         self.projtheta = proj(torch.linspace(0, torch.pi, n,device=device))
#     def projection(self, xy, projtheta):
#         names = xy.names
#         newnames = (names[:-1]+(PROJ,))
#         # print(self.projtheta.dtype,xy.dtype)
#         res = torch.inner(xy.rename(None), projtheta)
#         # print(res.shape)
#         return res.rename(*newnames)
#     def dist(self, xy1, xy2, dimsInSet:tuple[str]):
#         for name in xy2.names:
#             assert(name not in dimsInSet)
#         projtheta = self.projtheta.to(xy1.dtype)
#         def support(proj,dims):
#             return amaxnames(proj,dim=dims)#/np.sqrt(2.)
#         xy1 = xy1.align_to(...,XY)
#         xy2 = xy2.align_as(xy1)
#         p1 = self.projection(xy1,projtheta)
#         p2 = self.projection(xy2,projtheta)#.align_as(p1)
#         assert p1.names==p2.names
#         p1low = named.amin(p1,dim=dimsInSet)
#         p1high = named.amax(p1,dim=dimsInSet)
#         p2low = named.amin(p2,dim=dimsInSet)
#         p2high = named.amax(p2,dim=dimsInSet)
#         d1 = p2low - p1high
#         d2 = p1low - p2high
#         # print(d1.shape)
#         d = torch.maximum(d1.rename(None),d2.rename(None))
#         # print(d.shape)
#         return torch.amax(d,dim=-1).refine_names(*d1.names[:-1])#amaxnames(d1, dim=()


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import time
    dims = [(40,"batch"),(2,"dt0"),(2,"dt1"),(2,"dangle"),(2,"dspeed"),(900,"t")]
    device = torch.device("cpu")#torch.device("cuda")
    torch.manual_seed(0)
    qh = QhullDist(device,n=3000)
    ds = tuple(x[0] for x in dims)
    ns = tuple(x[1] for x in dims)
    xy1 = torch.randn(ds+(2,),names=ns+(XY,)) + torch.tensor([15,0])
    # xy2 = torch.randn((ds[0],ds[-1])+(2,),names=(ns[0],ns[-1],XY,))
    xy2 = torch.randn(ds+(2,),names=ns+(XY,))
    xy1 = xy1.to(device)
    xy2 = xy2.to(device)
    # print(xy1)
    # print(xy2)
    # xy1 = torch.tensor([[5.,0.]]).rename("param","xy")
    # xy2 = torch.tensor([[1.],[0.]]).rename("xy","param")
    # print(xy1)
    # print(xy2.align_as(xy1))
    t0 = time.perf_counter()
    with torch.no_grad():
        resmem = qh.dist(xy1,xy2, dimsInSet=("dt0","dt1","dangle","dspeed"),capmem=18000000)
    print(time.perf_counter()-t0)
    t0 = time.perf_counter()
    # with torch.no_grad():
    #     res = qh.dist(xy1,xy2, dimsInSet=("dt0","dt1","dangle","dspeed"))
    # print(time.perf_counter()-t0)
    # assert((res==resmem).rename(None).all())
    # print(res,res.shape)
    # print(torch.norm((xy1-xy2.align_as(xy1)).rename(None)))
    # plt.scatter(xy1[:,0],xy1[:,1])
    # xy2 = xy2.align_as(xy1)
    # plt.scatter(xy2[:,0],xy2[:,1])
    # plt.show()


