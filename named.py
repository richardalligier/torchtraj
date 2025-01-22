import torch



def mergenames(n1, n2):
    n1


def align(x,y,names):
    return x.align_to(*names),y.align_to(*names)

def namestoints(t,whichnames):
    return tuple(t.names.index(n) for n in whichnames)


def amax(t, dim:tuple[str]):
    names = t.names
    intsdim = namestoints(t, dim)
    res = torch.amax(t.rename(None), dim=intsdim)
    newnames = tuple(n for n in t.names if n not in dim)
    # print(len(res.shape), len(newnames))
    assert len(res.shape)==len(newnames)
    return res.rename(*newnames)

def amin(t, dim:tuple[str]):
    names = t.names
    intsdim = namestoints(t, dim)
    res = torch.amin(t.rename(None), dim=intsdim)
    newnames = tuple(n for n in t.names if n not in dim)
    # print(len(res.shape), len(newnames))
    # print(res.shape)
    # print(t.names,dim)
    # print(newnames)
    assert len(res.shape)==len(newnames)
    return res.rename(*newnames)

def unsqueeze(t, dim, newname):
    if dim < 0:
        dim = dim + len(t.shape) +1
    newnames = list(t.names)
    newnames.insert(dim,newname)
    return torch.unsqueeze(t.rename(None), dim).refine_names(*newnames)

def unsqueeze_(t, dim, newname):
    if dim < 0:
        dim = dim + len(t.shape) +1
    newnames = list(t.names)
    newnames.insert(dim,newname)
    return t.rename_(None).unsqueeze_(dim).rename_(*newnames)


def pad(input, pad, mode='constant', value=None):
    return torch.nn.functional.pad(input.rename(None),pad,mode,value).rename(*input.names)


def repeat(x, sizes):
    names = x.names
    return x.rename(None).repeat(*sizes).rename(*names)
def stack(t,newname):
    names = t[0].names
    for ti in t:
        assert(ti.names==names)
    return torch.stack([ti.rename(None) for ti in t]).rename(*((newname,)+names))

def cat(t,dim=0):
    names = t[0].names
    for ti in t:
        assert(ti.names==names)
    return torch.cat([ti.rename(None) for ti in t],dim=dim).rename(*names)
