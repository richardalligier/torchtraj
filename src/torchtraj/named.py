import torch



def broadcastshapes(s1,s2):
    m = min(len(s1),len(s2))
    res = s1[:-m]+s2[:-m]
    s1 = s1[-m:]
    s2 = s2[-m:]
    print(res,s1,s2)
    for a,b in zip(s1,s2):
        assert(a==b or a==1 or b==1)
        res += (max(a,b),)
    return res



def mergenames(args):
    for x in args:
        assert(None not in x)
    res = set()
    for x in args:
        res = res.union(x)
    return tuple(sorted(res))

# def mergenames(n1, n2):
#     assert(None not in n1)
#     assert(None not in n2)
#     return tuple(set(n1).union(set(n2)))


def maximum(input,other):
    assert(input.names==other.names)
    names = input.names
    return torch.maximum(input.rename(None),other.rename(None)).rename(*names)

def align(x,y,names):
    return x.align_to(*names),y.align_to(*names)

def align_common(*args):
    names = mergenames([x.names for x in args])
    return tuple(x.align_to(*names) for x in args)

def namestoints(t,whichnames):
    return tuple(t.names.index(n) for n in whichnames)

def sort(t,dim):
    names = t.names
    i = names.index(dim)
    return torch.sort(t.rename(None),dim=i)[0].rename(*names)


def gather(input,dimname,index):
    assert(dimname in input.names)
    assert(dimname in index.names)
    newnames = mergenames((input.names,index.names))
    print(f"{newnames=}")
    inp = input.align_to(*newnames).rename(None)
    ind = index.align_to(*newnames).rename(None)
    bshape = list(broadcastshapes(inp.shape,ind.shape))
    i = newnames.index(dimname)
    inp = inp.broadcast_to(bshape)
    bshape[i]=ind.shape[i]
    ind = ind.broadcast_to(bshape)
    # print(index)
    # print(input)
    # print()
    # raise Exception
    # print(i)
    # print(ind)
    # print(newnames)
    return torch.gather(inp,i,ind).rename(*newnames)

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


def flip(t,dims):
    names = t.names
    indices = [names.index(s) for s in dims]
    return torch.flip(t.rename(None),indices).rename(*names)
