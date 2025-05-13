import torch


def transposewith0(x,name):
    i = x.names.index(name)
    newnames = list(x.names)
    newnames[i],newnames[0]=newnames[0],newnames[i]
    return torch.transpose(x.rename(None),0,i).rename(*newnames)


def forward_fill(x: torch.Tensor,mask, dim: str) -> torch.Tensor:
    # Move the fill dimension to the front
    assert x.names==mask.names
    names =x.names
    x = x.rename(None)
    mask = mask.rename(None)
    dim = names.index(dim)
    x_transposed = x.transpose(0, dim)
    mask = mask.transpose(0,dim)
    # Replace NaNs with 0 temporarily
    x_filled = x_transposed.clone()
    #x_filled[mask] = 0
    # Create an indicator where valid values occur
    valid = (~mask).to(x.dtype)
    # Compute the cumulative sum of valid indicators
    cumsum = valid.cumsum(dim=0)
    # Avoid division by zero
    # cumsum[cumsum == 0] = 1
    cumsum = torch.where(cumsum == 0, torch.ones_like(cumsum),cumsum)
    # Compute running max index
    idx = torch.arange(x_transposed.size(0), device=x.device).unsqueeze(1)
    idx = idx.expand(-1, x_transposed[0].numel()).reshape_as(x_transposed)
    idx = torch.where(valid.bool(), idx, torch.zeros_like(idx))
    idx = idx.cummax(dim=0).values
    # Gather forward-filled values
    flat_x = x_filled.reshape(x_transposed.size(0), -1)
    flat_idx = idx.reshape(x_transposed.size(0), -1)
    result = flat_x.gather(0, flat_idx)
    # Reshape back and transpose to original dimension order
    result = result.view_as(x_transposed)
    return result.transpose(0, dim).rename(*names)


# def flip(x,dims):
#     return torch.flip(x.rename(None),dims=dims).rename(*x.names)

def backward_fill(x: torch.Tensor,mask, dim: str) -> torch.Tensor:
    # Flip along the fill dimension, apply forward fill, then flip back
    flipped = flip(x, dims=[dim])
    flipped_mask = flip(mask, dims=[dim])
    # print(f"{flipped[45:]=}")
    filled_flipped = forward_fill(flipped,flipped_mask, dim=dim)
    # print(f"{filled_flipped[45:]=}")
    return flip(filled_flipped, dims=[dim])


def deserialize(d):
    t,names = d
    return t.rename(*names)

def serialize(v):
    return deserialize,(v.rename(None).cpu(),v.names)

def broadcastshapes(s1,s2):
    m = min(len(s1),len(s2))
    res = s1[:-m]+s2[:-m]
    s1 = s1[-m:]
    s2 = s2[-m:]
    # print("b",res,s1,s2)
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


def minimum(input,other):
    assert(input.names==other.names)
    names = input.names
    return torch.minimum(input.rename(None),other.rename(None)).rename(*names)

def align(x,y,names):
    return x.align_to(*names),y.align_to(*names)

def align_common(*args):
    names = mergenames([x.names for x in args])
    return tuple(x.align_to(*names) for x in args)

def namestoints(t,whichnames):
    return tuple(t.names.index(n) for n in whichnames)

def where(condition,input,other):
    l = [condition,input,other]
    names = condition.names
    for x in l:
        assert(x.names==names)
    res = torch.where(condition.rename(None),input.rename(None),other.rename(None))
    return res.rename(*names)


def sort(t,dim):
    names = t.names
    # for x in names:
    #     assert(x is not None)
    i = names.index(dim)
    res = torch.sort(t.rename(None),dim=i)[0]
    # print(res.shape,res.names,names)
    # assert(len(res.shape)==len(names))
    return res.rename(*names)


def gather(input,dimname,index):
    assert(dimname in input.names)
    assert(dimname in index.names)
    newnames = mergenames((input.names,index.names))
    inp = input.align_to(*newnames).rename(None)
    ind = index.align_to(*newnames).rename(None)
    # assert (inp.names==ind.names)
    # inp = inp.rename(None)
    # ind = ind.rename(None)
    # print(f"{newnames=}")
    bshape = list(broadcastshapes(inp.shape,ind.shape))
    i = newnames.index(dimname)
    inp = inp.broadcast_to(bshape)
    bshape[i]=ind.shape[i]
    ind = ind.broadcast_to(bshape)
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




def nanamax(tensor, dim:tuple[str]):
    min_value = torch.finfo(tensor.dtype).min
    out= torch.nan_to_num(tensor.rename(None),nan=min_value).rename(*tensor.names)
    out = amax(out,dim)
    outnames = out.names
    out = out.rename(None)
    out[out==min_value]= torch.nan
    return out.rename(*outnames)


def nanamin(tensor, dim:tuple[str]):
    min_value = torch.finfo(tensor.dtype).max
    out= torch.nan_to_num(tensor.rename(None),nan=min_value).rename(*tensor.names)
    out = amin(out,dim)
    outnames = out.names
    out = out.rename(None)
    out[out==min_value]= torch.nan
    return out.rename(*outnames)



# def nanamax(tensor, dim:tuple[str]):
#     min_value = torch.finfo(tensor.dtype).min
#     print(tensor.shape,tensor.names)
#     tensor = torch.nan_to_num(tensor.rename(None),nan=min_value).rename(*tensor.names)
#     print(tensor.shape,tensor.names)
#     # raise Exception
#     return amax(tensor,dim)


# def nanamin(tensor, dim:tuple[str]):
#     min_value = torch.finfo(tensor.dtype).max
#     print(tensor.shape,tensor.names)
#     tensor = torch.nan_to_num(tensor.rename(None),nan=min_value).rename(*tensor.names)
#     print(tensor.shape,tensor.names)
#     # raise Exception
#     return amin(tensor,dim)


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
    return torch.nn.functional.pad(input.rename(None),pad,mode,value=value).rename(*input.names)


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


def nanmax(tensor, dim=None, keepdim=False):
    min_value = torch.finfo(tensor.dtype).min
    output = tensor.nan_to_num(min_value)
    if dim is None:
        return torch.max(output)
    else:
        return torch.max(output,dim=dim,keepdim=keepdim)


def nanmin(tensor, dim=None, keepdim=False):
    max_value = torch.finfo(tensor.dtype).max
    output = tensor.nan_to_num(max_value)
    if dim is None:
        return torch.min(output)
    else:
        return torch.min(output,dim=dim,keepdim=keepdim)

