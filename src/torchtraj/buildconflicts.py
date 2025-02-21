import pandas as pd
import matplotlib.pyplot as plt
import tqdm
import datetime

OUT="conflicts"

def check(f,tstart,tstop):
    l = [
        f.timestamp.diff().min()==f.timestamp.diff().max(),
        f.timestamp.diff().min()==datetime.timedelta(seconds=1),
        f.timestamp.min()==tstart,
        f.timestamp.max()==tstop,
    ]
    return l

def main():
    df = pd.read_parquet("couples_4.parquet")
    # trajf = pd.read_parquet("THEtrajectoires.parquet")
    trajf = pd.read_parquet("raw_2207_LFBBBDX.parquet")#"THEtrajectoires.parquet")
    # print(df)
    for (i,line) in tqdm.tqdm(df.iterrows()):
        l = [trajf.query(f"flight_id=='{x}'").reset_index(drop=True) for x in (line.flight_id,line.neighbour_id)]
        print(i)
        # for k,x in enumerate(l):
        #     print(k,x.timestamp.min(),x.timestamp.max())
        tstart = line.start-datetime.timedelta(seconds=30)
        l = [f for f in l]#[f.query(f'"{tstart}"<=timestamp<="{line.stop}"').reset_index(drop=True) for f in l]
        # try:
        #     for x in l:
        #         assert(all(check(x,tstart,line.stop)))
        # except AssertionError as e:
        #     print("Reject begin")
        #     print(line,x.timestamp.min(),x.timestamp.max(),x.timestamp.diff().min(),x.timestamp.diff().max())
        #     print(check(x,line.start,line.stop))
        #     print("Reject end")
        #     continue
        d = {"flight":l[0],"other":l[1]}
        dfconf = pd.concat(d.values(),axis=1,keys=d.keys())
        dfconf.to_parquet(f"{OUT}/{i}.parquet")
        # print(dfconf)
        # raise Exception
        # print(list(x))

        # print(line)
main()

