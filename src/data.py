import numpy as np
from pathlib import Path

def generate(path,seed=4101):
    rng=np.random.default_rng(seed);N,T,D,H,W=5,4,4,12,12
    z,y,x=np.mgrid[:D,:H,:W];labels=np.zeros((N,T,D,H,W),np.int64);images=[];points=np.full((N,3,D,3,2),-1,np.float32)
    for n in range(N):
        seq=[]
        for t in range(T):
            xx=x-(5+.20*t+.08*n);yy=y-(6-.12*t);rr=xx*xx+yy*yy
            lv=(rr<(2.0-.12*t)**2);my=(rr<(3.15-.12*t)**2)&~lv;rv=((x-8-.15*t)**2+(y-4)**2<2)
            lab=np.zeros((D,H,W),np.int64);lab[my]=3;lab[rv]=2;lab[lv]=1;labels[n,t]=lab
            seq.append((.12+.6*lv+.35*my+.45*rv+.05*z+rng.normal(0,.025,(D,H,W))).astype(np.float32))
        images.append(seq)
        for c in range(3):
            for zz in range(D):
                yy,xx=np.where(labels[n,0,zz]==c+1)
                ids=rng.choice(len(yy),3,replace=len(yy)<3)
                points[n,c,zz]=np.stack([xx[ids],yy[ids]],-1)
    affine=np.diag([1.2,1.2,4.,1.]).astype(np.float32)
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(path,images=np.asarray(images),labels=labels,points=points,affine=affine,patient_id=np.arange(N),split=np.array(['train']*3+['dev','test']),seed=seed)

if __name__=='__main__':generate(Path(__file__).resolve().parents[1]/'data/sample.npz')
