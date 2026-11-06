"""Small DAM carrier; no SAM/SACB weights or clinical claims."""
import torch
from torch import nn
from torch.nn import functional as F

def freeze(module, flag=True):
    for p in module.parameters(): p.requires_grad_(not flag)

def coords(shape, device=None, dtype=torch.float32):
    z,y,x=torch.meshgrid(*[torch.arange(n,device=device,dtype=dtype) for n in shape],indexing='ij')
    return torch.stack([x,y,z],-1)

def sampling_grid(u_mm, affine_src, affine_tgt):
    """u [D,H,W,3], world xyz mm; tensors stored D,H,W; voxel centres."""
    q=coords(u_mm.shape[:3],u_mm.device,u_mm.dtype)
    world=q@affine_tgt[:3,:3].T+affine_tgt[:3,3]+u_mm
    inv=torch.linalg.inv(affine_src)
    xi=world@inv[:3,:3].T+inv[:3,3]
    n=u_mm.new_tensor([u_mm.shape[2],u_mm.shape[1],u_mm.shape[0]])
    valid=((xi>=0)&(xi<=n-1)).all(-1)
    return 2*(xi+.5)/n-1,valid

def read(source,u_mm,affine):
    grid,valid=sampling_grid(u_mm,affine,affine)
    out=F.grid_sample(source[None],grid[None],mode='bilinear',padding_mode='zeros',align_corners=False)[0]
    return out,valid

def compose(u_target_mid,u_mid_source,affine):
    """Target->mid then mid->source, in common physical coordinates."""
    w,valid=read(u_mid_source.permute(3,0,1,2),u_target_mid,affine)
    out=u_target_mid+w.permute(1,2,3,0)
    _,final_valid=sampling_grid(out,affine,affine)
    return out,valid&final_valid

def reliability(forward,reverse,affine,tau):
    rev,v1=read(reverse.permute(3,0,1,2),forward,affine)
    residual=(forward+rev.permute(1,2,3,0)).norm(dim=-1)
    _,v2=sampling_grid(forward,affine,affine)
    _,v3=sampling_grid(forward+rev.permute(1,2,3,0),affine,affine)
    # Fixed validity is detached; source grids in this demo have no artificial padding.
    valid=(v1&v2&v3).detach()
    return (valid*torch.exp(-residual.square()/(2*tau*tau))).detach(),valid

def target_positions(height,width,reference):
    # Generated AFTER reading, never sampled from source positional encodings.
    yy,xx=torch.meshgrid(torch.linspace(-1,1,height),torch.linspace(-1,1,width),indexing='ij')
    return torch.stack([xx,yy,torch.sin(torch.pi*xx),torch.sin(torch.pi*yy)],-1).reshape(-1,4).to(reference)

def weighted_attention(q,k,v,weight,valid):
    """Projected V scaling; safe empty residual, never all-masked softmax."""
    if not bool(valid.any()): return q.new_zeros(q.shape)
    attn=(q@k[valid].T/q.shape[-1]**.5).softmax(-1)
    return attn@(v[valid]*weight[valid,None])

def hard_labels(p):
    value,index=p.max(0)  # torch max uses first class for ties: LV,RV,myocardium
    return torch.where(value>=.5,index+1,torch.zeros_like(index))

def hard_seed(points,shape,radius=1.5):
    """Frozen point-only Gaussian initializer. No full mask or later-phase argument."""
    q=coords(shape);out=torch.zeros(3,*shape)
    for c in range(3):
        for z in range(shape[0]):
            chosen=points[c,z]
            chosen=chosen[(chosen>=0).all(-1)]
            if len(chosen):
                d=((q[z,...,:2].unsqueeze(-2)-chosen[None,None])**2).sum(-1)
                out[c,z]=torch.exp(-d.min(-1).values/(2*radius**2))
    labels=hard_labels(out)
    return torch.stack([(labels==c+1).float() for c in range(3)])

class TinyRegistration(nn.Module):
    def __init__(self):
        super().__init__();self.net=nn.Sequential(nn.Conv3d(2,8,3,padding=1),nn.Tanh(),nn.Conv3d(8,3,1))
    def forward(self,source,target):
        # Image-derived translation-only approximation, NOT SACB cluster kernels.
        v=self.net(torch.stack([source,target])[None]).mean((2,3,4))[0].tanh()*1.5
        return v.expand(*target.shape,3)

class DAM(nn.Module):
    def __init__(self,tau=1.5,max_recent=6):
        super().__init__();self.tau=tau;self.max_recent=max_recent
        self.encoder=nn.Sequential(nn.Conv3d(1,4,3,padding=1),nn.Tanh())
        freeze(self.encoder);self.reg=TinyRegistration()
        self.memory=nn.Conv3d(5,4,1);self.q=nn.Linear(4,4);self.k=nn.Linear(4,4);self.v=nn.Linear(4,4)
        self.decoder=nn.Conv2d(8,1,1);self.ed_head=nn.Conv3d(5,1,1)
    def encode_memory(self,features,prob):
        return torch.stack([self.memory(torch.cat([features,p[None]],0)[None])[0] for p in prob])
    def forward(self,images,seed,affine,identity=False):
        T,D,H,W=images.shape
        features=[self.encoder(x[None,None])[0] for x in images]
        bank={0:self.encode_memory(features[0],seed)}
        pointers={0:torch.stack([(features[0]*p[None]).mean((-1,-2)).T for p in seed])}
        initial=torch.cat([self.ed_head(torch.cat([features[0],p[None]],0)[None]).sigmoid() for p in seed],1)[0]
        preds=[initial];fw={};rv={};events=[]
        for t in range(1,T):
            fw[t]=self.reg(images[t-1],images[t]);rv[t]=self.reg(images[t],images[t-1])
            per_class=[]
            selected=sorted(bank)
            assert all(s<t for s in selected)
            events.append({'phase':t,'read_phases':selected.copy(),'write_after_slices':D})
            for c in range(3):
                reads=[];weights=[];valids=[];ptr=[]
                for s in selected:
                    u=fw[t];reverse=rv[s+1]
                    _,uv=sampling_grid(u,affine,affine);_,rv_valid=sampling_grid(reverse,affine,affine)
                    for k in range(t-1,s,-1):
                        u,step_valid=compose(u,fw[k],affine);uv=uv&step_valid
                    for k in range(s+2,t+1):
                        reverse,step_valid=compose(reverse,rv[k],affine);rv_valid=rv_valid&step_valid
                    if identity:
                        u=u*0;reverse=reverse*0;uv=torch.ones_like(uv);rv_valid=torch.ones_like(rv_valid)
                    gate,valid=reliability(u,reverse,affine,self.tau)
                    roundtrip_valid,_=read(rv_valid.float()[None],u,affine)
                    valid=valid&uv&(roundtrip_valid[0]>.999)
                    gate=gate*valid.detach()
                    m,inside=read(bank[s][c],u,affine)
                    reads.append(m);weights.append(gate);valids.append(valid&inside);ptr.append(pointers[s][c])
                slices=[]
                for z in range(D):
                    spatial=torch.cat([m[:,z].flatten(1).T for m in reads])
                    p=torch.stack([a[z] for a in ptr]) # nominal-slice pointers never warped
                    kv=torch.cat([spatial,p]);pos=target_positions(H,W,kv).repeat(len(reads),1);key_input=kv+torch.cat([pos,torch.zeros_like(p)]);w=torch.cat([g[z].flatten() for g in weights]+[p.new_ones(len(p))])
                    valid=torch.cat([g[z].flatten() for g in valids]+[torch.ones(len(p),dtype=torch.bool)])
                    query=features[t][:,z].flatten(1).T
                    att=weighted_attention(self.q(query+target_positions(H,W,query)),self.k(key_input),self.v(kv),w,valid)
                    conditioned=att.T.reshape(4,H,W)
                    slices.append(self.decoder(torch.cat([features[t][:,z],conditioned],0)[None]).sigmoid()[0,0])
                per_class.append(torch.stack(slices))
            pred=torch.stack(per_class);preds.append(pred)
            # Only after all objects/slices finished. Soft write retains autograd history.
            bank[t]=self.encode_memory(features[t],pred)
            pointers[t]=torch.stack([(features[t]*p[None]).mean((-1,-2)).T for p in pred])
            for stale in sorted([s for s in bank if s!=0])[:-self.max_recent]:
                del bank[stale];del pointers[stale]
        return torch.stack(preds),events

def segmentation_loss(p,y):
    bce=F.binary_cross_entropy(p,y)
    dice=1-(2*(p*y).sum((-1,-2,-3))+1e-5)/(p.sum((-1,-2,-3))+y.sum((-1,-2,-3))+1e-5)
    return bce+dice.mean()

def lncc_loss(source,target,u,affine):
    warped,valid=read(source[None],u,affine)
    x=warped[None];y=target[None,None];kernel=3
    mean=lambda a:F.avg_pool3d(a,kernel,stride=1)
    mx,my=mean(x),mean(y);vx=(mean(x*x)-mx*mx).clamp_min(0);vy=(mean(y*y)-my*my).clamp_min(0)
    ok=(mean(valid.float()[None,None])>.999)&(vx>1e-6)&(vy>1e-6)
    n=int(ok.sum())
    if not n:return u.sum()*0,0
    cov=mean(x*y)-mx*my
    return (1-cov.square()/(vx*vy+1e-6))[ok].mean(),n

def registration_loss(model,images,affine):
    terms=[];windows=0
    for t in range(1,len(images)):
        for s,a in [(t-1,t),(t,t-1)]:
            u=model.reg(images[s],images[a]);loss,n=lncc_loss(images[s],images[a],u,affine)
            spacing=affine[:3,:3].norm(dim=0)
            smooth=sum((torch.diff(u,dim=k)/spacing[2-k]).square().mean() for k in range(3))
            terms.append(loss+.01*smooth);windows+=n
    return torch.stack(terms).mean(),windows
