from pathlib import Path
import sys,argparse,json,time,hashlib
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from src.core import DAM,hard_seed,hard_labels,freeze,segmentation_loss,registration_loss
from src.data import generate

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=ROOT/'outputs');a=parser.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    c=json.loads((ROOT/'config.json').read_text(encoding='utf-8'));torch.set_num_threads(1);torch.manual_seed(c['seed']);np.random.seed(c['seed']);start=time.time()
    path=ROOT/'data/sample.npz'
    if not path.exists():generate(path,c['data_seed'])
    d=np.load(path);im=torch.from_numpy(d['images']);labels=torch.from_numpy(d['labels']);aff=torch.from_numpy(d['affine']);points=torch.from_numpy(d['points'])
    model=DAM(c['tau_mm']);history=[]
    for stage,epochs in [('registration',c['registration_epochs']),('memory',c['memory_epochs']),('joint',c['joint_epochs'])]:
        freeze(model,True);freeze(model.reg,stage=='memory')
        if stage!='registration':
            for name,module in model.named_children():
                if name not in ('encoder','reg'):freeze(module,False)
        optimizer=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=c['lr']*(.3 if stage=='joint' else 1))
        for epoch in range(epochs):
            losses=[]
            for n in np.where(d['split']=='train')[0]:
                seed=hard_seed(points[n,:,:,:c['point_budget']],im[n,0].shape)
                optimizer.zero_grad();reg,nwin=registration_loss(model,im[n],aff)
                if stage=='registration':loss=reg
                else:
                    pred,_=model(im[n],seed,aff);y=torch.stack([(labels[n]==k).float() for k in (1,2,3)],1)
                    loss=segmentation_loss(pred[[0,-1]],y[[0,-1]])+.1*reg
                loss.backward();optimizer.step();losses.append(float(loss.detach()))
            history.append({'stage':stage,'epoch':epoch,'loss':sum(losses)/len(losses),'valid_lncc_windows_last_patient':nwin})
    model.eval();n=int(np.where(d['split']=='test')[0][0]);seed=hard_seed(points[n,:,:,:c['point_budget']],im[n,0].shape)
    with torch.no_grad():p,events=model(im[n],seed,aff)
    pred=hard_labels(p[-1]);gt=labels[n,-1];dice=[]
    for k in (1,2,3):
        aa=pred==k;bb=gt==k;dice.append(float((2*(aa&bb).sum()+1e-5)/(aa.sum()+bb.sum()+1e-5)))
    voxel_ml=float(torch.det(aff[:3,:3]).abs())/1000;ved=float(seed[0].sum())*voxel_ml;ves=float((pred==1).sum())*voxel_ml
    metrics={'status':'synthetic smoke test; NOT paper experiment','seed':c['seed'],'training_patient_ids':[0,1,2],'test_patient_id':n,'synthetic_es_dice_LV_RV_MYO':dice,'ed_seed_ml':ved,'es_prediction_ml':ves,'synthetic_lvef_pct':100*(ved-ves)/ved if ved else None,'ed_zero_volume_failure':ved==0,'es_empty_foreground_diagnostic':bool((pred==0).all()),'training_history':history,'bank_events':events,'seconds':time.time()-start,'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'checkpoint_substitutions':['frozen tiny encoder instead of SAM 2.1','image-conditioned translation-only regressor instead of SACB-Net','target-grid four-channel positional carrier; not SAM rotary/temporal embeddings','native identity preprocessing, no artificial padding']}
    np.savez_compressed(a.output/'predictions.npz',probabilities=p.numpy(),hard_es=pred.numpy(),ed_seed=seed.numpy(),evaluation_reference=gt.numpy())
    torch.save(model.state_dict(),a.output/'tiny_model.pt');(a.output/'metrics.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8');print(json.dumps({'status':metrics['status'],'seconds':metrics['seconds'],'output':str(a.output.resolve())}))
if __name__=='__main__':main()
