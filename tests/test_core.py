import unittest,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from src.core import *
torch.set_num_threads(1)
class Contracts(unittest.TestCase):
 def setUp(self):torch.manual_seed(13);self.a=torch.diag(torch.tensor([1.,2.,4.,1.]))
 def test_physical_axis_and_centres(self):
  x=coords((4,5,6))[...,0][None];u=torch.zeros(4,5,6,3);u[...,0]=1
  y,v=read(x,u,self.a);self.assertTrue(torch.allclose(y[0,:,:,:-1],x[0,:,:,1:],atol=1e-5));self.assertFalse(v[:,:,-1].any())
 def test_composition_not_negation(self):
  u=torch.zeros(4,5,6,3);u[...,0]=.5;v=u.clone();c,_=compose(u,v,self.a)
  self.assertTrue(torch.allclose(c[:,:,:-1,0],torch.ones(4,5,5)))
 def test_gate_detached_coordinates_differentiable(self):
  u=torch.zeros(4,5,6,3,requires_grad=True);w,_=reliability(u,-u,self.a,1)
  self.assertFalse(w.requires_grad);x=coords((4,5,6))[...,0][None];y,_=read(x,u,self.a);y[:,:,:,:-1].sum().backward();self.assertGreater(u.grad.abs().sum(),0)
 def test_empty_attention_and_scaling(self):
  q=torch.randn(2,4);k=torch.randn(3,4);v=torch.randn(3,4);valid=torch.ones(3,dtype=torch.bool)
  a=weighted_attention(q,k,v,torch.ones(3),valid);b=weighted_attention(q,k,v,torch.full((3,),.2),valid)
  self.assertTrue(torch.allclose(b,.2*a));self.assertEqual(weighted_attention(q,k,v,torch.ones(3),~valid).abs().sum(),0)
 def test_seed_nonoverlap_and_points_only(self):
  points=torch.full((3,4,1,2),-1.);points[0,0,0]=torch.tensor([2.,2.]);s=hard_seed(points,(4,6,6));self.assertLessEqual(float(s.sum(0).max()),1);self.assertEqual(s[1:].sum(),0)
 def test_joint_vs_identity_registration_gradients(self):
  m=DAM();images=torch.rand(3,4,6,6);seed=torch.zeros(3,4,6,6);seed[0,:,2:4,2:4]=1
  for identity in (False,True):
   m.zero_grad();p,events=m(images,seed,self.a,identity);p[-1].sum().backward()
   norm=sum(float(x.grad.abs().sum()) for x in m.reg.parameters() if x.grad is not None)
   self.assertGreater(norm,0) if not identity else self.assertEqual(norm,0)
   self.assertTrue(all(x.grad is None for x in m.encoder.parameters()));self.assertEqual(events[-1]['read_phases'],[0,1])
 def test_soft_write_temporal_gradient(self):
  m=DAM();f=torch.randn(4,4,5,5);p=torch.rand(3,4,5,5,requires_grad=True);bank=m.encode_memory(f,p);saved=bank.detach().clone();u=torch.zeros(4,5,5,3)
  r,_=read(bank[0],u,self.a);r.sum().backward();self.assertGreater(p.grad.abs().sum(),0);self.assertTrue(torch.equal(bank.detach(),saved))
 def test_constant_lncc_undefined_logged(self):
  x=torch.ones(4,5,6);loss,n=lncc_loss(x,x,torch.zeros(4,5,6,3),self.a);self.assertEqual(n,0);self.assertEqual(loss,0)
 def test_future_frame_cannot_change_earlier_prediction(self):
  m=DAM();images=torch.rand(4,4,5,5);seed=torch.zeros(3,4,5,5);seed[0,:,1:3,1:3]=1;changed=images.clone();changed[-1]+=100
  with torch.no_grad():a,_=m(images,seed,self.a);b,_=m(changed,seed,self.a)
  self.assertTrue(torch.equal(a[:-1],b[:-1]))
 def test_bank_retains_ed_plus_recent_and_commits_whole_phase(self):
  m=DAM(max_recent=2);images=torch.rand(5,4,4,4);seed=torch.zeros(3,4,4,4);seed[0,:,1:3,1:3]=1
  with torch.no_grad():_,events=m(images,seed,self.a)
  self.assertEqual(events[-1]['read_phases'],[0,2,3]);self.assertTrue(all(e['write_after_slices']==4 and max(e['read_phases'])<e['phase'] for e in events))
 def test_target_position_axis_and_shape(self):
  p=target_positions(3,4,torch.zeros(1));self.assertEqual(tuple(p.shape),(12,4));self.assertTrue(torch.equal(p,target_positions(3,4,torch.zeros(1))));self.assertEqual(float(p[0,0]),-1);self.assertEqual(float(p[-1,0]),1)
if __name__=='__main__':unittest.main()
