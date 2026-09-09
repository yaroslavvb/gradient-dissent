"""Compare CPU-reconstructed initial states, without training or test data.

Run inside an already allocated worker after both variant states are saved in
/work/initialization-audit. This is a numerical/provenance audit, not a GPU job.
"""
import hashlib
import json
from pathlib import Path
import torch

ROOT=Path('/work/initialization-audit')
THRESHOLDS={'max_absolute_difference':1e-7,'relative_l2_difference':1e-6}
torch.set_num_threads(2)
def digest_state(state):
    h=hashlib.sha256()
    for name,value in state.items():
        h.update(name.encode());h.update(value.numpy().tobytes())
    return h.hexdigest()

reports=[]
for seed in (1000,2000,2001,2002):
    aa=torch.load(ROOT/'avx512'/f'{seed}.pt',map_location='cpu',weights_only=True)
    bb=torch.load(ROOT/'avx2'/f'{seed}.pt',map_location='cpu',weights_only=True)
    assert list(aa)==list(bb)
    rows=[];sqdiff=0.;sqref=0.;maximum=0.;changed=0;total=0
    for name,a in aa.items():
        b=bb[name];assert a.shape==b.shape and a.dtype==b.dtype
        ad=a.double();bd=b.double();d=ad-bd
        absolute=float(d.abs().max());ds=float((d*d).sum());rs=float((ad*ad).sum());n=int((a!=b).sum())
        maximum=max(maximum,absolute);sqdiff+=ds;sqref+=rs;changed+=n;total+=a.numel()
        rows.append({'name':name,'shape':list(a.shape),'dtype':str(a.dtype),'numel':a.numel(),'changed_elements':n,'max_absolute_difference':absolute,'squared_difference':ds,'squared_reference_norm':rs})
    relative=(sqdiff/sqref)**.5
    reports.append({'task':'convnext_cifar100','seed':seed,'state_hashes':[digest_state(aa),digest_state(bb)],'parameters':total,'changed_elements':changed,'max_absolute_difference':maximum,'relative_l2_difference':relative,'passed':maximum<=THRESHOLDS['max_absolute_difference'] and relative<=THRESHOLDS['relative_l2_difference'],'tensors':rows})
    del aa,bb,ad,bd,d
result={'purpose':'Full-tensor comparison of same-seed CPU initializations reconstructed from the frozen source on two existing workers; no trained weights or test outcomes used.','thresholds':THRESHOLDS,'torch':str(torch.__version__),'vision_source_sha256':hashlib.sha256(Path('/opt/a100_transfer/vision.py').read_bytes()).hexdigest(),'rows':reports,'passed':all(r['passed'] for r in reports)}
print(json.dumps(result))
