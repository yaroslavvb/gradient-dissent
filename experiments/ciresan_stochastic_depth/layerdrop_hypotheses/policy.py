"""Predeclared image-only routing, fitted/calibrated without test outcomes.

Training needs scikit-learn; exported-tree application needs NumPy only. Mask
bits indicate present body branches; mask15 is dense. No dense forward pass,
margin, label, or pruned prediction is used by apply_policy.
"""
import hashlib
import math

import numpy as np

LAMBDAS = (0., .001, .002, .005, .01, .02, .05, .1, .2, .5, 1.)
FIT_N = 5000
CALIBRATION_N = 5000
EXCESS_ERROR_TOLERANCE = .002
FULL_MASK = 15
RANDOM_STATE = 20260909
VALIDATION_ORDER_SEED = 20260910
RANDOM_COMPARATOR_SEED = 20260912


def pooled_image_features(images):
    """49 means of nonoverlapping 4x4 raw-pixel patches, divided by255."""
    images=np.asarray(images)
    if images.ndim==4 and images.shape[1:]==(1,28,28): images=images[:,0]
    elif images.ndim==2 and images.shape[1]==784: images=images.reshape(-1,28,28)
    if images.ndim!=3 or images.shape[1:]!=(28,28) or not np.isfinite(images).all():
        raise ValueError('Expected finite raw images [N,28,28], [N,1,28,28], or [N,784]')
    if np.any(images<0) or np.any(images>255): raise ValueError('Expected raw0..255 pixels')
    return (images.astype(np.float64).reshape(-1,7,4,7,4).mean(axis=(2,4))/255.).reshape(-1,49).astype(np.float32)


def _features(features):
    x=np.asarray(features)
    if x.ndim!=2 or x.shape[1]!=49 or not np.isfinite(x).all() or np.any(x<0) or np.any(x>1):
        raise ValueError('Expected finite normalized image features [N,49] in [0,1]')
    return np.asarray(x,dtype=np.float32)


def _predictions(predictions, labels, n):
    pred=np.asarray(predictions);y=np.asarray(labels)
    if pred.shape!=(16,n) or y.shape!=(n,) or not np.issubdtype(pred.dtype,np.integer) or not np.issubdtype(y.dtype,np.integer):
        raise ValueError('Need integer mask predictions [16,N] and labels [N]')
    if np.any(pred<0) or np.any(pred>9) or np.any(y<0) or np.any(y>9):
        raise ValueError('Expected MNIST class IDs0..9')
    return pred.astype(np.int64,copy=False), y.astype(np.int64,copy=False)


def _costs(costs):
    if isinstance(costs,(list,tuple)) and costs and isinstance(costs[0],dict):
        if [row['mask_id'] for row in costs]!=list(range(16)): raise ValueError('Cost rows must have mask IDs0..15')
        costs=[row['cost'] for row in costs]
    c=np.asarray(costs,dtype=np.float64)
    if c.shape!=(16,) or not np.isfinite(c).all() or np.any(c<0) or c[15]<=0 or np.any(c>c[15]):
        raise ValueError('Need16 finite nonnegative costs, with positive dense cost maximal')
    return c/c[15]


def _digest(array):
    value=np.ascontiguousarray(array)
    return hashlib.sha256(str(value.dtype).encode()+str(value.shape).encode()+value.tobytes()).hexdigest()


def _export_tree(estimator):
    tree=estimator.tree_
    return {'children_left':tree.children_left.tolist(),'children_right':tree.children_right.tolist(),
            'feature':tree.feature.tolist(),'threshold':tree.threshold.tolist(),
            'values':tree.value[:,:,0].tolist(), 'n_node_samples':tree.n_node_samples.tolist(),
            'feature_dtype':'float32','comparison_dtype':'float64','n_features':49,'n_outputs':16}


def predict_extra_error(model,features):
    """Pure NumPy tree traversal; always force dense predicted extra error0."""
    x=_features(features);tree=model['tree']
    left=np.asarray(tree['children_left'],dtype=np.int64);right=np.asarray(tree['children_right'],dtype=np.int64)
    feat=np.asarray(tree['feature'],dtype=np.int64);threshold=np.asarray(tree['threshold'],dtype=np.float64)
    values=np.asarray(tree['values'],dtype=np.float64)
    if values.shape!=(len(left),16) or right.shape!=left.shape or feat.shape!=left.shape or threshold.shape!=left.shape or not np.isfinite(values).all():
        raise ValueError('Malformed serialized tree')
    nodes=np.zeros(len(x),dtype=np.int64)
    # A fitted depth3 tree takes at most3 branches;4 iterations also checks leaves.
    for _ in range(4):
        active=np.flatnonzero(left[nodes]!=-1)
        if not len(active):break
        node=nodes[active];feature=feat[node]
        if np.any(feature<0) or np.any(feature>=49):raise ValueError('Invalid split feature')
        go_left=x[active,feature].astype(np.float64)<=threshold[node]
        nodes[active]=np.where(go_left,left[node],right[node])
        if np.any(nodes<0) or np.any(nodes>=len(left)):raise ValueError('Invalid child index')
    if np.any(left[nodes]!=-1):raise ValueError('Serialized tree exceeds max_depth3 or has a cycle')
    predicted=values[nodes].copy();predicted[:,15]=0.
    return predicted


def choose_masks(predicted_extra_error,cost_fractions,penalty):
    """Minimize predicted extra error+lambda*cost, ties favor higher cost."""
    p=np.asarray(predicted_extra_error,dtype=np.float64);c=_costs(cost_fractions);penalty=float(penalty)
    if p.ndim!=2 or p.shape[1]!=16 or not np.isfinite(p).all() or not math.isfinite(penalty) or penalty<0:
        raise ValueError('Invalid predicted errors or lambda')
    p=p.copy();p[:,15]=0.
    # Highest mask ID breaks any remaining equal-cost tie, ensuring dense wins.
    priority=np.lexsort((-np.arange(16),-c))
    score=p+penalty*c[None,:]
    return priority[np.argmin(score[:,priority],axis=1)].astype(np.int64)


def apply_policy(model,features):
    """Route only from49 image features; no label/prediction/margin arguments."""
    x=_features(features)
    if model['chosen_lambda'] is None:return np.full(len(x),15,dtype=np.int64)
    return choose_masks(predict_extra_error(model,x),model['cost_fractions'],model['chosen_lambda'])


def _calibration_row(mask_ids,pred,y,cost,penalty,kind):
    selected=pred[mask_ids,np.arange(len(y))];errors=int(np.sum(selected!=y));dense_errors=int(np.sum(pred[15]!=y))
    excess_count=errors-dense_errors
    return {'kind':kind,'lambda':penalty,'n':len(y),'errors':errors,'dense_errors':dense_errors,
            'accuracy':1-errors/len(y),'excess_errors':excess_count,'excess_error_rate':excess_count/len(y),
            'mean_cost_fraction':float(cost[mask_ids].mean()),'mac_saving_fraction':float(1-cost[mask_ids].mean()),
            'feasible':excess_count<=EXCESS_ERROR_TOLERANCE*len(y)+1e-12,
            'mask_counts':np.bincount(mask_ids,minlength=16).tolist()}


def _select_candidate(rows):
    feasible=[r for r in rows if r['feasible']]
    # Fallback is considered after an exactly tied lambda0 candidate; both
    # produce the same calibration accuracy/cost if they tie the dense policy.
    return max(feasible,key=lambda r:(r['mac_saving_fraction'],r['accuracy'],
                                     -float('inf') if r['lambda'] is None else -r['lambda']))


def fit_policy(val_features,val_pred,val_labels,costs):
    """Fit first5000, calibrate last5000 of caller-reordered validation data.

    Caller supplies the same torch.randperm(10000, seed20260910) permutation
    for features, all16 prediction rows, and labels. This function does not
    apply a second shuffle.
    """
    from sklearn.tree import DecisionTreeRegressor
    import sklearn
    x=_features(val_features)
    if len(x)!=FIT_N+CALIBRATION_N:raise ValueError('Protocol requires exactly10000 validation examples')
    pred,y=_predictions(val_pred,val_labels,len(x));cost=_costs(costs)
    wrong=(pred!=y[None,:]).astype(np.float64)
    targets=(wrong-wrong[15]).T
    tree=DecisionTreeRegressor(max_depth=3,min_samples_leaf=200,random_state=RANDOM_STATE)
    tree.fit(x[:FIT_N],targets[:FIT_N])
    result={'schema_version':1,'tree':_export_tree(tree),'cost_fractions':cost.tolist(),
            'mask_tie_priority':np.lexsort((-np.arange(16),-cost)).tolist(),
            'decision_score_dtype':'float64',
            'chosen_lambda':None,'chosen_policy_kind':'dense_fallback',
            'protocol':{'feature':'49 raw-image4x4 patch means divided by255;7x7 grid, row-major',
                        'fit_indices':'[0,5000)','calibration_indices':'[5000,10000)',
                        'input_order':'Caller already applied torch.randperm(10000,seed20260910) identically to features/predictions/labels; no additional shuffle here',
                        'validation_order_seed':VALIDATION_ORDER_SEED,
                        'max_depth':3,'min_samples_leaf':200,'random_state':RANDOM_STATE,
                        'lambdas':list(LAMBDAS),'excess_error_tolerance':EXCESS_ERROR_TOLERANCE,
                        'target':'mask0/1error minus dense0/1error;16 outputs; dense forced0',
                        'selection':'Max calibration MAC saving with excess error<=.002; then higher accuracy; then lower lambda. Explicit dense fallback.',
                        'apply_inputs':'Image features only; neither labels nor model logits/margins'},
            'provenance':{'sklearn_version':sklearn.__version__,'features_sha256':_digest(x),
                          'mask_predictions_sha256':_digest(pred),'labels_sha256':_digest(y)}}
    prediction=predict_extra_error(result,x[FIT_N:])
    np.testing.assert_allclose(prediction,tree.predict(x[FIT_N:]),rtol=0,atol=0)
    cpred=pred[:,FIT_N:];cy=y[FIT_N:]
    candidates=[_calibration_row(choose_masks(prediction,cost,l),cpred,cy,cost,l,'tree') for l in LAMBDAS]
    candidates.append(_calibration_row(np.full(CALIBRATION_N,15),cpred,cy,cost,None,'dense_fallback'))
    winner=_select_candidate(candidates)
    result.update(chosen_lambda=winner['lambda'],chosen_policy_kind=winner['kind'],
                  calibration={'candidates':candidates,'selected':winner})
    static=[]
    for mask in range(16):
        row=_calibration_row(np.full(CALIBRATION_N,mask),cpred,cy,cost,None,'static')
        row['mask_id']=mask;static.append(row)
    selected=max((r for r in static if r['feasible']),key=lambda r:(r['mac_saving_fraction'],r['accuracy'],r['mask_id']))
    result['static_comparator']={'selected_mask_id':selected['mask_id'],'calibration_selected':selected,'calibration_candidates':static,
                                 'selection':'Same calibration excess-error tolerance, maximum saving, then higher accuracy, then higher mask ID'}
    return result


def _paired_interval(error_delta):
    delta=np.asarray(error_delta,dtype=np.float64);n=len(delta);mean=float(delta.mean())
    sd=float(delta.std(ddof=1)) if n>1 else None
    half=1.959963984540054*sd/math.sqrt(n) if n>1 else None
    return {'n':n,'mean_error_delta':mean,'sample_sd':sd,'error_delta_ci95':[mean-half,mean+half] if half is not None else None,
            'accuracy_delta_pp':-100*mean,'accuracy_delta_pp_ci95':[-100*(mean+half),-100*(mean-half)] if half is not None else None,
            'method':'Paired-example normal approximation using sample SD of {-1,0,1} error differences; descriptive on a reused MNIST test set, not seed or dataset replication; no selection correction. Zero observed discordances gives a degenerate Wald interval, not proof of population equivalence.'}


def evaluate_routing(mask_ids,predictions,labels,costs):
    """Score a fixed policy's mask IDs; does not fit or alter the policy."""
    ids=np.asarray(mask_ids);n=len(ids)
    if ids.ndim!=1 or not n or not np.issubdtype(ids.dtype,np.integer) or np.any(ids<0) or np.any(ids>15):
        raise ValueError('Need nonempty integer mask IDs0..15')
    pred,y=_predictions(predictions,labels,n);cost=_costs(costs)
    selected=pred[ids,np.arange(n)];wrong=selected!=y;dense_wrong=pred[15]!=y
    harm=wrong & ~dense_wrong;repair=~wrong & dense_wrong
    per_class=[]
    for cls in range(10):
        take=y==cls;count=int(take.sum())
        per_class.append({'class_id':cls,'n':count,'accuracy':float((~wrong[take]).mean()) if count else None,
                          'dense_accuracy':float((~dense_wrong[take]).mean()) if count else None,
                          'accuracy_delta_pp':float(100*(dense_wrong[take].mean()-wrong[take].mean())) if count else None,
                          'mac_saving_fraction':float(1-cost[ids[take]].mean()) if count else None,
                          'dense_correct_harmed':int(harm[take].sum()),'dense_wrong_repaired':int(repair[take].sum())})
    return {'n':n,'errors':int(wrong.sum()),'accuracy':float((~wrong).mean()),
            'dense_errors':int(dense_wrong.sum()),'dense_accuracy':float((~dense_wrong).mean()),
            'accuracy_delta_pp':float(100*(dense_wrong.mean()-wrong.mean())),
            'mean_cost_fraction':float(cost[ids].mean()),'mac_saving_fraction':float(1-cost[ids].mean()),
            'mask_counts':np.bincount(ids,minlength=16).tolist(),'mask_shares':(np.bincount(ids,minlength=16)/n).tolist(),
            'dense_correct_harmed':int(harm.sum()),'dense_wrong_repaired':int(repair.sum()),
            'harm_given_dense_correct':float(harm.sum()/(~dense_wrong).sum()) if (~dense_wrong).any() else None,
            'paired_error_interval':_paired_interval(wrong.astype(float)-dense_wrong.astype(float)),
            'per_class':per_class,
            'cost_scope':'Nominal affine MAC saving; image feature extraction, tree routing, batching and execution overhead excluded. Dense predictions are a scoring reference only, not used at application.'}


def random_cost_matched_masks(mask_ids):
    """Label-free shuffle of the exact test mask multiset, with fixed RNG."""
    masks=np.asarray(mask_ids)
    if masks.ndim!=1 or not np.issubdtype(masks.dtype,np.integer) or np.any(masks<0) or np.any(masks>15):
        raise ValueError('Expected integer mask IDs0..15')
    random_masks=masks[np.random.default_rng(RANDOM_COMPARATOR_SEED).permutation(len(masks))]
    np.testing.assert_array_equal(np.bincount(masks,minlength=16),np.bincount(random_masks,minlength=16))
    return random_masks


def evaluate_policy(model,test_features,test_pred,test_labels):
    """Score frozen routes, static mask, dense, and shuffled-route comparator."""
    masks=apply_policy(model,test_features);cost=model['cost_fractions'];n=len(masks)
    static_id=model['static_comparator']['selected_mask_id']
    random_masks=random_cost_matched_masks(masks)
    return {'routed':evaluate_routing(masks,test_pred,test_labels,cost),
            'static_calibrated':{'mask_id':static_id,**evaluate_routing(np.full(n,static_id),test_pred,test_labels,cost)},
            'dense':evaluate_routing(np.full(n,15),test_pred,test_labels,cost),
            'random_cost_matched':{'seed':RANDOM_COMPARATOR_SEED,
                                   'assignment':'NumPy default_rng(seed).permutation of the exact test routing-mask multiset; uses no labels or prediction values',
                                   **evaluate_routing(random_masks,test_pred,test_labels,cost)},
            'policy_chosen_lambda':model['chosen_lambda'],'policy_kind':model['chosen_policy_kind'],
            'evaluation_rule':'Model frozen using validation only; this function never changes fit, lambda or calibration. The caller must not retune from this test report.'}
