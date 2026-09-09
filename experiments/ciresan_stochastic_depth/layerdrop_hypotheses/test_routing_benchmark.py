"""CPU-only parity of exported routing, feature rounding, and grouped IO."""
import copy
import unittest

import numpy as np
import torch

try:
    from . import policy
    from . import routing_benchmark as rb
except ImportError:
    import policy
    import routing_benchmark as rb


def serialized_policy():
    lower=np.float32(.5);upper=np.nextafter(lower,np.float32(1.))
    values=np.zeros((3,16));values[1,5]=-.2;values[2,11]=-.2
    cost=np.linspace(.1,1,16)
    return {'chosen_lambda':.01,'cost_fractions':cost.tolist(),'mask_tie_priority':list(range(15,-1,-1)),
            'static_comparator':{'selected_mask_id':7},
            'tree':{'children_left':[1,-1,-1],'children_right':[2,-1,-1],'feature':[0,-2,-2],
                    'threshold':[(float(lower)+float(upper))/2,-2.,-2.],'values':values.tolist(),
                    'n_node_samples':[1000,500,500]}}


def artificial_forward(model,images,mask):
    # Distinct row-dependent outputs verify that grouping preserves row order.
    value=images.reshape(len(images),-1)
    return torch.stack((value[:,0]+mask,value[:,1]-mask,value[:,2]*(mask+1)),dim=-1)


class RoutingParity(unittest.TestCase):
    def test_all_raw_patch_sums_have_exact_numpy_torch_features(self):
        # Enumerate all4081 possible integer sums in a4x4 patch; this catches
        # one-ULP divide-by255 differences that could cross learned thresholds.
        images=np.zeros((4081,28,28),dtype=np.uint8)
        for total in range(4081):
            full,remainder=divmod(total,255)
            patch=np.zeros(16,dtype=np.uint8);patch[:full]=255
            if full<16:patch[full]=remainder
            images[total,:4,:4]=patch.reshape(4,4)
        actual=rb.torch_image_features(torch.from_numpy(images).float()).numpy()
        expected=policy.pooled_image_features(images)
        np.testing.assert_array_equal(actual,expected)

    def test_float64_threshold_comparison_matches_numpy_between_float32_values(self):
        model=serialized_policy();features=np.zeros((6,49),dtype=np.float32)
        features[:,0]=[0,.5,np.nextafter(np.float32(.5),np.float32(1)),1,.1,.9]
        routed=rb.TorchPolicy(model,'cpu').route_features(torch.from_numpy(features)).numpy()
        np.testing.assert_array_equal(routed,policy.apply_policy(model,features))
        np.testing.assert_array_equal(routed,[5,5,11,11,5,11])

    def test_full_three_level_tree_and_float64_cost_ties(self):
        model=serialized_policy();values=np.zeros((15,16))
        for node in range(7,15):values[node,node-7]=-.1
        model['tree']={'children_left':[2*i+1 if i<7 else -1 for i in range(15)],
                       'children_right':[2*i+2 if i<7 else -1 for i in range(15)],
                       'feature':[i%3 if i<7 else -2 for i in range(15)],
                       'threshold':[.5]*15,'values':values.tolist()}
        features=np.random.default_rng(3).uniform(size=(400,49)).astype(np.float32)
        np.testing.assert_array_equal(rb.TorchPolicy(model,'cpu').route_features(torch.from_numpy(features)).numpy(),policy.apply_policy(model,features))
        model['chosen_lambda']=0.;model['tree']['values']=np.zeros((15,16)).tolist()
        np.testing.assert_array_equal(rb.TorchPolicy(model,'cpu').route_features(torch.from_numpy(features)).numpy(),np.full(400,15))

    def test_group_gather_scatter_all16_masks_and_row_order(self):
        images=torch.arange(37*3,dtype=torch.float32).reshape(37,3)
        masks=torch.tensor(np.random.default_rng(2).permutation(np.arange(37)%16))
        result=rb.grouped_forward(None,images,masks,artificial_forward)
        expected=torch.cat([artificial_forward(None,images[i:i+1],int(masks[i])) for i in range(len(images))])
        torch.testing.assert_close(result,expected,atol=0,rtol=0)
        for mask in [0,15]:
            same=rb.grouped_forward(None,images,torch.full((len(images),),mask),artificial_forward)
            torch.testing.assert_close(same,artificial_forward(None,images,mask),atol=0,rtol=0)

    def test_complete_methods_match_reference_with_incomplete_last_batch(self):
        raw=np.random.default_rng(9).integers(0,256,size=(19,28,28),dtype=np.uint8)
        raw[:10,:4,:4]=0;raw[10:,:4,:4]=255
        x=torch.from_numpy(raw).float();model=serialized_policy();router=rb.TorchPolicy(model,'cpu')
        cpu_masks=policy.apply_policy(model,policy.pooled_image_features(raw))
        random_masks=policy.random_cost_matched_masks(cpu_masks)
        for method in rb.METHODS:
            pred,ids=rb._predict_pass(method,None,x,router,7,torch.from_numpy(random_masks),artificial_forward,batch_size=7)
            expected_masks={'dense':np.full(19,15),'static':np.full(19,7),'routed':cpu_masks,'random_cost_matched':random_masks}[method]
            np.testing.assert_array_equal(ids.numpy(),expected_masks)
            expected=torch.cat([artificial_forward(None,x[i:i+1],int(mask)) for i,mask in enumerate(expected_masks)]).argmax(-1)
            torch.testing.assert_close(pred,expected,atol=0,rtol=0)

    def test_dense_fallback_and_identity_parsing(self):
        model=serialized_policy();model['chosen_lambda']=None
        x=torch.ones((3,28,28))
        torch.testing.assert_close(rb.TorchPolicy(model,'cpu')(x),torch.full((3,),15),atol=0,rtol=0)
        self.assertEqual(rb._record_identity('sd_constant-selected-s102',{}),('sd_constant',102))
        self.assertEqual(rb._record_identity('102/sd_constant-selected',{}),('sd_constant',102))
        self.assertEqual(rb._record_identity('arbitrary',{'recipe':'residual','seed':103,'endpoint':'selected'}),('residual',103))
        with self.assertRaises(ValueError):rb._record_identity('103/residual-final',{})

    def test_invalid_tree_depth_and_priority_rejected(self):
        bad=copy.deepcopy(serialized_policy());bad['mask_tie_priority']=list(range(16))
        with self.assertRaises(ValueError):rb.TorchPolicy(bad,'cpu')
        bad=copy.deepcopy(serialized_policy());bad['tree']['children_left'][0]=0
        with self.assertRaises(ValueError):rb.TorchPolicy(bad,'cpu')


if __name__=='__main__':
    unittest.main()
