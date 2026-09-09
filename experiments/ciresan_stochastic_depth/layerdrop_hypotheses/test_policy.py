"""CPU checks of policy fitting, calibration, serialization, and held-out use."""
import copy
import inspect
import json
import unittest

import numpy as np

try:
    from . import policy as p
except ImportError:
    import policy as p


COST=np.linspace(.16,1,16)


def routing_fixture():
    x=np.zeros((10000,49),dtype=np.float32)
    x[:,0]=np.tile(np.repeat([0.,1.],2500),2)
    y=np.zeros(10000,dtype=np.int64)
    pred=np.ones((16,10000),dtype=np.int64)
    pred[15]=0;pred[0]=x[:,0].astype(np.int64)
    return x,pred,y


class FeatureAndDecisionChecks(unittest.TestCase):
    def test_pooled_features_exact_blocks_scaling_and_shapes(self):
        image=np.zeros((2,28,28),dtype=np.uint8)
        for i in range(7):
            for j in range(7):image[0,i*4:(i+1)*4,j*4:(j+1)*4]=i*7+j
        image[1]=255
        expected=np.vstack((np.arange(49)/255,np.ones(49))).astype(np.float32)
        np.testing.assert_array_equal(p.pooled_image_features(image),expected)
        np.testing.assert_array_equal(p.pooled_image_features(image[:,None]),expected)
        np.testing.assert_array_equal(p.pooled_image_features(image.reshape(2,784)),expected)

    def test_quality_conservative_ties_and_dense_prediction_forced_zero(self):
        predicted=np.zeros((2,16));predicted[:,15]=99
        np.testing.assert_array_equal(p.choose_masks(predicted,COST,0),[15,15])
        np.testing.assert_array_equal(p.choose_masks(predicted,COST,.1),[0,0])
        predicted[:]=-1;predicted[:,15]=0
        np.testing.assert_array_equal(p.choose_masks(predicted,np.ones(16),0),[14,14])
        np.testing.assert_array_equal(p.choose_masks(np.zeros((1,16)),np.ones(16),0),[15])

    def test_image_only_apply_signature(self):
        self.assertEqual(list(inspect.signature(p.apply_policy).parameters),['model','features'])
        model={'chosen_lambda':None}
        np.testing.assert_array_equal(p.apply_policy(model,np.zeros((3,49))),[15,15,15])
        with self.assertRaises(TypeError):p.apply_policy(model,np.zeros((3,49)),labels=[1,2,3])

    def test_paired_interval_uses_harm_repair_and_correct_sign(self):
        ids=np.array([0,0,0,0]);pred=np.zeros((16,4),dtype=np.int64);labels=np.array([0,0,0,1])
        pred[0]=[1,0,0,1] # one harm and one repair: net0, nonzero uncertainty
        result=p.evaluate_routing(ids,pred,labels,COST)
        self.assertEqual(result['dense_correct_harmed'],1)
        self.assertEqual(result['dense_wrong_repaired'],1)
        self.assertEqual(result['accuracy_delta_pp'],0)
        self.assertGreater(result['paired_error_interval']['sample_sd'],0)
        low,high=result['paired_error_interval']['accuracy_delta_pp_ci95']
        self.assertLess(low,0);self.assertGreater(high,0)
        self.assertEqual(result['mask_counts'][0],4)
        self.assertEqual(sum(result['mask_shares']),1)
        self.assertIsNone(result['per_class'][9]['accuracy'])
        dense=p.evaluate_routing(np.full(4,15),pred,labels,COST)
        self.assertEqual(dense['paired_error_interval']['accuracy_delta_pp_ci95'],[-0.,-0.])


class FitChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.x,cls.pred,cls.y=routing_fixture()
        cls.model=p.fit_policy(cls.x,cls.pred,cls.y,COST)

    def test_predeclared_fit_hyperparameters_split_and_lambda_selection(self):
        model=self.model
        self.assertEqual(model['protocol']['max_depth'],3)
        self.assertEqual(model['protocol']['min_samples_leaf'],200)
        self.assertEqual(model['protocol']['random_state'],20260909)
        self.assertEqual(model['chosen_lambda'],.001)
        self.assertEqual(model['calibration']['selected']['errors'],0)
        self.assertAlmostEqual(model['calibration']['selected']['mac_saving_fraction'],.42)
        self.assertEqual(model['static_comparator']['selected_mask_id'],15)
        self.assertEqual(len(model['calibration']['candidates']),12)
        self.assertTrue(model['calibration']['candidates'][-1]['feasible'])
        self.assertEqual(model['calibration']['candidates'][-1]['kind'],'dense_fallback')
        self.assertEqual(model['tree']['n_node_samples'][0],5000)

    def test_numpy_export_matches_fitted_behavior_and_json_roundtrip(self):
        restored=json.loads(json.dumps(self.model,allow_nan=False))
        expected=np.where(self.x[:,0]==0,0,15)
        np.testing.assert_array_equal(p.apply_policy(restored,self.x),expected)
        extra=p.predict_extra_error(restored,self.x)
        np.testing.assert_array_equal(extra[:,15],np.zeros(len(self.x)))
        np.testing.assert_array_equal(extra[:,0],self.x[:,0])

    def test_calibration_labels_predictions_do_not_change_fitted_tree(self):
        pred=self.pred.copy();pred[:,5000:]=1;pred[15,5000:]=0
        alternate=p.fit_policy(self.x,pred,self.y,COST)
        self.assertEqual(alternate['tree'],self.model['tree'])
        self.assertEqual(alternate['calibration']['selected']['mac_saving_fraction'],0)
        np.testing.assert_array_equal(p.apply_policy(alternate,self.x),np.full(10000,15))

    def test_calibration_tolerance_boundary_is_ten_errors_of_five_thousand(self):
        x=np.zeros((10000,49),dtype=np.float32);y=np.zeros(10000,dtype=np.int64)
        pred=np.ones((16,10000),dtype=np.int64);pred[15]=0;pred[0]=0
        pred[0,5000:5010]=1
        model=p.fit_policy(x,pred,y,COST)
        self.assertEqual(model['calibration']['selected']['excess_errors'],10)
        self.assertEqual(model['chosen_lambda'],.001)
        pred[0,5010]=1
        failed=p.fit_policy(x,pred,y,COST)
        self.assertEqual(failed['calibration']['selected']['mac_saving_fraction'],0)
        self.assertFalse(next(r for r in failed['calibration']['candidates'] if r['lambda']==.001)['feasible'])

    def test_test_evaluation_does_not_mutate_or_recalibrate_policy(self):
        model=copy.deepcopy(self.model);before=json.dumps(model,sort_keys=True)
        x=np.zeros((20,49));pred=np.zeros((16,20),dtype=np.int64);pred[0]=1;y=np.zeros(20,dtype=np.int64)
        report=p.evaluate_policy(model,x,pred,y)
        self.assertEqual(report['routed']['accuracy'],0)
        self.assertEqual(report['dense']['accuracy'],1)
        self.assertEqual(report['routed']['accuracy_delta_pp'],-100)
        self.assertEqual(json.dumps(model,sort_keys=True),before)
        self.assertEqual(model['chosen_lambda'],.001)

    def test_random_comparator_preserves_exact_test_mask_multiset(self):
        report=p.evaluate_policy(self.model,self.x,self.pred,self.y)
        self.assertEqual(report['routed']['mask_counts'],report['random_cost_matched']['mask_counts'])
        self.assertAlmostEqual(report['routed']['mac_saving_fraction'],report['random_cost_matched']['mac_saving_fraction'])
        self.assertEqual(report['routed']['accuracy'],1)
        self.assertLess(report['random_cost_matched']['accuracy'],1)
        again=p.evaluate_policy(self.model,self.x,self.pred,1-self.y)
        self.assertEqual(report['random_cost_matched']['mask_counts'],again['random_cost_matched']['mask_counts'])
        self.assertEqual(report['random_cost_matched']['seed'],20260912)
        json.dumps(report,allow_nan=False)

    def test_reject_bad_shapes_costs_and_missing_rows(self):
        with self.assertRaises(ValueError):p.fit_policy(self.x[:9999],self.pred[:,:9999],self.y[:9999],COST)
        with self.assertRaises(ValueError):p.apply_policy(self.model,np.zeros((2,50)))
        with self.assertRaises(ValueError):p.apply_policy(self.model,np.ones((2,49))*2)
        with self.assertRaises(ValueError):p.choose_masks(np.zeros((2,16)),np.zeros(16),0)
        with self.assertRaises(ValueError):p.evaluate_routing(np.array([16]),np.zeros((16,1),dtype=int),np.array([0]),COST)


if __name__=='__main__':
    unittest.main()
