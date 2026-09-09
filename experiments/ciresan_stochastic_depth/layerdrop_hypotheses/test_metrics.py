"""Numerical counterexamples and exact combinatorial checks; CPU only."""
import json
import math
import unittest

import numpy as np

try:
    from . import metrics as m
except ImportError:
    import metrics as m


BITS = np.asarray([[bool(mask & (1 << i)) for i in range(4)] for mask in range(16)], dtype=float)


class TransformChecks(unittest.TestCase):
    def test_walsh_roundtrip_parseval_each_pure_degree(self):
        for term in range(16):
            character=np.prod(2*BITS[:,[i for i in range(4) if term & (1 << i)]]-1,axis=1)
            values=character[:,None]*np.array([1.,2.,-4.])[None,:]
            coefficients=m.walsh_transform(values)
            target=np.zeros_like(values);target[term]=[1.,2.,-4.]
            np.testing.assert_allclose(coefficients,target,atol=0,rtol=0)
            np.testing.assert_allclose(m.inverse_walsh_transform(coefficients),values,atol=0,rtol=0)
            spectrum=m.walsh_spectrum(values)
            self.assertAlmostEqual(spectrum['degree_energy'][term.bit_count()],7.)
            self.assertAlmostEqual(spectrum['total_energy'],7.)
            self.assertEqual(sum(x!=0 for x in spectrum['degree_energy']),1)

    def test_mobius_pair_is_not_pure_walsh_degree_two(self):
        values=BITS[:,0]*BITS[:,1]
        coefficient=m.mobius_transform(values)
        expected=np.zeros(16);expected[3]=1
        np.testing.assert_array_equal(coefficient,expected)
        np.testing.assert_array_equal(m.inverse_mobius_transform(coefficient),values)
        np.testing.assert_allclose(m.walsh_spectrum(values)['degree_energy'],[1/16,1/8,1/16,0,0])

    def test_pair_difference_changes_with_higher_order_context(self):
        values=BITS[:,0]*BITS[:,1]*BITS[:,2]
        result=m.interaction_summary(values)
        pair=next(p for p in result['pair_mixed_differences'] if p['branches']==[0,1])
        self.assertEqual(pair['empty_context_rms'],0)
        self.assertEqual(pair['full_context_rms'],1)
        self.assertEqual(pair['context_change_rms'],1)
        self.assertEqual(result['term_coefficient_rms_by_mask_id'][7],1)

    def test_additive_centered_logits_and_common_shift_invariance(self):
        score=(-2+BITS@np.array([.5,1.,2.,4.]))[:,None]*np.array([1.,2.])[None,:]
        logits=np.stack((score,-score,0*score),axis=-1)
        delta=m.centered_dense_deltas(logits)
        np.testing.assert_array_equal(delta[15],np.zeros((2,3)))
        self.assertEqual(sum(m.walsh_spectrum(delta)['degree_energy'][2:]),0)
        self.assertEqual(sum(m.interaction_summary(delta)['term_coefficient_rms_by_mask_id'][i] for i in range(16) if i.bit_count()>=2),0)
        shift=np.arange(32).reshape(16,2,1)*.25
        np.testing.assert_allclose(m.centered_dense_deltas(logits+shift),delta,rtol=0,atol=1e-13)
        centered_energy=m.walsh_spectrum(logits-logits.mean(axis=-1,keepdims=True))['degree_energy']
        np.testing.assert_allclose(centered_energy[1:],m.walsh_spectrum(delta)['degree_energy'][1:])

    def test_common_logit_interaction_is_removed_by_class_centering(self):
        interaction=(100*BITS[:,0]*BITS[:,1]*BITS[:,2])[:,None,None]
        logits=np.broadcast_to(np.array([1.,-1.]),(16,2,2)).copy()+interaction
        np.testing.assert_array_equal(m.centered_dense_deltas(logits),np.zeros_like(logits))
        spectrum=m.walsh_spectrum(m.centered_dense_deltas(logits))
        self.assertIsNone(spectrum['degree_ge2_fraction_of_nonconstant_energy'])
        self.assertEqual(spectrum['degree_energy_fraction'],[None]*5)

    def test_fixed_cardinality_monomial_and_walsh_laws_differ(self):
        retention=m.uniform_cardinality_retention()
        characters=m.uniform_cardinality_walsh_expectation()
        self.assertAlmostEqual(retention[2,2],1/6)
        self.assertNotEqual(retention[2,2],(.5)**2)
        self.assertAlmostEqual(characters[2,2],-1/3)
        for term in range(16):
            bit_ids=[i for i in range(4) if term & (1 << i)]
            monomial=np.prod(BITS[:,bit_ids],axis=1)
            walsh=np.prod(2*BITS[:,bit_ids]-1,axis=1)
            for k in range(5):
                ids=[i for i in range(16) if i.bit_count()==k]
                self.assertAlmostEqual(float(monomial[ids].mean()),retention[k,term.bit_count()])
                self.assertAlmostEqual(float(walsh[ids].mean()),characters[k,term.bit_count()])

    def test_random_roundtrip_both_cardinality_reconstructions(self):
        rng=np.random.default_rng(1)
        values=rng.normal(size=(16,7,3))
        a=m.mobius_transform(values);w=m.walsh_transform(values)
        np.testing.assert_allclose(m.inverse_mobius_transform(a),values,atol=1e-14)
        np.testing.assert_allclose(m.inverse_walsh_transform(w),values,atol=1e-14)
        sums=lambda coefficients:np.asarray([sum(coefficients[t] for t in range(16) if t.bit_count()==d) for d in range(5)])
        np.testing.assert_allclose(np.tensordot(m.uniform_cardinality_retention(),sums(a),axes=(1,0)),m.cardinality_means(values),atol=1e-14)
        np.testing.assert_allclose(np.tensordot(m.uniform_cardinality_walsh_expectation(),sums(w),axes=(1,0)),m.cardinality_means(values),atol=1e-14)


class ClassificationChecks(unittest.TestCase):
    def test_additive_logits_create_nonlinear_correctness_and_ce_interaction(self):
        # The predicted class0 implements OR on two bits despite additive logits.
        score=-.5+BITS[:,0]+BITS[:,1]
        logits=np.stack((score,-score),axis=-1)[:,None,:]
        result=m.analyze_masks(logits,np.array([0]))
        self.assertEqual(sum(result['centered_logit_delta_walsh']['degree_energy'][2:]),0)
        self.assertGreater(result['correctness_delta_walsh']['degree_energy'][2],0)
        self.assertGreater(result['ce_delta_walsh']['degree_energy'][2],0)
        correct=(logits.argmax(axis=-1)==0).astype(float)
        self.assertEqual(m.mobius_transform(correct)[3,0],-1)

    def test_conditional_robustness_separates_harm_and_repair(self):
        logits=np.zeros((16,3,2));logits[...,0]=1
        labels=np.array([0,0,1]) # dense correct for examples0/1 only
        logits[0,0]=[0,1]       # harm one formerly correct example
        logits[0,2]=[0,1]       # repair the dense mistake
        result=m.classification_robustness(logits,labels)
        self.assertEqual(result['dense_correct_examples'],2)
        row=result['by_mask'][0]
        self.assertAlmostEqual(row['accuracy'],2/3)
        self.assertEqual(row['accuracy_given_dense_correct'],.5)
        self.assertAlmostEqual(row['harm_probability'],1/3)
        self.assertAlmostEqual(row['repair_probability'],1/3)
        self.assertEqual(result['by_mask'][15]['accuracy_given_dense_correct'],1)
        empty=m.classification_robustness(np.zeros((16,2,2)),np.array([1,1]))
        self.assertIsNone(empty['by_mask'][0]['accuracy_given_dense_correct'])
        self.assertIsNone(empty['spearman_dense_margin_vs_retention_given_dense_correct']['rho'])

    def test_uniform_cardinality_is_not_uniform_over_all_masks(self):
        logits=np.zeros((16,1,2));logits[...,0]=1;logits[3,0]=[0,1]
        result=m.classification_robustness(logits,np.array([0]))
        self.assertAlmostEqual(result['by_retained_cardinality'][2]['accuracy'],5/6)
        self.assertAlmostEqual(result['per_example']['fraction_correct_over14_nonempty_pruned_masks'][0],13/14)
        self.assertEqual([r['mask_count'] for r in result['by_retained_cardinality']],[1,4,6,4,1])

    def test_risk_coverage_ranking_label_free_with_stable_ties(self):
        logits=np.zeros((16,4,2));logits[...,0]=[1,4,2,4]
        logits[0,1]=[-4,0]
        a=m.risk_coverage(logits,np.array([0,0,0,0]),[.5,1])
        b=m.risk_coverage(logits,np.array([1,1,1,1]),[.5,1])
        self.assertEqual(a['dense_ranking_example_indices'],[1,3,2,0])
        self.assertEqual(a['dense_ranking_example_indices'],b['dense_ranking_example_indices'])
        self.assertEqual(a['rows'][0]['selected_count'],2)
        self.assertEqual(a['rows'][0]['by_retained_cardinality'][0]['risk'],.5)
        self.assertEqual(a['rows'][0]['by_retained_cardinality'][4]['risk'],0)
        self.assertEqual(m.risk_coverage(logits,np.array([0,0,0,0]),[.26])['rows'][0]['selected_count'],2)

    def test_spearman_ties_constants_and_small_n(self):
        self.assertAlmostEqual(m.spearman_stats([1,1,3],[4,4,2])['rho'],-1)
        self.assertAlmostEqual(m.spearman_stats([1,2,3],[5,7,9])['rho'],1)
        self.assertIsNone(m.spearman_stats([1,1,1],[1,2,3])['rho'])
        self.assertIsNone(m.spearman_stats([],[])['rho'])
        with self.assertRaises(ValueError):m.spearman_stats([1,float('nan')],[1,2])

    def test_mask_costs_preserve_mandatory_stem_head(self):
        costs=m.mask_costs([5e6,3e6,1.5e6,.5e6],1.965e6)
        self.assertEqual(costs[0]['cost'],1965000)
        self.assertEqual(costs[15]['cost'],11965000)
        self.assertEqual(costs[1]['cost'],6965000)
        self.assertEqual(costs[15]['fraction_of_full_cost'],1)
        self.assertGreater(costs[0]['fraction_of_full_cost'],0)

    def test_reorder_validation_json_safety_and_shifted_ce(self):
        rng=np.random.default_rng(17);logits=rng.normal(size=(16,8,3));labels=np.arange(8)%3
        permutation=rng.permutation(16)
        report=m.analyze_masks(logits[permutation],labels,mask_ids=permutation)
        np.testing.assert_array_equal(m.canonical_logits(logits[permutation],permutation),logits)
        json.dumps(report,allow_nan=False)
        self.assertTrue(report['uniform_cardinality']['both_basis_reconstructions_verified'])
        shift=rng.normal(size=(16,8,1))*100
        np.testing.assert_allclose(m.cross_entropy_by_example(logits+shift,labels),m.cross_entropy_by_example(logits,labels),rtol=1e-12,atol=1e-12)
        with self.assertRaises(ValueError):m.analyze_masks(logits,labels,mask_ids=[0]*16)
        with self.assertRaises(ValueError):m.analyze_masks(logits,np.arange(8,dtype=float))
        with self.assertRaises(ValueError):m.mask_costs([1,2,3,4],-1)
        with self.assertRaises(ValueError):m.risk_coverage(logits,labels,[0])
        bad=logits.copy();bad[1,2,0]=math.nan
        with self.assertRaises(ValueError):m.analyze_masks(bad,labels)


if __name__ == '__main__':
    unittest.main()
