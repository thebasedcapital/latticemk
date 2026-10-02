"""Consumer-visible target, EOS reversibility and replica-exchange checks."""
import json
import math
import random
import unittest
from pathlib import Path

from sampler import Record, accept, local_ratio, swap_ratio, sweep, ppt
from run import answer


class ToyEngine:
    # The paper's Figure 3 binary model at horizon 2. Token 1 is EOS.
    def __init__(self, seed=19):
        self.rng = random.Random(seed)
        self.generated = 0

    def generate(self, prompt, jobs):
        result=[]
        for r,alpha,end in jobs:
            r=r.prefix(len(r.tokens))
            while len(r.tokens)<end and not r.terminal:
                token=int(self.rng.random()<0.5)
                r.tokens.append(token); r.logp.append(math.log(0.5))
                r.zeta.append([(1-a)*math.log(2) for a in prompt])
                r.terminal=token==1
                self.generated+=1
            result.append(r)
        return result


class SamplerTests(unittest.TestCase):
    def test_fixed_horizon_eos_can_lengthen_and_has_power_target(self):
        rng=random.Random(17); engine=ToyEngine(); state=Record([1],[math.log(.5)],[[-math.log(2)]],True)
        counts=[0,0,0]; lengthen=0; accepted=0; proposals=100000
        for i in range(proposals):
            restart=rng.randrange(2)
            if restart>=len(state.tokens):
                new=state
            else:
                new=engine.generate([2.0],[(state.prefix(restart),2.0,2)])[0]
            if accept(local_ratio(state,new,restart,0),rng):
                lengthen+=len(state.tokens)==1 and len(new.tokens)==2
                accepted+=1; state=new
            if i>=1000:
                counts[0 if len(state.tokens)==1 else (1 if state.terminal else 2)]+=1
        frequencies=[n/sum(counts) for n in counts]
        self.assertGreater(lengthen,1000)
        for observed,expected in zip(frequencies,[2/3,1/6,1/6]):
            self.assertAlmostEqual(observed,expected,delta=.01)
        self.diagnostics=dict(target=[2/3,1/6,1/6],observed=frequencies,
                              eos_lengthenings=lengthen,local_acceptance=accepted/proposals)

    def test_swap_direction_and_ordered_multirung_transport(self):
        records=[Record([k],[-float(k)],[]) for k in range(4)]
        stats=dict(swap_attempts=[0]*3,swap_accepts=[0]*3)
        self.assertGreater(swap_ratio(records[0],records[1],2,3),0)
        self.assertLess(swap_ratio(records[1],records[0],2,3),0)
        sweep(records,[2,2.3,2.6,3],random.Random(2),stats)
        self.assertEqual(records[-1].tokens,[0])
        self.assertEqual(stats['swap_accepts'],[1,1,1])
        rng=random.Random(31)
        rate=sum(accept(-1,rng) for _ in range(100000))/100000
        self.assertAlmostEqual(rate,math.exp(-1),delta=.005)
        self.swap_diagnostics=dict(negative_log_ratio=-1,observed_acceptance=rate,
            target_acceptance=math.exp(-1),ordered_records=[r.tokens for r in records],
            ordered_sweep_accepts=stats['swap_accepts'])

    def test_normalizers_follow_swapped_record_not_source_power(self):
        old=Record([0,0],[-.7,-.7],[[-1,-2],[-3,-4]])
        new=Record([0,1],[-.7,-.8],[[-1,-2],[-6,-9]])
        self.assertEqual(local_ratio(old,new,1,1),-5)
        self.assertEqual(local_ratio(old,new,1,0),-3)

    def test_resume_matches_uninterrupted_schedule(self):
        config=dict(powers=[1.,2.],horizon=2,block=1,rounds=4)
        e=ToyEngine(); saved=[]
        expected,stats=ppt(e,config['powers'],123,config,
            checkpoint=lambda s:saved.append((json.loads(json.dumps(s)),e.rng.getstate())))
        state,rngstate=saved[3]
        resumed_engine=ToyEngine(); resumed_engine.rng.setstate(rngstate)
        observed,other=ppt(resumed_engine,config['powers'],123,config,state=state)
        self.assertEqual(expected,observed); self.assertEqual(stats,other)

    def test_answer_exact_match_and_missing_answer(self):
        self.assertEqual(answer('#### 1,200.00'),'1.2E+3')
        self.assertEqual(answer('#### -3.5'),'−3.5'.replace('−','-'))
        self.assertIsNone(answer('worked with 34 apples but gave no final answer'))

    def test_controls_exact_budget_and_truncated_sample_exclusion(self):
        from controls import independent, single_chain
        class Engine:
            generated=0
            def generate(self,prompt,jobs):
                result=[]
                for _,alpha,cap in jobs:
                    n=min(2,cap); self.generated+=n
                    result.append(Record([0]*n,[-float(n)]*n,[],n==2))
                return result
            def text(self,r): return '#### 7'
        engine=Engine(); result=independent(engine,[],5,1.,answer,lambda s:None)
        self.assertEqual(result['spent'],5)
        self.assertEqual(result['truncated'],1)
        self.assertEqual(len(result['candidates']),2)
        self.assertEqual(result['vote'],'7')
        complete=single_chain(Engine(),[],2,answer)
        self.assertEqual(complete['answer'],'7')
        self.assertLessEqual(complete['accepts'],complete['attempts'])
        truncated=single_chain(Engine(),[],1,answer)
        self.assertIsNone(truncated['answer'])
        self.assertEqual(truncated['truncated'],1)

    def test_paired_significance_and_wilson_bounds(self):
        from summarize import paired, wilson
        left={i:dict(correct=True) for i in range(100)}
        right={i:dict(correct=i>=10) for i in range(100)}
        result=paired(left,right)
        self.assertEqual(result['wins'],10)
        self.assertEqual(result['losses'],0)
        self.assertAlmostEqual(result['p'],2/1024)
        self.assertEqual(result['delta'],.1)
        lo,hi=wilson(50,100)
        self.assertLess(lo,.5); self.assertGreater(hi,.5)


if __name__=='__main__':
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(SamplerTests)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    # Exercise the sampler itself, not only arithmetic helpers.
    check=SamplerTests(); check.test_fixed_horizon_eos_can_lengthen_and_has_power_target()
    check.test_swap_direction_and_ordered_multirung_transport()
    payload=dict(passed=result.wasSuccessful(),tests=result.testsRun,swaps=check.swap_diagnostics,**check.diagnostics)
    Path(__file__).with_name('sampler-proof.json').write_text(json.dumps(payload,indent=2)+'\n')
    print(json.dumps(payload)); raise SystemExit(not result.wasSuccessful())
