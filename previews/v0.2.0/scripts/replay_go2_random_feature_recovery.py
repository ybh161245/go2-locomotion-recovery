"""141 frozen, input-only fitted actor, explicit fresh replay or zero-step loading."""
import argparse,json
import numpy as np
import torch
from stable_baselines3 import PPO
from archive_project import ROOT
from diagnose_go2_update import read,sha,dump
from go2_observed_burst_frozen import verify_frozen130
from go2_random_feature_recovery import RandomFeatureActor
from go2_action_transfer import norm_load
from go2_bilateral_pilot import Stack
from run_go2_random_feature_recovery import CONFIG,make_factory
from validate_go2_context_states import run_transient
from evaluate_go2_release_regression import raw_equal
import evaluate_go2_robust as ev

def verify_frozen141():
    f=read(ROOT/'docs/data/go2-random-feature-recovery/frozen-candidate.json');verify_frozen130()
    for key in ['models','runtime_code_sha256']:
        for n,h in f[key].items():assert sha(ROOT/n)==h,n
    for key in ['config','summary','audit','fit_audit','router_audit']:assert sha(ROOT/f[key])==f[key+'_sha256'],key
    assert f['development_qualified'] and not f['new_conditions_opened_at_freeze'] and not f['formal_selection_changed'];return f

def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['c','normal'],default='c');p.add_argument('--load-only',action='store_true');p.add_argument('--output');a=p.parse_args()
    verify_frozen141();cfg=read(ROOT/CONFIG);ev.CONFIG=cfg;spec=cfg['random_feature_recovery'];torch.set_num_threads(1);stack=Stack(cfg)
    try:
        student=PPO.load(ROOT/cfg['observed_entry_burst']['student_model'],device='cpu');actor=RandomFeatureActor(ROOT/spec['parent_model'],ROOT/cfg['run']/'fitted-actor.npz')
        norms=[stack.D[1],stack.formal[1],stack.default[1],stack.low[1],stack.adapter[1],norm_load(ROOT/spec['parent_norm']),norm_load(ROOT/cfg['observed_entry_burst']['student_norm'])]
        assert all(not n.training and not n.norm_reward for n in norms)
        if a.load_only:
            print(json.dumps(dict(mode=a.mode,frozen_PPO_models_loaded=7,fitted_random_feature_heads=1,bank_entries=len(stack.bank),all_normalization_frozen=True,physics_control_steps=0,optimizer_updates=0)));return
        assert a.output,'Fresh --output required;creates one extra physical run outside141.'
        out=(ROOT/a.output).resolve();assert out.is_relative_to(ROOT.resolve()) and not out.exists();out.mkdir(parents=True)
        source=spec['core_sources'][0] if a.mode=='c' else spec['protection_sources'][3]
        plain,make=make_factory(cfg,stack,student,actor,source)
        r=ev.normal_run(out,'replay',source['seed'],make,stack.adapter) if a.mode=='normal' else run_transient(out,'replay',source['protocol'],source['seed'],make,stack.adapter,False)
        ref=next(r for r in read(ROOT/cfg['run']/'summary.json')['records'] if r['source']['folder']==source['folder']);assert raw_equal(r,ref)
        dump(out/'comparison.json',dict(all8_arrays_exact=True,new_physical_runs=1,result=r));print('REPLAY141_EXACT')
    finally:stack.close()

if __name__=='__main__':main()
