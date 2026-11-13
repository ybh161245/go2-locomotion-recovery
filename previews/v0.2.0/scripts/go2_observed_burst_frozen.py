"""131 verifier for the already frozen130 candidate; no input mutations."""
from archive_project import ROOT
from diagnose_go2_update import read,sha
from run_go2_supply_new_conditions import verify_supply_frozen

FROZEN130='docs/data/go2-observed-entry-burst/frozen-candidate.json'
FROZEN130_SHA='b5e711081163446082720bdf20775f878a2758752727aec5aff98402fe4eb5fe'


def verify_frozen130():
    verify_supply_frozen()
    assert sha(ROOT/FROZEN130)==FROZEN130_SHA
    f=read(ROOT/FROZEN130)
    assert sha(ROOT/f['config'])==f['config_sha256']
    for key in ['models','runtime_code_sha256','native_assets_sha256','bank_sha256']:
        for name,digest in f[key].items():assert sha(ROOT/name)==digest,name
    for key in ['summary','audit']:assert sha(ROOT/f[key])==f[key+'_sha256'],key
    assert sha(ROOT/f['base123_frozen_manifest'])==f['base123_sha256']
    assert not f['new_condition_qualified'] and not f['formal_selection_changed']
    assert read(ROOT/f['audit'])['evidence_valid']
    return f
