"""Regression for the deployed shared-embedding startup path."""
import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from rl.algorithms.accumulating_ppo import AccumulatingMaskablePPO


@pytest.mark.parametrize('recovery_state', [None, Path('resume-generation'), Path('fork-generation')])
@pytest.mark.parametrize('train_shared', [False, True])
def test_startup_preserves_shared_weights_on_recovery(recovery_state, train_shared):
    # Execute the actual startup block without entering SSH/W&B orchestration.
    source = Path(__file__).resolve().parents[2] / 'rl/training/distributed_sb3.py'
    tree = ast.parse(source.read_text())
    blocks = [node for node in ast.walk(tree) if isinstance(node, ast.If)
              and isinstance(node.test, ast.Name)
              and node.test.id == 'source_nosl_inner_only']
    assert len(blocks) == 1
    block = compile(ast.Module(body=blocks, type_ignores=[]), str(source), 'exec')
    shared = torch.nn.ModuleDict({'cards': torch.nn.Embedding(3, 4),
                                'relics': torch.nn.Embedding(2, 4)})
    with torch.no_grad():
        for parameter in shared.parameters():
            parameter.fill_(17.0)
    before = {name: p.detach().clone() for name, p in shared.named_parameters()}
    configured = []
    model = SimpleNamespace(nosl_is_az=True, nosl_is_az_inner_only=True,
                            policy=SimpleNamespace(core=SimpleNamespace(shared=shared)))
    model.reset_nosl_is_az_shared_embeddings = lambda seed: AccumulatingMaskablePPO.reset_nosl_is_az_shared_embeddings(model, seed)
    model.configure_nosl_is_az_inner_only = lambda **kw: (configured.append(kw) or (0, 20))
    exec(block, dict(source_nosl_inner_only=True, recovery_state=recovery_state,
                     args=SimpleNamespace(nosl_is_az_train_shared_embeddings=train_shared,
                                          seed=42, rank=0), model=model))
    assert configured == [{'train_shared_embeddings': train_shared}]
    preserved = all(torch.equal(before[name], p) for name, p in shared.named_parameters())
    assert preserved == (recovery_state is not None or not train_shared)
