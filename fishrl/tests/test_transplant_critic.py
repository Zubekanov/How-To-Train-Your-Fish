import copy
import torch

from fishrl.tests.test_critic_epochs import _cfg
from fishrl.train import checkpoint as ckpt
from fishrl.train.train_loop import build_models
from fishrl.train.transplant_critic import transplant


def _payload(m, done):
    opt = torch.optim.Adam(list(m.actor.parameters()) + list(m.critic.parameters()), lr=1e-3)
    x = torch.randn(4, m.actor.in_dim); feat = torch.randn(4, m.critic.enc.globals_dim + m.critic.enc.R * 44)
    loss = m.actor(x).sum() + m.critic(feat).sum(); loss.backward(); opt.step()
    from fishrl.train.train_loop import _encoders
    cfg = _cfg()
    return {"config": {"seed": 0, "encoders": _encoders(cfg), "use_belief": cfg.use_belief,
                       "critic_hidden": list(cfg.critic_hidden), "hidden": list(cfg.hidden),
                       "actor_hidden": list(cfg.actor_hidden), "card_dim": cfg.card_dim,
                       "belief_mode": cfg.belief_mode, "critic_view": cfg.critic_view,
                       "critic_deckout_aux": cfg.critic_deckout_aux, "text_change_mode": cfg.text_change_mode,
                       "obs_counts": cfg.obs_counts},
            "models": {"actor": m.actor.state_dict(), "critic": m.critic.state_dict()},
            "optim": {"ppo": opt.state_dict()}, "done": done, "frozen": {"actor": m.actor.state_dict()},
            "handoff_start": 0}


def test_transplant_swaps_only_the_critic(tmp_path):
    torch.manual_seed(0); ms = build_models(_cfg()); ps = _payload(ms, 10)
    torch.manual_seed(1); md = build_models(_cfg()); pd = _payload(md, 20)
    src, dst = str(tmp_path / "src.pt"), str(tmp_path / "dst.pt")
    ckpt.save_checkpoint(src, ps); ckpt.save_checkpoint(dst, pd)
    out = transplant(src, dst)
    for k, v in out["models"]["critic"].items():
        assert torch.equal(v, ps["models"]["critic"][k])
    for k, v in out["models"]["actor"].items():
        assert torch.equal(v, pd["models"]["actor"][k])
    assert out["done"] == 20 and out["handoff_start"] == 0
    n_actor = len(list(md.actor.parameters()))
    st = out["optim"]["ppo"]["state"]
    assert torch.equal(st[n_actor]["exp_avg"], ps["optim"]["ppo"]["state"][n_actor]["exp_avg"])
    assert torch.equal(st[0]["exp_avg"], pd["optim"]["ppo"]["state"][0]["exp_avg"])
    assert out["critic_transplant"]["src_it"] == 10
