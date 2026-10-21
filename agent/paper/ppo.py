"""One likelihood ratio and one reward per physical event, including Periodic-DAN.

Two advantage estimators are registered. ``gae`` is the per-event lambda-return
estimator with a trained critic. ``self_critical`` is the episode-level estimator
against the anchor rule (Periodic-ATC) on the episode's own case, centred on the
replicates of that case: every decision of an episode shares one Monte Carlo
credit (gamma = 1), no bootstrapping and no critic are needed.
"""
import numpy as np
import torch


def returns_and_advantages(records, bootstrap, lam):
    last_value = dict(bootstrap)
    advantage = {key: 0.0 for key in bootstrap}
    for row in reversed(records):
        continuation = 0.0 if row.done else 1.0
        delta = row.reward + continuation * last_value[row.env_id] - row.value
        gae = delta + continuation * lam * advantage[row.env_id]
        row.advantage, row.target = gae, gae + row.value
        last_value[row.env_id], advantage[row.env_id] = row.value, gae


def monte_carlo_targets(records):
    """Undiscounted return-to-go per slot; auxiliary critic targets only."""
    total = {}
    for row in reversed(records):
        total[row.env_id] = row.reward + (0.0 if row.done else total.get(row.env_id, 0.0))
        row.target = total[row.env_id]


def self_critical_advantages(records, episodes, group_center=True):
    """(B - TWT) / B per episode, B = anchor TWT of the episode's own case.

    Replicates of the same case form one group whose mean is subtracted when
    ``group_center`` is set, so the shared baseline is estimated from the batch
    itself and the anchor offset cancels. Returns the per-slot advantages.
    """
    by_slot = {int(r["env_id"]): r for r in episodes}
    raw = {slot: (float(r["baseline_twt"]) - float(r["twt"])) / max(float(r["baseline_twt"]), 1.0)
           for slot, r in by_slot.items()}
    if group_center:
        groups = {}
        for slot, r in by_slot.items():
            groups.setdefault(r["group"], []).append(slot)
        for slots in groups.values():
            centre = float(np.mean([raw[s] for s in slots]))
            for s in slots:
                raw[s] -= centre
    monte_carlo_targets(records)
    for row in records:
        if row.env_id not in raw:
            raise ValueError(f"Transition slot {row.env_id} has no episode row")
        row.advantage = raw[row.env_id]
    return raw


class PPO:
    def __init__(self, network, options, total_updates=None):
        self.network, self.options = network, options
        self.total_updates = int(total_updates or 1)
        self.actor = torch.optim.Adam(network.actor_parameters(), lr=options["actor_lr"])
        self.critic = torch.optim.Adam(network.critic_parameters(), lr=options["critic_lr"])
        self.actor_scheduler = torch.optim.lr_scheduler.LambdaLR(self.actor, self.lr_factor)
        self.critic_scheduler = torch.optim.lr_scheduler.LambdaLR(self.critic, self.lr_factor)
        self.trains_critic = options["advantage_mode"] == "gae" or options["critic"] == "auxiliary"

    def lr_factor(self, step):
        warm = int(self.options.get("lr_warmup_updates", 0))
        warmup = min(1.0, (step + 1) / warm) if warm > 0 else 1.0
        progress = min(1.0, step / max(1, self.total_updates))
        return warmup * (1.0 - (1.0 - float(self.options.get("lr_final_ratio", 1.0))) * progress)

    def kl_coefficient(self, step):
        start, end = float(self.options["kl_prior_coef_start"]), float(self.options["kl_prior_coef_end"])
        span = max(1, int(self.options["kl_anneal_updates"]))
        return start + (end - start) * min(1.0, step / span)

    def state_dict(self):
        return dict(actor_optimizer=self.actor.state_dict(), critic_optimizer=self.critic.state_dict(),
                    actor_scheduler=self.actor_scheduler.state_dict(), critic_scheduler=self.critic_scheduler.state_dict())

    def load_state_dict(self, state):
        self.actor.load_state_dict(state["actor_optimizer"])
        self.critic.load_state_dict(state["critic_optimizer"])
        self.actor_scheduler.load_state_dict(state["actor_scheduler"])
        self.critic_scheduler.load_state_dict(state["critic_scheduler"])

    def update(self, records, bootstrap, *, episodes=None, step=0):
        if len({r.version for r in records}) != 1:
            raise ValueError("Mixed on-policy versions")
        o = self.options
        mode = o["advantage_mode"]
        if mode == "gae":
            returns_and_advantages(records, bootstrap, o["gae_lambda"])
        else:
            if episodes is None:
                raise ValueError("Self-critical advantages require the batch episode rows")
            self_critical_advantages(records, episodes, bool(o["group_center"]))
        train_critic = mode == "gae" or o["critic"] == "auxiliary"
        device = next(self.network.parameters()).device
        targets = torch.tensor([r.target for r in records], dtype=torch.float32, device=device)
        if train_critic:
            self.network.value.update(targets, self.critic)
        choice = torch.tensor([bool(r.choice) for r in records], device=device)
        advantages = torch.tensor([r.advantage for r in records], dtype=torch.float32, device=device)
        if choice.any():
            selected = advantages[choice]
            advantages = (advantages - selected.mean()) / selected.std(unbiased=False).clamp_min(1e-8)
        advantages = torch.where(choice, advantages, torch.zeros_like(advantages))
        entropy_coef = float(o["entropy_dan"] if self.network.method == "dan" else o["entropy"])
        kl_coef = 0.0 if self.network.method == "dan" else self.kl_coefficient(step)
        clip = float(o["clip"])
        lr_used = float(self.actor.param_groups[0]["lr"])
        minibatches, gradients, passes = [], [], 0
        for _ in range(int(o["ppo_update_epochs"])):
            ids = torch.randperm(len(records)).tolist()
            pass_rows = []
            for start in range(0, len(ids), int(o["minibatch"])):
                batch = ids[start:start + int(o["minibatch"])]
                n_choice = int(choice[batch].sum())
                self.actor.zero_grad(set_to_none=True)
                self.critic.zero_grad(set_to_none=True)
                sums = dict(policy_loss=0.0, value_loss=0.0, kl=0.0, entropy=0.0, kl_prior=0.0, clip_fraction=0.0)
                for offset in range(0, len(batch), int(o["microbatch"])):
                    index = batch[offset:offset + int(o["microbatch"])]
                    rows = [records[i] for i in index]
                    logp, entropy, kl_prior, v = self.network.evaluate_actions(
                        [r.observation for r in rows], [r.action for r in rows], need_values=train_critic)
                    old_logp = torch.tensor([r.logp for r in rows], dtype=torch.float32, device=device)
                    logratio = logp - old_logp
                    ratio = logratio.exp()
                    adv = advantages[index]
                    weight = choice[index].float() / max(1, n_choice)
                    surrogate = torch.minimum(ratio * adv, ratio.clamp(1 - clip, 1 + clip) * adv)
                    policy = -(weight * surrogate).sum()
                    regulariser = (weight * (kl_coef * kl_prior - entropy_coef * entropy)).sum()
                    if train_critic:
                        target = self.network.value.normalize(targets[index])
                        if o["value_clip"]:
                            old = self.network.value.normalize(torch.tensor([r.value for r in rows], dtype=torch.float32, device=device))
                            clipped = old + (v - old).clamp(-clip, clip)
                            loss = .5 * torch.maximum((v - target).square(), (clipped - target).square()).mean()
                        else:
                            loss = .5 * (v - target).square().mean()
                    else:
                        loss = torch.zeros((), device=device)
                    if not torch.isfinite(policy + regulariser + loss):
                        raise FloatingPointError("Non-finite PPO objective")
                    if n_choice:
                        (policy + regulariser).backward()
                    if train_critic:
                        (loss * len(index) / len(batch)).backward()
                    with torch.no_grad():
                        sums["policy_loss"] += float(policy)
                        sums["value_loss"] += float(loss) * len(index) / len(batch)
                        sums["kl"] += float((weight * (ratio - 1 - logratio)).sum())
                        sums["entropy"] += float((weight * entropy).sum())
                        sums["kl_prior"] += float((weight * kl_prior).sum())
                        sums["clip_fraction"] += float((weight * ((ratio - 1).abs() > clip).float()).sum())
                ag = torch.nn.utils.clip_grad_norm_(self.network.actor_parameters(), o["max_grad_norm"], error_if_nonfinite=True)
                cg = torch.nn.utils.clip_grad_norm_(self.network.critic_parameters(), o["max_grad_norm"], error_if_nonfinite=True)
                # No Adam momentum update for a batch with no genuine action choice.
                if n_choice:
                    self.actor.step()
                if train_critic:
                    self.critic.step()
                gradients.append([float(ag), float(cg)])
                pass_rows.append(sums)
            minibatches.extend(pass_rows)
            passes += 1
            target_kl = o.get("target_kl")
            if target_kl is not None and pass_rows and float(np.mean([r["kl"] for r in pass_rows])) > float(target_kl):
                break
        self.actor_scheduler.step()
        if self.trains_critic:
            self.critic_scheduler.step()
        keys = ("policy_loss", "value_loss", "kl", "entropy", "kl_prior", "clip_fraction")
        metrics = {k: float(np.mean([r[k] for r in minibatches])) for k in keys}
        metrics.update(actor_grad_norm=float(np.mean(gradients, 0)[0]), critic_grad_norm=float(np.mean(gradients, 0)[1]),
                       actor_lr=lr_used, kl_coefficient=float(kl_coef), passes_completed=passes, advantage_mode=mode)
        return metrics
