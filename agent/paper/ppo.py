"""One likelihood ratio and one reward per physical event, including Periodic-DAN."""
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


class PPO:
    def __init__(self, network, options):
        self.network, self.options = network, options
        self.actor = torch.optim.Adam(network.actor_parameters(), lr=options["actor_lr"])
        self.critic = torch.optim.Adam(network.critic_parameters(), lr=options["critic_lr"])

    def update(self, records, bootstrap):
        if len({r.version for r in records}) != 1:
            raise ValueError("Mixed on-policy versions")
        returns_and_advantages(records, bootstrap, self.options["gae_lambda"])
        targets = torch.tensor([r.target for r in records])
        self.network.value.update(targets, self.critic)
        advantages = torch.tensor([r.advantage for r in records])
        advantages = (advantages - advantages.mean()) / advantages.std(unbiased=False).clamp_min(1e-8)
        metrics, gradients = [], []
        for _ in range(self.options["ppo_update_epochs"]):
            ids = torch.randperm(len(records)).tolist()
            for start in range(0, len(ids), self.options["minibatch"]):
                batch = ids[start:start + self.options["minibatch"]]
                self.actor.zero_grad(set_to_none=True)
                self.critic.zero_grad(set_to_none=True)
                choice = any(records[i].choice for i in batch)
                for offset in range(0, len(batch), self.options["microbatch"]):
                    index = batch[offset:offset + self.options["microbatch"]]
                    rows = [records[i] for i in index]
                    logp, entropy, v = self.network.evaluate_actions([r.observation for r in rows], [r.action for r in rows])
                    old_logp = torch.tensor([r.logp for r in rows])
                    logratio = logp - old_logp
                    ratio = logratio.exp()
                    clip = self.options["clip"]
                    adv = advantages[index]
                    policy = -torch.minimum(ratio * adv, ratio.clamp(1 - clip, 1 + clip) * adv).mean()
                    old = self.network.value.normalize(torch.tensor([r.value for r in rows]))
                    target = self.network.value.normalize(targets[index])
                    clipped = old + (v - old).clamp(-clip, clip)
                    loss = .5 * torch.maximum((v - target).square(), (clipped - target).square()).mean()
                    if not torch.isfinite(policy + loss + entropy.mean()):
                        raise FloatingPointError("Non-finite PPO objective")
                    weight = len(index) / len(batch)
                    if choice:
                        ((policy - self.options["entropy"] * entropy.mean()) * weight).backward()
                    (loss * weight).backward()
                    metrics.append([float(policy.detach()), float(loss.detach()), float((ratio - 1 - logratio).mean().detach()), float(entropy.mean().detach())])
                ag = torch.nn.utils.clip_grad_norm_(self.network.actor_parameters(), self.options["max_grad_norm"], error_if_nonfinite=True)
                cg = torch.nn.utils.clip_grad_norm_(self.network.critic_parameters(), self.options["max_grad_norm"], error_if_nonfinite=True)
                # No Adam momentum update for a batch with no genuine action choice.
                if choice:
                    self.actor.step()
                self.critic.step()
                gradients.append([float(ag), float(cg)])
        return dict(zip(("policy_loss", "value_loss", "kl", "entropy"), np.mean(metrics, 0).tolist()),
                    actor_grad_norm=float(np.mean(gradients, 0)[0]), critic_grad_norm=float(np.mean(gradients, 0)[1]))
