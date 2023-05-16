from collections import OrderedDict

import torch
from torch import nn as nn
import torch.optim as optim

from rlkit.torch.networks.cnn import ClipWrapper
from rlkit.torch.policies import MakeDeterministic
from rlkit.torch.alg.torch_rl_algorithm import TorchTrainer
from rlkit.util.misc_functions import tile_embs_by_batch_size
import rlkit.util.pytorch_util as ptu


class BCTrainer(TorchTrainer):
    def __init__(
            self,
            env,
            policy,

            policy_lr=1e-3,
            policy_weight_decay=0,
            optimizer_class=optim.Adam,

            bc_batch_size=64,
            video_batch_size=None,

            bc_weight=1.0,
            task_encoder_weight=0.0,
            aux_task_weight=0.0,

            multitask=False,

            meta_batch_size=4,
            train_tasks=[],

            task_encoder=None,  # For example, BCZVideoEncoder
            task_embedding_type=None,

            finetune_lang_enc=False,
            lang_encoder=None,  # For example, DistilBERT
            target_emb_net=None,  # For example, Mlp

            # obs_key_to_dim_map={},
            emb_obs_keys=[],
            # observation_keys=[],

            task_emb_input_mode="concat_to_img_embs",
            policy_num_film_inputs=0,
            policy_film_input_order="",
            policy_loss_type="logp",
            gripper_loss_weight=None,
            gripper_loss_type=None,
            **kwargs
    ):
        super().__init__()
        self.env = env
        self.policy = policy

        self.optimizers = {}

        # torch.autograd.set_detect_anomaly(True)

        policy_params = list(self.policy.parameters())

        if task_encoder is not None:
            policy_params.extend(list(task_encoder.encoder.parameters()))
        if lang_encoder is not None:
            policy_params.extend(list(lang_encoder.parameters()))
        if target_emb_net is not None:
            policy_params.extend(list(target_emb_net.parameters()))

        self.policy_optimizer = optimizer_class(
            policy_params,
            weight_decay=policy_weight_decay,
            lr=policy_lr,
        )

        self.eval_statistics = OrderedDict()
        self._n_train_steps_total = 0
        self._need_to_update_eval_statistics = True

        self.bc_batch_size = bc_batch_size
        self.bc_weight = bc_weight
        self.eval_policy = MakeDeterministic(self.policy)

        self.multitask = multitask

        self.meta_batch_size = meta_batch_size
        self.train_tasks = train_tasks

        self.task_encoder = task_encoder
        self.task_embedding_type = task_embedding_type
        self.task_encoder_weight = task_encoder_weight
        self.video_batch_size = video_batch_size  # num cached embs to retrieve
        self.aux_task_weight = aux_task_weight

        self.finetune_lang_enc = finetune_lang_enc
        self.lang_encoder = lang_encoder
        self.target_emb_net = target_emb_net

        # self.obs_key_to_dim_map = obs_key_to_dim_map
        self.emb_obs_keys = emb_obs_keys
        self.obs_key_to_obs_idx_pairs = {}
        # self.observation_keys = observation_keys

        self.task_emb_input_mode = task_emb_input_mode
        self.use_film = bool(task_emb_input_mode in [
            "film", "film_video_concat_lang", "film_lang_concat_video"])
        self.policy_num_film_inputs = policy_num_film_inputs
        self.policy_film_input_order = policy_film_input_order

        assert policy_loss_type in ["logp", "mse"]
        self.policy_loss_type = policy_loss_type
        if self.policy.gripper_policy_arch == "sep_head":
            self.gripper_loss_weight = gripper_loss_weight
            self.gripper_loss_type = gripper_loss_type

            if self.gripper_loss_type == "ce":
                self.gripper_loss_fn = nn.CrossEntropyLoss()
            elif self.gripper_loss_type == "mse":
                self.gripper_loss_fn = nn.MSELoss()
            else:
                raise NotImplementedError

    def maybe_transform_target_embs(self, traj_emb_targets):
        if self.target_emb_net is not None:
            # traj_emb_targets.shape: (meta_bs, 768)
            traj_emb_targets = self.target_emb_net(traj_emb_targets)
        return traj_emb_targets

    def run_bc_batch(self, batch):
        t, b, _ = batch['observations'].size()
        o = batch["observations"].view(t * b, -1)
        u = batch["actions"].view(t * b, -1)
        if self.policy.gripper_policy_arch == "sep_head":
            u_gripper = batch["gripper_actions"].view(t * b, -1)
            u_gripper = u_gripper.squeeze()  # Make 1-D for targets
        else:
            u_gripper = None

        losses = dict()

        if self.task_encoder is not None:
            if self.task_encoder.use_cached_embs:
                traj_emb_preds_dict = (
                    self.task_encoder.get_cached_embs_from_task_ids(
                        batch['task_indices'], self.bc_batch_size))
                traj_emb_targets = batch['target_traj_embs']
                traj_emb_targets = self.maybe_transform_target_embs(
                    traj_emb_targets)
                traj_emb_targets_tiled = tile_embs_by_batch_size(
                    traj_emb_targets, self.bc_batch_size)
                loss_kwargs = {}
            else:
                task_encoder_batch_dict = batch['task_encoder_batch_dict']
                # Process task_encoder_batch_dict
                for key, val in task_encoder_batch_dict.items():
                    # task_encoder_batch_dict[key].shape:
                    # (meta_bs, bs, 3, im_size * num_rows, im_size * num_cols)
                    task_encoder_batch_dict[key] = torch.cat(
                        [val[i] for i in range(val.shape[0])], dim=0)
                    # task_encoder_batch_dict[key].shape:
                    # (meta_bs * bs, 3, im_size * num_rows, im_size * num_cols)

                # Process traj_emb_targets (language)
                if (isinstance(self.task_encoder.encoder, ClipWrapper)
                        and not self.task_encoder.use_cached_embs):
                    task_lang_tokens = (
                        self.task_encoder.encoder.get_task_lang_tokens_matrix(
                            batch['task_indices']))
                    task_encoder_batch_dict["clip_lang"] = task_lang_tokens
                else:
                    traj_emb_targets = batch['target_traj_embs']

                    if self.finetune_lang_enc:
                        # traj_emb_targets contains the tokens, not the embs.
                        # We want to compute the embs
                        assert not self.lang_encoder.freeze
                        traj_emb_targets = self.lang_encoder(
                            traj_emb_targets.long())

                    traj_emb_targets = self.maybe_transform_target_embs(
                        traj_emb_targets)
                    if self.task_embedding_type in ["demo_lang", "mcil"]:
                        # traj_emb_targets: (mbs, latent_out_dim)
                        # --> (mbs * bs, latent_out_dim)
                        traj_emb_targets_tiled = tile_embs_by_batch_size(
                            traj_emb_targets, self.bc_batch_size)
                        # Check the order of the emb_obs_keys in
                        # init_emb_keys_and_obs_space(...)
                        if self.task_encoder.num_film_inputs > 0:
                            task_encoder_batch_dict["lang"] = (
                                traj_emb_targets_tiled)

                traj_emb_preds_dict = self.task_encoder(
                    task_encoder_batch_dict)

                loss_kwargs = {}
                if isinstance(self.task_encoder.encoder, ClipWrapper):
                    loss_kwargs['logit_scale_list'] = [
                        traj_emb_preds_dict.pop("logit_scale")]
                    traj_emb_targets = traj_emb_preds_dict.pop("lang")
                    traj_emb_targets = self.maybe_transform_target_embs(
                        traj_emb_targets)
                    traj_emb_targets_tiled = tile_embs_by_batch_size(
                        traj_emb_targets, self.bc_batch_size)

            # Order the embs as according to self.emb_obs_keys
            if len(self.emb_obs_keys) > 1:
                traj_emb_preds_list = []
                for emb_obs_key in self.emb_obs_keys:
                    if "video" in emb_obs_key:
                        vid_enc_mod_key = emb_obs_key.replace("_embedding", "")
                        traj_emb_preds_list.append(
                            traj_emb_preds_dict[vid_enc_mod_key])
            elif len(self.emb_obs_keys) == 1:
                if (self.task_embedding_type == "demo"
                        and "lang" in traj_emb_preds_dict):
                    traj_emb_preds_dict.pop("lang")
                assert len(traj_emb_preds_dict) == 1
                traj_emb_preds_list = list(traj_emb_preds_dict.values())
            else:
                raise NotImplementedError

            if self.task_embedding_type != "mcil":
                # t = time.time()
                losses['task_encoder_loss'] = (
                    self.task_encoder.loss_criterion.calc(
                        traj_emb_preds_list, traj_emb_targets, **loss_kwargs))
                # print("task enc loss computation", time.time() - t)

            if self.task_embedding_type == 'demo':
                emb_or_emb_list = traj_emb_preds_list
            elif self.task_embedding_type == 'demo_lang':
                if (self.task_encoder.num_film_inputs == 0 or
                        self.task_emb_input_mode in [
                            "film_video_concat_lang",
                            "film_lang_concat_video"]):
                    if self.policy_film_input_order == "lv":
                        emb_or_emb_list = (
                            [traj_emb_targets_tiled] + traj_emb_preds_list)
                    elif self.policy_film_input_order in ["vl", ""]:
                        emb_or_emb_list = (
                            traj_emb_preds_list + [traj_emb_targets_tiled])
                elif self.task_encoder.num_film_inputs > 0:
                    # Language (targets) was already added to video encoder
                    # via film.
                    emb_or_emb_list = traj_emb_preds_list
            elif self.task_embedding_type == 'mcil':
                emb_or_emb_list = None
        elif self.task_embedding_type in ["lang", "onehot"]:
            emb_or_emb_list = None
        else:
            print("self.task_embedding_type", self.task_embedding_type)
            raise NotImplementedError

        if self.task_embedding_type == "mcil":
            assert len(traj_emb_preds_list) == 1
            self.modality_to_emb_map = {
                "lang": [traj_emb_targets_tiled],
                "video": traj_emb_preds_list,
            }

            # These loss terms average over all modalities
            losses['logp_loss'] = 0.0
            losses['mse_loss'] = 0.0

            if self.policy.gripper_policy_arch == "sep_head":
                losses['gripper_loss'] = 0.0

            for modality, film_inputs in self.modality_to_emb_map.items():
                policy_losses, stats = self.compute_losses(
                    o, film_inputs, u, u_gripper)

                # add a prefix to keys in policy_losses when adding
                # to losses dict.
                for key, val in policy_losses.items():
                    new_key = f"{modality}_{key}"
                    losses[new_key] = val

                losses['logp_loss'] += policy_losses['logp_loss'] * (
                    1 / len(self.modality_to_emb_map))
                losses['mse_loss'] += policy_losses['mse_loss'] * (
                    1 / len(self.modality_to_emb_map))

                if self.policy.gripper_policy_arch == "sep_head":
                    losses['gripper_loss'] += policy_losses['gripper_loss'] * (
                        1 / len(self.modality_to_emb_map))
        else:
            o, emb = self.process_obs_and_emb(o, emb_or_emb_list)
            # print("train:", torch.norm(o[:,-768:], dim=1)[:10])
            film_inputs = emb if self.use_film else None
            policy_losses, stats = self.compute_losses(
                o, film_inputs, u, u_gripper)
            losses.update(policy_losses)

        if self.policy_loss_type == "logp":
            losses['policy_loss'] = losses['logp_loss']
        elif self.policy_loss_type == "mse":
            losses['policy_loss'] = losses['mse_loss']

        if self.policy.gripper_policy_arch == "sep_head":
            losses['policy_loss'] += (
                self.gripper_loss_weight * losses['gripper_loss'])

        return losses, stats

    def compute_losses(self, o, film_inputs, u, u_gripper=None):
        policy_losses = {}

        if self.use_film:
            dist, policy_stats_dict, aux_outputs = self.policy(
                o, film_inputs=film_inputs)
        else:
            dist, policy_stats_dict, aux_outputs = self.policy(o)

        pred_u, _ = dist.rsample_and_logprob()
        stats = dist.get_diagnostics()
        stats.update(policy_stats_dict)

        # get gripper output from aux_outputs
        if self.policy.gripper_policy_arch == "sep_head":
            gripper_class_preds = torch.round(
                1 + aux_outputs["preds"]["gripper_actions"]).squeeze()
            if self.gripper_loss_type == "ce":
                gripper_preds_for_loss = aux_outputs["preds"]["gripper_logits"]
                u_gripper = u_gripper.long()
            elif self.gripper_loss_type == "mse":
                gripper_preds_for_loss = (
                    1.0 + aux_outputs["preds"]["gripper_actions"]).squeeze()
            else:
                raise NotImplementedError
            policy_losses["gripper_loss"] = self.gripper_loss_fn(
                gripper_preds_for_loss, u_gripper)
            stats["gripper accuracy"] = float(ptu.get_numpy(
                torch.mean((u_gripper == gripper_class_preds).float())))
            # Proportion of predicted actions that are close or open gripper.
            stats["gripper close prop"] = float(ptu.get_numpy(
                torch.mean((gripper_class_preds == 0).float())))
            stats["gripper open prop"] = float(ptu.get_numpy(
                torch.mean((gripper_class_preds == 2).float())))

        if len(self.policy.aux_tasks) > 0 and isinstance(aux_outputs, dict):
            policy_losses.update(aux_outputs['losses'])

        mse_loss = nn.MSELoss()(pred_u, u)

        policy_losses.update(dict(
            logp_loss=-dist.log_prob(u, ).mean(),
            mse_loss=mse_loss,
        ))

        return policy_losses, stats

    def split_emb_from_obs(self, obs):
        assert len(self.emb_obs_keys) == 1
        assert self.task_embedding_type in ["lang", "onehot"]

        emb_start_idx, emb_end_idx = self.policy.obs_key_to_obs_idx_pairs[
            self.emb_obs_keys[0]]
        emb = obs[:, emb_start_idx:emb_end_idx]
        if emb_end_idx == obs.shape[-1]:
            obs_without_emb = obs[:, :emb_start_idx]
        else:
            obs_without_emb = torch.cat(
                [obs[:, :emb_start_idx], obs[:, emb_end_idx:]], dim=-1)
        return obs_without_emb, emb

    def process_obs_and_emb(self, o, emb_or_emb_list):
        if self.task_embedding_type in ["lang", "onehot"]:
            if self.use_film:
                o, emb = self.split_emb_from_obs(o)
                if self.finetune_lang_enc:
                    # emb contains the tokens, not the actual embs.
                    # We want to compute the embs
                    assert not self.lang_encoder.freeze
                    emb = self.lang_encoder(emb.long())
            else:
                emb = None
        elif self.task_embedding_type == 'demo':
            # assert torch.is_tensor(emb_or_emb_list)
            emb = emb_or_emb_list
        elif self.task_embedding_type == 'demo_lang':
            assert isinstance(emb_or_emb_list, list)
            if self.task_emb_input_mode in [
                    "film_video_concat_lang", "film_lang_concat_video"]:
                assert len(emb_or_emb_list) == 2
                assert self.policy_film_input_order in ["vl", ""]
                video_emb = emb_or_emb_list[0]  # Video
                lang_emb = emb_or_emb_list[1]
                # In accordance with the order specified in the creation of
                # emb_or_emb_list right before this function is called

                if self.task_emb_input_mode == "film_video_concat_lang":
                    emb = video_emb
                elif self.task_emb_input_mode == "film_lang_concat_video":
                    emb = lang_emb
            elif self.policy_num_film_inputs in [2, 3]:
                emb = emb_or_emb_list
            elif self.policy_num_film_inputs in [0, 1]:
                # Might be a slowdown to do two cats instead of one.
                emb = torch.cat(emb_or_emb_list, dim=1)
            else:
                raise NotImplementedError

        if self.use_film:
            if self.policy_num_film_inputs == 1 and torch.is_tensor(emb):
                # Make into a singleton list
                emb = [emb]
            assert isinstance(emb, list)

            if self.task_emb_input_mode == "film_video_concat_lang":
                o = torch.cat([o, lang_emb], dim=1)
            elif self.task_emb_input_mode == "film_lang_concat_video":
                o = torch.cat([o, video_emb], dim=1)
        elif self.task_embedding_type in ["demo", "demo_lang"]:
            # not self.use_film
            if isinstance(emb, list):
                # Singleton lists end up as emb[0] after cat.
                emb = torch.cat(emb, dim=1)
            o = torch.cat([o, emb], dim=1)

        return o, emb

    def train_from_torch(self, batch):
        losses, train_stats = self.run_bc_batch(batch)
        self.eval_statistics.update(train_stats)

        train_policy_loss = losses['policy_loss'] * self.bc_weight

        if (self.task_encoder is not None
                and self.task_embedding_type != "mcil"):
            train_policy_loss += (
                losses['task_encoder_loss'] * self.task_encoder_weight)

        if len(self.policy.aux_tasks) > 0 and self.aux_task_weight != 0.0:
            aux_loss_sum = 0.0
            for aux_task in self.policy.aux_tasks:
                aux_loss_sum += losses[aux_task]
            train_policy_loss += aux_loss_sum * self.aux_task_weight

        self.policy_optimizer.zero_grad()
        train_policy_loss.backward()
        self.policy_optimizer.step()

        if self._need_to_update_eval_statistics:
            self._need_to_update_eval_statistics = False
            stats = {
                "Train Logprob Loss": ptu.get_numpy(losses['logp_loss']),
                "Train MSE": ptu.get_numpy(losses['mse_loss']),
                "train_policy_loss": ptu.get_numpy(train_policy_loss),
            }

            if self.policy.gripper_policy_arch == "sep_head":
                stats['Train Gripper Loss'] = ptu.get_numpy(
                    losses['gripper_loss'])

            if self.task_embedding_type == "mcil":
                loss_category_to_str_map = {
                    "logp_loss": "Logprob Loss",
                    "mse_loss": "MSE",
                }
                for modality in self.modality_to_emb_map:
                    for loss_category in ["logp_loss", "mse_loss"]:
                        key = (
                            f"Train {modality} "
                            f"{loss_category_to_str_map[loss_category]}")
                        stats[key] = ptu.get_numpy(
                            losses[f"{modality}_{loss_category}"])

            if (self.task_encoder is not None
                    and self.task_embedding_type != "mcil"):
                stats['Train Task encoder loss'] = ptu.get_numpy(
                    losses['task_encoder_loss'])

            if (len(self.policy.aux_tasks) > 0
                    and self.policy.aux_to_feed_fc in ["none", "preds"]):
                for aux_task in self.policy.aux_tasks:
                    stats[f"Train {aux_task} loss"] = ptu.get_numpy(
                        losses[aux_task])

            self.eval_statistics.update(stats)

        self._n_train_steps_total += 1

    def get_diagnostics(self):
        stats = super().get_diagnostics()
        stats.update(self.eval_statistics)
        return stats

    def end_epoch(self, epoch):
        self._need_to_update_eval_statistics = True

    @property
    def networks(self):
        nets = [self.policy, ]
        if self.task_encoder is not None:
            nets.append(self.task_encoder.encoder)
        if self.finetune_lang_enc:
            nets.append(self.lang_encoder)
        if self.target_emb_net is not None:
            nets.append(self.target_emb_net)
        return nets

    def get_snapshot(self):
        snapshot = dict(policy=self.policy)
        if self.task_encoder is not None:
            snapshot.update(task_encoder=self.task_encoder)
        if self.target_emb_net is not None:
            snapshot.update(target_emb_net=self.target_emb_net)
        if self.lang_encoder is not None:
            snapshot.update(lang_encoder=self.lang_encoder)
        return snapshot
