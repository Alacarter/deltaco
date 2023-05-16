from collections import deque, OrderedDict
from functools import partial

import numpy as np

from rlkit.util.eval_util import create_stats_ordered_dict
from rlkit.data_collector.base import PathCollector
from rlkit.data_collector.rollout_functions import rollout


class MdpPathCollector(PathCollector):
    def __init__(
            self,
            env,
            policy,
            max_num_epoch_paths_saved=None,
            rollout_fn=rollout,
            save_env_in_snapshot=True,
            task_embedding_type=None,
            **kwargs
    ):
        """

        :param env:
        :param policy:
        :param max_num_epoch_paths_saved: Maximum number of paths to save per
        epoch for computing statistics.
        :param rollout_fn: Some function with signature
        ```
        def rollout_fn(
            env, policy, max_path_length, *args, **kwargs
        ) -> List[Path]:
        ```

        :param save_env_in_snapshot: If True, save the environment in the
        snapshot.
        :param kwargs: Unused kwargs are passed on to `rollout_fn`
        """
        self._env = env
        self._policy = policy
        self._max_num_epoch_paths_saved = max_num_epoch_paths_saved
        self._epoch_paths = deque(maxlen=self._max_num_epoch_paths_saved)
        self._rollout_fn = partial(rollout_fn, **kwargs)

        self._num_steps_total = 0
        self._num_paths_total = 0

        self._save_env_in_snapshot = save_env_in_snapshot
        self.task_embedding_type = task_embedding_type

    def collect_new_paths(
            self,
            max_path_length,
            num_steps,
            discard_incomplete_paths,
            multi_task=False,
            task_index=0,
            log_obj_info=False,
            singletask_buffer=False,
    ):
        paths = []
        if log_obj_info:
            infos_list = []
        num_steps_collected = 0
        rollouts_collected = 0
        if multi_task:
            self._env.reset_task(task_index)
        # print("num_steps", num_steps)

        while num_steps_collected < num_steps:
            # print("num_steps_collected", num_steps_collected)
            max_path_length_this_loop = min(  # Do not go over num_steps
                max_path_length,
                num_steps - num_steps_collected,
            )

            obs_processor_kwargs = {}
            if self.task_embedding_type == "mcil":
                # Alternate between using z_demo and z_lang for conditioning the policy;
                # MCIL doesn't use both at the same time.
                obs_processor_kwargs["emb_obs_key_idx"] = rollouts_collected % 2

            if log_obj_info:
                path, infos = self._rollout_fn(
                    self._env,
                    self._policy,
                    max_path_length=max_path_length_this_loop,
                    obs_processor_kwargs=obs_processor_kwargs,
                )
            else:
                path = self._rollout_fn(
                    self._env,
                    self._policy,
                    max_path_length=max_path_length_this_loop,
                    obs_processor_kwargs=obs_processor_kwargs,
                )

            self._env.reset()

            path_len = len(path['actions'])
            if (
                    path_len != max_path_length
                    and not path['terminals'][-1]
                    and discard_incomplete_paths
            ):
                break
            num_steps_collected += path_len
            paths.append(path)
            rollouts_collected += 1
            if log_obj_info:
                infos_list.append(infos)
        self._num_paths_total += len(paths)
        self._num_steps_total += num_steps_collected
        self._epoch_paths.extend(paths)
        if log_obj_info:
            return paths, infos_list
        else:
            return paths

    def get_epoch_paths(self):
        return self._epoch_paths

    def end_epoch(self, epoch):
        self._epoch_paths = deque(maxlen=self._max_num_epoch_paths_saved)

    def get_diagnostics(self):
        path_lens = [len(path['actions']) for path in self._epoch_paths]
        stats = OrderedDict([
            ('num steps total', self._num_steps_total),
            ('num paths total', self._num_paths_total),
        ])
        stats.update(create_stats_ordered_dict(
            "path length",
            path_lens,
            always_show_all_stats=True,
        ))
        return stats

    def get_snapshot(self):
        snapshot_dict = dict(
            policy=self._policy,
        )
        if self._save_env_in_snapshot:
            snapshot_dict['env'] = self._env
        return snapshot_dict


class ObsDictPathCollector(MdpPathCollector):
    def __init__(
            self,
            *args,
            observation_keys=['observation',],
            emb_obs_keys=[],
            task_embedding_type=None,
            task_emb_input_mode="concat_to_img_embs",
            emb_key_to_concat=None,
            aux_tasks=[],
            **kwargs
    ):
        # Create obs keys separated by emb and non_emb
        self.emb_obs_keys = list(emb_obs_keys)
        self.non_emb_obs_keys = []

        for obs_key in observation_keys:
            if obs_key not in self.emb_obs_keys:
                self.non_emb_obs_keys.append(obs_key)

        if task_emb_input_mode in ["film_video_concat_lang", "film_lang_concat_video"]:
            assert emb_key_to_concat in self.emb_obs_keys
            assert len(self.emb_obs_keys) == 2
            self.non_emb_obs_keys.append(emb_key_to_concat) # should go at the end of the list
            self.emb_obs_keys.remove(emb_key_to_concat)

        def obs_processor(obs):
            return np.concatenate([obs[key] for key in observation_keys])

        def film_obs_processor(obs):
            out_dict = {}
            out_dict["non_emb_obs"] = np.concatenate([np.squeeze(obs[key]) for key in self.non_emb_obs_keys])
            out_dict["emb"] = [obs[emb_obs_key] for emb_obs_key in self.emb_obs_keys]
            return out_dict

        def mcil_film_obs_processor(obs, emb_obs_key_idx):
            """
            emb_obs_key_idx is a number from {0, 1, ..., len(self.emb_obs_keys) - 1}
            that dictates which emb_obs_key (either lang or video)
            we pass to the policy as the task emb
            """
            assert emb_obs_key_idx in range(len(self.emb_obs_keys))
            out_dict = {}
            out_dict["non_emb_obs"] = np.concatenate([np.squeeze(obs[key]) for key in self.non_emb_obs_keys])
            out_dict["emb"] = [obs[self.emb_obs_keys[emb_obs_key_idx]]]
            return out_dict

        if task_embedding_type == "mcil":
            assert len(self.emb_obs_keys) == 2
            preprocess_obs_for_policy_fn = mcil_film_obs_processor
        elif task_emb_input_mode in ["film", "film_video_concat_lang", "film_lang_concat_video"]:
            assert len(self.emb_obs_keys) > 0
            preprocess_obs_for_policy_fn = film_obs_processor
        else:
            preprocess_obs_for_policy_fn = obs_processor

        rollout_fn = partial(
            rollout,
            preprocess_obs_for_policy_fn=preprocess_obs_for_policy_fn,
            task_emb_input_mode=task_emb_input_mode,
            aux_tasks=aux_tasks,
        )
        super().__init__(
            *args, rollout_fn=rollout_fn,
            task_embedding_type=task_embedding_type, **kwargs)
        self._observation_keys = observation_keys

    def get_snapshot(self):
        snapshot = super().get_snapshot()
        snapshot.update(
            observation_keys=self._observation_keys,
        )
        return snapshot
