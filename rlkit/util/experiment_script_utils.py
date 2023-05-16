from collections import Counter
import warnings

import h5py
import numpy as np
import os

from rlkit.data_management.multitask_replay_buffer import (
    MultitaskReplayBufferHdf5, ObsDictMultiTaskReplayBuffer)
from rlkit.util.roboverse_utils import (
    add_multitask_data_to_multitask_buffer_v2,
    add_multitask_data_to_multitask_buffer_v3,
    add_data_to_multitask_buffer_v3,
)
from rlkit.torch.networks.cnn import ClipWrapper


def init_filled_buffer(
        buffer_datas, variant, max_replay_buffer_size, env,
        buffer_task_idxs, observation_keys, internal_keys, num_tasks,
        buffer_embeddings, success_only, gripper_idx, video_encoder=None,
        ext=None, max_num_demos_per_task=None,
        sep_gripper_action=False, gripper_smear_ds_actions=False):
    if ext == "npy":
        buf = init_filled_buffer_npy(
            buffer_datas, variant, max_replay_buffer_size, env,
            buffer_task_idxs, observation_keys, internal_keys, num_tasks,
            buffer_embeddings, success_only, gripper_idx,
            video_encoder=video_encoder, sep_gripper_action=sep_gripper_action,
            gripper_smear_ds_actions=gripper_smear_ds_actions)
    elif ext == "hdf5":
        buf = init_filled_buffer_hdf5(
            buffer_datas, variant, video_encoder, success_only,
            max_num_demos_per_task, buffer_task_idxs, observation_keys,
            buffer_embeddings, num_tasks, sep_gripper_action, gripper_idx,
            gripper_smear_ds_actions, variant['max_path_length'])
    else:
        raise NotImplementedError
    return buf


def init_filled_buffer_npy(
        buffer_datas, variant, max_replay_buffer_size, env,
        buffer_task_idxs, observation_keys, internal_keys, num_tasks,
        buffer_embeddings, success_only, gripper_idx, video_encoder=None,
        sep_gripper_action=False, gripper_smear_ds_actions=False):

    assert isinstance(buffer_datas, list)

    replay_buffer = ObsDictMultiTaskReplayBuffer(
        max_replay_buffer_size,
        env,
        buffer_task_idxs,
        path_len=variant['max_path_length'],
        use_next_obs_in_context=False,
        sparse_rewards=False,
        observation_keys=observation_keys,
        internal_keys=internal_keys,
        video_encoder=video_encoder,
        sep_gripper_action=sep_gripper_action,
        gripper_idx=gripper_idx,
        gripper_smear_ds_actions=gripper_smear_ds_actions,
    )

    using_clip_vid_enc = (
        hasattr(video_encoder, "encoder")
        and isinstance(video_encoder.encoder, ClipWrapper))

    for data in buffer_datas:
        if variant['task_embedding'] == "none":
            add_data_to_multitask_buffer_v3(
                data, replay_buffer, observation_keys + internal_keys,
                num_tasks, success_only=success_only)
        elif variant['task_embedding'] == "lang" and not using_clip_vid_enc:
            add_multitask_data_to_multitask_buffer_v3(
                data, replay_buffer, observation_keys, num_tasks,
                buffer_embeddings, success_only=success_only)
        elif (variant['task_embedding'] in ["demo", "demo_lang", "mcil"]
                or using_clip_vid_enc):
            add_multitask_data_to_multitask_buffer_v3(
                data, replay_buffer, observation_keys + internal_keys,
                num_tasks, None, success_only=success_only)
        elif variant['task_embedding'] == "onehot":
            add_multitask_data_to_multitask_buffer_v2(
                data, replay_buffer, observation_keys, num_tasks,
                success_only=success_only)
        else:
            raise NotImplementedError

    print(dict([
        (i, replay_buffer.task_buffers[i]._top // variant['max_path_length'])
        for i in buffer_task_idxs]))

    return replay_buffer


def init_filled_buffer_hdf5(
        buffer_datas, variant, video_encoder, success_only,
        max_num_demos_per_task, buffer_task_idxs, obs_keys,
        task_idx_to_emb_map, num_tasks, sep_gripper_action,
        gripper_idx, gripper_smear_ds_actions, max_path_len):
    assert len(buffer_datas) == 1, (
        "No support for multiple hdf5 buffer datas yet")
    buf = MultitaskReplayBufferHdf5(
        hdf5_path=buffer_datas[0],
        bsz=variant['batch_size'],
        video_encoder=video_encoder,
        task_emb=variant['task_embedding'],
        num_tasks=num_tasks,
        num_data_workers=0,
        hdf5_cache_mode=variant['hdf5_cache_mode'],
        subseq_len=variant['hdf5_subseq_len'],
        demo_enc_frame_ranges=variant['vid_enc_frame_ranges'],
        success_only=success_only,
        max_num_demos_per_task=max_num_demos_per_task,
        task_idxs=buffer_task_idxs,
        obs_keys=obs_keys,
        task_idx_to_emb_map=task_idx_to_emb_map,
        sep_gripper_action=sep_gripper_action,
        gripper_idx=gripper_idx,
        gripper_smear_ds_actions=gripper_smear_ds_actions,
        max_path_len=max_path_len)
    # Strangely, 2 and 5 are slightly slower than num_data_workers=0
    return buf


def create_task_indices(
        eval_task_interval_str_list, train_task_interval_str_list,
        train_target_task_interval_str_list, buffer_datas,
        train_target_buffer_datas, num_tasks, train_ext, train_target_ext):
    train_target_task_indices = get_task_indices_from_buffer(
        train_target_buffer_datas, train_target_ext)
    if len(eval_task_interval_str_list) == 0:
        eval_task_indices = list(range(num_tasks))
        train_task_indices = eval_task_indices
    elif len(train_task_interval_str_list) == 0:
        eval_task_indices = create_task_indices_from_task_int_str_list(
            eval_task_interval_str_list, num_tasks)
        train_task_indices = get_task_indices_from_buffer(
            buffer_datas, train_ext)
    else:
        eval_task_indices = create_task_indices_from_task_int_str_list(
            eval_task_interval_str_list, num_tasks)
        train_task_indices = create_task_indices_from_task_int_str_list(
            train_task_interval_str_list, num_tasks)
        if len(set(eval_task_indices).intersection(
                set(train_task_indices))) > 0:
            warnings.warn("Train and Eval task indices overlap.")

    if len(train_target_task_indices) > 0:
        if len(train_target_task_interval_str_list) != 0:
            # Override the task indices in the buffer if it was provided
            # on command line
            train_target_buffer_task_indices = train_target_task_indices
            train_target_task_indices = (
                create_task_indices_from_task_int_str_list(
                    train_target_task_interval_str_list, num_tasks))
            assert set(train_target_task_indices).issubset(
                set(train_target_buffer_task_indices))

        assert not set(train_target_task_indices).issubset(
            set(train_task_indices))
        assert set(train_target_task_indices).issubset(set(eval_task_indices))

        # Merge train_task_indices and train_target_task_indices
        train_task_indices = set(train_task_indices)
        train_task_indices.update(train_target_task_indices)
        train_task_indices = sorted(list(train_task_indices))

    combined_task_indices = set(eval_task_indices)
    combined_task_indices.update(set(train_task_indices))
    combined_task_indices = sorted(list(combined_task_indices))
    if combined_task_indices != list(range(num_tasks)):
        warnings.warn(
            "Train and Eval task indices don't cover all possible tasks.")

    return train_task_indices, eval_task_indices, train_target_task_indices


def create_task_indices_from_task_int_str_list(
        task_interval_str_list, num_tasks):
    task_idx_interval_list = []
    for interval in task_interval_str_list:
        interval = tuple([int(x) for x in interval.split("-")])
        assert len(interval) == 2

        if len(task_idx_interval_list) >= 1:
            # Make sure most recently added interval's endpoint is smaller than
            # current interval's startpoint.
            assert task_idx_interval_list[-1][-1] < interval[0]

        task_idx_interval_list.append(interval)

    task_indices = []  # to collect_data on
    for interval in task_idx_interval_list:
        start, end = interval
        assert 0 <= start <= end <= num_tasks
        task_indices.extend(list(range(start, end + 1)))

    return task_indices


def get_task_indices_from_buffer(buffer_datas, ext):
    assert isinstance(buffer_datas, list)
    """buffer_datas is a list of np.load-ed arrays."""
    all_task_indices = set()
    for buffer_data in buffer_datas:
        if ext == "npy":
            task_indices = [buffer_data[i]['env_infos'][0]['task_idx']
                            for i in range(len(buffer_data))]
        elif ext == "hdf5":
            # buffer_data is an fpath in hdf5 ext.
            with h5py.File(buffer_data, 'r', swmr=True, libver='latest') as h5f:
                task_indices = list(h5f['data'].keys())  # list of strings
            task_indices = [int(i) for i in task_indices]
        else:
            raise NotImplementedError
        all_task_indices.update(task_indices)
    all_task_indices = sorted(list(all_task_indices))
    return all_task_indices


def filter_out_task_indices_from_buffer(buffer_datas, task_indices_to_keep):
    """Only keep trajectories in buffer that have a certain task index"""
    traj_indices_to_delete = []
    for buf_idx in range(len(buffer_datas)):
        for i in range(len(buffer_datas[buf_idx])):
            if (buffer_datas[buf_idx][i]['env_infos'][0]['task_idx']
                    not in task_indices_to_keep):
                traj_indices_to_delete.append(i)
        buffer_datas[buf_idx] = np.delete(
            buffer_datas[buf_idx], traj_indices_to_delete)
    return buffer_datas


def maybe_truncate_buffer(buffer_datas, max_trajs_per_task, ext):
    """Only keep trajectories in buffer that have a certain task index"""
    if max_trajs_per_task is None or ext == "hdf5":
        # Nothing to truncate OR
        # truncation happens when filling hdf5 buffer.
        return buffer_datas

    assert ext == "npy"
    print("truncating npy buffer...")
    traj_indices_to_delete = []
    num_trajs_per_task = Counter()
    for buf_idx in range(len(buffer_datas)):
        orig_buffer_i_size = len(buffer_datas[buf_idx])
        for i in range(orig_buffer_i_size):
            traj_task_idx = (
                buffer_datas[buf_idx][i]['env_infos'][0]['task_idx'])
            num_trajs_per_task[traj_task_idx] += 1
            if num_trajs_per_task[traj_task_idx] > max_trajs_per_task:
                traj_indices_to_delete.append(i)
        buffer_datas[buf_idx] = np.delete(
            buffer_datas[buf_idx], traj_indices_to_delete)
        if len(traj_indices_to_delete) > 0:
            print(f"truncated array...{len(traj_indices_to_delete)} out of "
                  f"{orig_buffer_i_size} trajs thrown away.")
    return buffer_datas


def load_buffer_datas(buffer_fpaths):
    def assert_all_exts_same(buffer_fpaths):
        def get_ext(fpath):
            ext = os.path.splitext(buffer_fpaths[0])[-1]
            return ext[1:]  # ".npy" --> "npy"
        if len(buffer_fpaths) == 0:
            return ""
        first_ext = get_ext(buffer_fpaths[0])
        for i, fpath in enumerate(buffer_fpaths):
            ext_i = get_ext(fpath)
            assert ext_i == first_ext
        return first_ext

    assert isinstance(buffer_fpaths, list)
    dataset_ext = assert_all_exts_same(buffer_fpaths)
    buffer_datas = []
    for buffer_fp in buffer_fpaths:
        if dataset_ext == "npy":
            with open(buffer_fp, 'rb') as fl:
                data = np.load(fl, allow_pickle=True)
                buffer_datas.append(data)
        elif dataset_ext == "hdf5":
            # Leave filepaths unprocessed.
            buffer_datas.append(buffer_fp)
        else:
            raise NotImplementedError
    return buffer_datas, dataset_ext


def filter_buffer_if_needed(buffer_datas, task_indices_to_keep, ext):
    if len(buffer_datas) == 0 or ext == "hdf5":
        # Filtering for hdf5 will happen when loading data
        return buffer_datas

    assert ext == "npy"
    buffer_task_indices = get_task_indices_from_buffer(buffer_datas, ext)
    if set(buffer_task_indices) != set(task_indices_to_keep):
        assert set(task_indices_to_keep).issubset(set(buffer_task_indices))
        print("filtering...")
        buffer_datas = filter_out_task_indices_from_buffer(
            buffer_datas, task_indices_to_keep)
    return buffer_datas


def create_vid_enc_frame_ranges(
        vid_enc_start_frame_range, vid_enc_final_frame_offset_range,
        max_path_len, k, tupify_arg_fn):
    """k = m * n == size of mosaic"""
    if vid_enc_start_frame_range is None:
        vid_enc_start_frame_range = (0, 1)
    else:
        vid_enc_start_frame_range = tupify_arg_fn(vid_enc_start_frame_range)

    if vid_enc_final_frame_offset_range is None:
        vid_enc_final_frame_offset_range = (max_path_len - 1, max_path_len)
    else:
        vid_enc_final_frame_offset_range = tupify_arg_fn(
            vid_enc_final_frame_offset_range)

    if k == 1:
        frame_ranges = [vid_enc_final_frame_offset_range]
        return frame_ranges

    frame_ranges = [vid_enc_start_frame_range]
    num_ts_from_start_to_end = (
        vid_enc_final_frame_offset_range[1] - vid_enc_start_frame_range[0])
    for i in range(1, k - 1):
        if i == 1:
            start = int((i / k) * num_ts_from_start_to_end)
        else:
            start = frame_ranges[-1][1]
        end = (vid_enc_start_frame_range[0] +
               int(((i + 1) / k) * num_ts_from_start_to_end))
        assert start < end
        frame_i_range = (start, end)
        frame_ranges.append(frame_i_range)
    frame_ranges.append(vid_enc_final_frame_offset_range)

    assert len(frame_ranges) == k

    return frame_ranges


def set_train_task_sample_probs(
        train_task_indices, focus_train_task_indices,
        focus_train_tasks_sample_prob):
    """
    Produces `sample_probs`, an array of probs that sum to one,
    where we sample task `train_task_indices[i]` w.p.
    `sample_probs[i]`.
    """

    if focus_train_tasks_sample_prob is None:
        return None

    assert set(focus_train_task_indices).issubset(set(train_task_indices))

    focus_idxs = set(focus_train_task_indices)
    non_focus_idxs = set(train_task_indices) - focus_idxs
    focus_set_prob = focus_train_tasks_sample_prob
    non_focus_set_prob = 1 - focus_train_tasks_sample_prob

    sample_probs = []
    for idx in train_task_indices:
        if idx in focus_idxs:
            prob = (1 / len(focus_idxs)) * focus_set_prob
        elif idx in non_focus_idxs:
            prob = (1 / len(non_focus_idxs)) * non_focus_set_prob
        else:
            raise ValueError
        sample_probs.append(prob)

    assert len(train_task_indices) == len(sample_probs)
    assert abs(sum(sample_probs) - 1.0) < 1e-6

    return sample_probs


def maybe_flatten_list_of_singleton_lists(list_of_lists):
    is_all_singleton = all([len(lst) == 1 for lst in list_of_lists])
    if not is_all_singleton:
        return list_of_lists
    else:
        return [lst[0] for lst in list_of_lists]


def tupify_arg(x):
    return eval("(" + x + ")")
