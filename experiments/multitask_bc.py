import argparse
import os
import time

from gym import spaces
import numpy as np
import torch

# Local Imports
from rlkit.core import logger
from rlkit.env_wrappers.embedding_wrappers import (
    EmbeddingWrapperOffline,
    TrainableEmbeddingWrapper,
    VideoTargetEmbeddingWrapper,
)

from rlkit.launchers.config import LOCAL_LOG_DIR
from rlkit.launchers.launcher_util import setup_logger

from rlkit.util.roboverse_utils import (
    VideoSaveFunctionBullet, PlotSaveFunctionBullet,
    get_buffer_size_multitask
)

from rlkit.util.pythonplusplus import identity

from rlkit.data_collector import ObsDictPathCollector

from rlkit.torch.networks import Mlp
from rlkit.torch.networks.cnn import CNN, StandaloneCNN, ClipWrapper
from rlkit.torch.networks.film import HIDDEN_ACTIVATION_STR_TO_FN_MAP
from rlkit.torch.networks.resnet import ResNet
from rlkit.torch.pretrained_models.language_models import (
    LM_STR_TO_CLASS_MAP, LM_STR_TO_FN_CLASS_MAP, LONGEST_SENTENCE_LEN)
from rlkit.torch.policies import (
    GaussianCNNPolicy, GaussianStandaloneCNNPolicy, MakeDeterministic)
from rlkit.torch.alg.bc_trainer import BCTrainer
from rlkit.torch.alg.torch_rl_algorithm import TorchBatchRLAlgorithm

import rlkit.util.experiment_script_utils as exp_utils
from rlkit.util.misc_functions import l2_unit_normalize
import rlkit.util.pytorch_util as ptu
from rlkit.util.video_encoders import (
    BCZVideoEncoder, VIDEO_ENCODER_LOSS_TYPE_TO_FN_MAP)
import roboverse


def get_embeddings_from_str_list(
        emb_model, task_strs, variant, return_type="embs", gpu=0):
    def get_embeddings_from_list_of_strs(emb_model, task_strs, return_type):
        assert return_type in ["embs", "tokens"]
        if variant['lang_emb_model_type'] == "distilbert":
            task_strs_tokenized = emb_model.tokenize_strs(task_strs)
            task_strs_tokenized = torch.Tensor(task_strs_tokenized).long().to(
                f"cuda:{gpu}")
            embeddings = emb_model(task_strs_tokenized)
        elif variant['lang_emb_model_type'] in ["distilroberta", "minilm"]:
            task_strs_tokenized = emb_model.model.tokenize(task_strs)
            # Tokens not used unless finetuning emb_model
            embeddings = emb_model(task_strs)
        elif variant['lang_emb_model_type'] == "clip":
            task_strs_tokenized = emb_model.tokenize_fn(task_strs)
            embeddings = emb_model(
                images=None, texts=task_strs_tokenized, train_mode=False)
        else:
            raise NotImplementedError

        if return_type == "tokens":
            embeddings = task_strs_tokenized
        embeddings = embeddings.cpu().numpy()
        print("embeddings.shape", embeddings.shape)
        return embeddings

    return_type = "embs"
    if variant['finetune_lang_enc']:
        return_type = "tokens"

    if isinstance(task_strs[0], str):
        embeddings = get_embeddings_from_list_of_strs(
            emb_model, task_strs, return_type)
        return embeddings
    elif isinstance(task_strs[0], list):
        embeddings = []
        # return a list of 2D tensors of (k, 768) in size. (k can = 1)
        # one 2D tensor for each task.
        for task_str_list in task_strs:
            embs_for_task = get_embeddings_from_str_list(
                emb_model, task_str_list, variant, return_type=return_type)
            embeddings.append(embs_for_task)
        return embeddings
    else:
        raise ValueError


def init_policy_net(variant, policy_params, cnn_params):
    def maybe_add_film_to_cnn_policy_params(
            cnn_params, policy_params, variant):
        if variant['task_emb_input_mode'] in [
                "film", "film_video_concat_lang", "film_lang_concat_video"]:
            cnn_params['film_emb_dim_list'] = policy_params.pop(
                'film_emb_dim_list')
            assert len(cnn_params['film_emb_dim_list']) > 0
            cnn_params['num_film_inputs'] = (
                variant['trainer_kwargs']['policy_num_film_inputs'])
            cnn_params['film_hidden_sizes'] = (
                variant['trainer_kwargs']['film_hidden_sizes'])
            cnn_params['film_hidden_activation'] = (
                variant['trainer_kwargs']['film_hidden_activation'])
            policy_params['film_attn_hidden_sizes'] = (
                variant['policy_film_attn_hidden_sizes'])
            if variant['policy_film_attn_mode'] != "none":
                cnn_params['use_film_attn'] = True

            if len(cnn_params['film_hidden_sizes']) == 0:
                print("Note: Only using a projection "
                      "for the film gamma/beta networks.")

    if variant['policy_cnn_type'] == "plain":
        policy_params.pop("freeze_policy_cnn")
        policy = GaussianCNNPolicy(**policy_params, **cnn_params)
    elif variant['policy_cnn_type'] == "r3m":
        from r3m import load_r3m
        r3m_cnn = load_r3m("resnet18")
        assert r3m_cnn.training, "Not in training mode"
        policy_params['cnn_out_dim'] = r3m_cnn.module.outdim
        print("policy_params", policy_params)
        print("cnn_params", cnn_params)
        policy = GaussianStandaloneCNNPolicy(
            cnn=r3m_cnn, **policy_params, **cnn_params)
    elif variant['policy_cnn_type'] in ["resnet18", "resnet34"]:
        # No output activation for ResNet.
        policy_params['added_fc_input_size'] = cnn_params.pop(
            "added_fc_input_size")
        maybe_add_film_to_cnn_policy_params(cnn_params, policy_params, variant)
        print("cnn_params", cnn_params)
        cnn = ResNet(**cnn_params)
        policy = GaussianStandaloneCNNPolicy(
            cnn=cnn, **policy_params, **cnn_params)
    elif variant['policy_cnn_type'] == "plain_standalone":
        if "aug_transforms" in policy_params:
            assert "aug_transforms" not in cnn_params
            # Else there will be two sets of image augmentations.
        policy_params['added_fc_input_size'] = cnn_params.pop(
            "added_fc_input_size")
        maybe_add_film_to_cnn_policy_params(cnn_params, policy_params, variant)
        for key in ["hidden_sizes"]:
            # These keys will be set to default value in CNN class.
            cnn_params.pop(key)
        print("cnn_params", cnn_params)
        cnn = CNN(**cnn_params)
        print("policy_params", policy_params)
        policy = GaussianStandaloneCNNPolicy(
            cnn=cnn, **policy_params, **cnn_params)

    if variant['policy_params']['gripper_policy_arch'] == "sep_head":
        assert isinstance(policy, GaussianStandaloneCNNPolicy)

    # Override policy with provided checkpoint.
    if variant['policy_ckpt'] is not None:
        del policy
        policy = args.policy_ckpt_dict['trainer/policy']
    return policy


def init_emb_keys_and_obs_space(
        variant, args, eval_env, num_tasks,
        emb_model_lang_out_dim=None, emb_model_visual_out_dim=None):
    def update_obs_space(env, obs_key, dim):
        # Used for buffer array allocation for obs.
        env.observation_space.spaces.update(
            {obs_key: spaces.Box(
                low=np.array([-1] * dim),
                high=np.array([1] * dim),
            )})

    policy_film_emb_dim_list = []
    vid_enc_film_emb_dim_list = []
    latent_fc_dim = 0
    emb_key_to_concat = None
    if variant['task_embedding'] in ["lang", "demo", "demo_lang", "mcil"]:
        emb_obs_keys = ["lang_embedding"]

        assert isinstance(emb_model_lang_out_dim, int)
        assert isinstance(emb_model_visual_out_dim, int)

        task_emb_to_latent_dim_map = {
            "lang": emb_model_lang_out_dim,
            "demo": emb_model_visual_out_dim,
            "demo_lang": emb_model_visual_out_dim + emb_model_lang_out_dim,
            "mcil": emb_model_visual_out_dim,
        }

        if variant['task_embedding'] == "mcil":
            assert emb_model_visual_out_dim == emb_model_lang_out_dim

        if args.transform_targets:
            task_emb_to_latent_dim_map.update(dict(
                demo=variant['latent_out_dim_size'],
                demo_lang=2 * variant['latent_out_dim_size'],
                mcil=variant['latent_out_dim_size'],
            ))

        cascaded_film_inputs = (
            variant['task_embedding'] == "demo_lang" and
            args.policy_num_film_inputs == 1 and
            args.vid_enc_num_film_inputs == 1)

        if variant['task_embedding'] == "demo_lang":
            if ((args.policy_num_film_inputs in [2, 3]) or
                (args.task_emb_input_mode in [
                    "film_video_concat_lang", "film_lang_concat_video"])):
                # (1) When there are multiple emb_obs_keys, each key must
                # contain the modality name as a substring.
                # This is used in embedding_wrappers.py
                # (2) Make sure this matches the order in bc_trainer.py
                if args.policy_film_input_order in ["vl", ""]:
                    emb_obs_keys = ["video_embedding", "lang_embedding"]
                elif args.policy_film_input_order == "lv":
                    emb_obs_keys = ["lang_embedding", "video_embedding"]
                else:
                    raise NotImplementedError
            if args.task_emb_input_mode == "film_video_concat_lang":
                emb_key_to_concat = "lang_embedding"
            elif args.task_emb_input_mode == "film_lang_concat_video":
                emb_key_to_concat = "video_embedding"
        elif variant['task_embedding'] == "mcil":
            emb_obs_keys = ["video_embedding", "lang_embedding"]
        elif variant['task_embedding'] == "demo":
            emb_obs_keys = ['video_embedding']

        video_latent_dim = task_emb_to_latent_dim_map['demo']
        lang_latent_dim = task_emb_to_latent_dim_map['lang']

        emb_obs_keys_to_dim_map = {}
        if len(emb_obs_keys) == 1:
            emb_obs_keys_to_dim_map[emb_obs_keys[0]] = (
                task_emb_to_latent_dim_map[variant['task_embedding']])
        elif len(emb_obs_keys) >= 2:
            for emb_obs_key in emb_obs_keys:
                if "video" in emb_obs_key:
                    emb_obs_keys_to_dim_map[emb_obs_key] = video_latent_dim
                elif "lang" in emb_obs_key:
                    emb_obs_keys_to_dim_map[emb_obs_key] = lang_latent_dim
                else:
                    raise NotImplementedError
        else:
            raise NotImplementedError

        for emb_obs_key in emb_obs_keys:
            emb_obs_key_dim = emb_obs_keys_to_dim_map[emb_obs_key]

            # Override the lang_embedding if using finetuned lang enc.
            # I don't think env.observation_space is used anywhere else
            # besides buffer creation.
            if (emb_obs_key == "lang_embedding" and
                    variant['finetune_lang_enc']):
                emb_obs_key_dim = LONGEST_SENTENCE_LEN

            update_obs_space(eval_env, emb_obs_key, emb_obs_key_dim)

        # Set vid_enc_film_emb_dim_list
        if cascaded_film_inputs:
            # Only Language passed to vid enc
            vid_enc_film_emb_dim_list = [lang_latent_dim]

        # Set policy_film_emb_dim_list and latent_fc_dim
        if cascaded_film_inputs:
            policy_film_emb_dim_list = [video_latent_dim]
        elif args.task_emb_input_mode == "film_video_concat_lang":
            policy_film_emb_dim_list = [video_latent_dim]
            latent_fc_dim = lang_latent_dim
        elif args.task_emb_input_mode == "film_lang_concat_video":
            policy_film_emb_dim_list = [lang_latent_dim]
            latent_fc_dim = video_latent_dim
        elif args.policy_num_film_inputs == 0:
            assert variant['task_emb_input_mode'] == "concat_to_img_embs"
            latent_fc_dim = task_emb_to_latent_dim_map[
                variant['task_embedding']]
        elif args.policy_num_film_inputs == 1:
            policy_film_emb_dim_list = [
                task_emb_to_latent_dim_map[variant['task_embedding']]]
        elif args.policy_num_film_inputs == 2:
            assert variant['task_embedding'] == "demo_lang"
            policy_film_emb_dim_list = [video_latent_dim, lang_latent_dim]
        elif args.policy_num_film_inputs == 3:
            assert variant['task_embedding'] == "demo_lang"
            policy_film_emb_dim_list = [
                emb_obs_keys_to_dim_map[emb_obs_key]
                for emb_obs_key in emb_obs_keys]
        else:
            raise NotImplementedError

    elif variant['task_embedding'] == "onehot":
        emb_obs_keys = ['one_hot_task_id']
        if args.policy_num_film_inputs == 0:
            latent_fc_dim = num_tasks
        elif args.policy_num_film_inputs == 1:
            policy_film_emb_dim_list = [num_tasks]
        emb_obs_keys_to_dim_map = {emb_obs_keys[0]: num_tasks}
        update_obs_space(eval_env, emb_obs_keys[0], num_tasks)
    elif variant['task_embedding'] == "none":
        emb_obs_keys = []
        emb_obs_keys_to_dim_map = {}
    else:
        raise NotImplementedError

    assert len(policy_film_emb_dim_list) == args.policy_num_film_inputs
    assert len(vid_enc_film_emb_dim_list) == args.vid_enc_num_film_inputs

    out_dict = dict(
        emb_obs_keys=emb_obs_keys,
        policy_film_emb_dim_list=policy_film_emb_dim_list,
        vid_enc_film_emb_dim_list=vid_enc_film_emb_dim_list,
        emb_obs_keys_to_dim_map=emb_obs_keys_to_dim_map,
        emb_key_to_concat=emb_key_to_concat,
        latent_fc_dim=latent_fc_dim,
    )

    return out_dict


def init_video_encoder(variant, args, clip_wrapper, vid_enc_film_emb_dim_list):
    # Initialize Video Encoder
    if variant['task_embedding'] in ["demo", "demo_lang", "mcil"]:
        loss_kwargs = dict()
        if variant['task_embedding'] in ["demo", "demo_lang"]:
            # Set loss_kwargs
            assert variant['task_encoder_loss_type'] is not None
            if (variant['task_encoder_loss_type'] in
                    ["contrastive", "cross_ent"]):
                loss_kwargs['temp'] = variant['task_encoder_contr_temp']
            if variant['task_encoder_loss_type'] == "cross_ent":
                loss_kwargs['meta_batch_size'] = variant['meta_batch_size']

        if variant['vid_enc_cnn_type'] == "resnet18":
            video_enc_net = ResNet(
                fc_layers=[variant['latent_out_dim_size']],
                output_activation=l2_unit_normalize)
        elif variant['vid_enc_cnn_type'] == "plain":
            if variant['task_encoder_cnn_params']['num_film_inputs'] > 0:
                variant['task_encoder_cnn_params']['film_emb_dim_list'] = (
                    vid_enc_film_emb_dim_list)
            video_enc_net = CNN(**variant['task_encoder_cnn_params'])
        elif variant['vid_enc_cnn_type'] == "r3m":
            from r3m import load_r3m
            r3m_cnn = load_r3m("resnet18")
            video_enc_net = StandaloneCNN(
                r3m_cnn, **variant['task_encoder_cnn_params'])
        elif variant['vid_enc_cnn_type'] == "clip":
            video_enc_net = clip_wrapper
        else:
            raise NotImplementedError

        if variant['policy_ckpt'] is not None:
            del video_enc_net
            video_encoder = args.policy_ckpt_dict['trainer/task_encoder']
            # Update loss args (like metabatch size)
            video_encoder.set_loss_criterion(
                variant['task_encoder_loss_type'], loss_kwargs)
        else:
            video_encoder = BCZVideoEncoder(
                video_enc_net,
                mosaic_rc=variant['vid_enc_mosaic_rc'],
                loss_type=variant['task_encoder_loss_type'],
                loss_kwargs=loss_kwargs,
                frame_ranges=variant['vid_enc_frame_ranges'],
                image_size=variant['image_size'],
                use_cached_embs=variant['use_cached_embs'],
                buffer_ext_dict=variant['buffer_ext_dict'])

        return video_encoder


def init_target_emb_net(variant, emb_dim_kwargs):
    target_emb_net = None
    if (variant['policy_ckpt'] is not None) and (
            'trainer/target_emb_net' in args.policy_ckpt_dict):
        target_emb_net = args.policy_ckpt_dict['trainer/target_emb_net']
    elif variant['transform_targets']:
        target_emb_net = Mlp(
            hidden_sizes=[max(128, emb_dim_kwargs['emb_model_lang_out_dim'])],
            output_size=variant['latent_out_dim_size'],
            input_size=emb_dim_kwargs['emb_model_lang_out_dim'],
            output_activation=l2_unit_normalize)
    return target_emb_net


def init_policy_nets_and_set_params(
        variant, eval_env, emb_obs_keys, emb_obs_keys_to_dim_map,
        policy_film_emb_dim_list, latent_fc_dim, num_tasks):
    action_dim = eval_env.action_space.low.size
    obs_key_to_dim_map = {}
    observation_keys = ['image']
    obs_key_to_dim_map['image'] = np.prod(variant['image_size'])

    if (variant['task_embedding'] in ["lang"] or
            variant['task_embedding'] == "onehot"):
        observation_keys.extend(emb_obs_keys)
        obs_key_to_dim_map.update(emb_obs_keys_to_dim_map)

    if variant['use_robot_state']:
        observation_keys.append('state')
        state_observation_dim = (
            eval_env.observation_space.spaces['state'].low.size)
    else:
        state_observation_dim = 0
    obs_key_to_dim_map['state'] = state_observation_dim

    aux_obs_bounds_dict = {}
    if len(variant['aux_tasks']) > 0:
        for aux_task in variant['aux_tasks']:
            aux_obs_bounds_dict[aux_task] = {}
            observation_keys.append(aux_task)
            obs_key_to_dim_map[aux_task] = (
                eval_env.observation_space.spaces[aux_task].low.size)
            aux_obs_bounds_dict[aux_task]["lo"] = (
                eval_env.observation_space.spaces[aux_task].low)
            aux_obs_bounds_dict[aux_task]["hi"] = (
                eval_env.observation_space.spaces[aux_task].high)

    # policy_obs_keys are only passed into policy;
    # only used to match obs_keys_to_dim_map
    if variant['task_emb_input_mode'] == "film_video_concat_lang":
        obs_key_to_dim_map["lang_embedding"] = (
            emb_obs_keys_to_dim_map["lang_embedding"])
        policy_obs_keys = list(observation_keys) + ["lang_embedding"]
    elif variant['task_emb_input_mode'] == "film_lang_concat_video":
        obs_key_to_dim_map["video_embedding"] = (
            emb_obs_keys_to_dim_map["video_embedding"])
        policy_obs_keys = list(observation_keys) + ["video_embedding"]
    else:
        policy_obs_keys = list(observation_keys)

    cnn_params = variant['cnn_params']
    policy_params = variant['policy_params']
    policy_params.update(
        state_obs_dim=state_observation_dim,
        emb_obs_keys=emb_obs_keys,
        aux_tasks=variant['aux_tasks'],
        aux_obs_bounds=aux_obs_bounds_dict,
        aux_to_feed_fc=variant['aux_to_feed_policy_fc'],
        obs_key_to_dim_map=obs_key_to_dim_map,
        observation_keys=policy_obs_keys,
        film_emb_dim_list=policy_film_emb_dim_list,
    )

    assert set(policy_params['obs_key_to_dim_map']).issubset(
        set(policy_params['observation_keys']))

    cnn_params.update(dict(
        added_fc_input_size=state_observation_dim + latent_fc_dim,
        # latent_fc_dim=latent_fc_dim,
    ))

    policy_params.update(action_dim=action_dim)

    policy = init_policy_net(variant, policy_params, cnn_params)

    other_nets = {}
    buffer_policy = None

    policy_and_params_out_dict = dict(
        policy=policy,
        observation_keys=observation_keys,
        cnn_params=cnn_params,
        action_dim=action_dim,
        other_nets=other_nets,
        buffer_policy=buffer_policy,
    )
    return policy_and_params_out_dict


def load_buffer_data_dict(args):
    # Load all buffer data once.
    buffer_datas, buffer_ext = exp_utils.load_buffer_datas(args.buffers)
    train_target_buffer_datas, train_target_buffer_ext = (
        exp_utils.load_buffer_datas(args.train_target_buffers))
    target_buffer_datas, target_buffer_ext = exp_utils.load_buffer_datas(
        args.target_buffers)

    train_task_indices, eval_task_indices, train_target_task_indices = (
        exp_utils.create_task_indices(
            args.eval_task_idx_intervals, args.train_task_idx_intervals,
            args.train_target_task_idx_intervals,
            buffer_datas, train_target_buffer_datas, args.num_tasks,
            buffer_ext, train_target_buffer_ext)
    )

    train_task_indices_wo_target = sorted(list(
        set(train_task_indices) - set(train_target_task_indices)))

    buffer_datas = exp_utils.maybe_truncate_buffer(
        buffer_datas, args.num_train_demos_per_task, buffer_ext)
    buffer_datas = exp_utils.filter_buffer_if_needed(
        buffer_datas, train_task_indices_wo_target, buffer_ext)

    train_target_buffer_datas = exp_utils.maybe_truncate_buffer(
        train_target_buffer_datas, args.num_train_target_demos_per_task, 
        train_target_buffer_ext)

    target_buffer_datas = exp_utils.filter_buffer_if_needed(
        target_buffer_datas, eval_task_indices, target_buffer_ext)

    if args.policy_ckpt is None and len(train_target_buffer_datas) == 0:
        # Not finetuning
        assert len(buffer_datas) > 0

    buffer_data_dict = dict(
        buffer_datas=buffer_datas + train_target_buffer_datas,
        target_buffer_datas=target_buffer_datas,
    )
    task_indices_dict = dict(
        train=train_task_indices,
        eval=eval_task_indices,
        train_target=train_target_task_indices,
        train_wo_target=train_task_indices_wo_target,
    )

    if train_target_buffer_ext != "":
        assert ((buffer_ext == train_target_buffer_ext) or
                buffer_ext == "")
    buffer_ext_dict = dict(
        train=buffer_ext or train_target_buffer_ext,
        target=target_buffer_ext,
    )

    assert len(buffer_data_dict['buffer_datas']) > 0

    return buffer_data_dict, task_indices_dict, buffer_ext_dict


def experiment(variant, buffer_data_dict):
    multiple_strs_per_task = bool(variant['random_target_obj_referent'])

    if variant['finetuning']:
        args.policy_ckpt_dict = torch.load(
            variant['policy_ckpt'], map_location="cuda:0")

    num_tasks = variant['num_tasks']
    env_num_tasks = num_tasks

    eval_task_indices = variant['eval_task_indices']
    train_task_indices = variant['train_task_indices']

    kwargs = {
        "observation_img_dim": variant['image_dim'],
        "observation_img_hd_dim": variant['eval_image_dim'],
    }
    if variant['distractor_obj_hard_mode_prob'] is not None:
        kwargs['distractor_obj_hard_mode_prob'] = (
            variant['distractor_obj_hard_mode_prob'])
    kwargs['deterministic_target_obj_referent'] = (
        not variant['random_target_obj_referent'])

    eval_env = roboverse.make(
        variant['env'], transpose_image=True, num_tasks=env_num_tasks,
        **kwargs)

    buffer_datas = buffer_data_dict['buffer_datas']

    max_replay_buffer_size = get_buffer_size_multitask(
        buffer_datas, success_only=True, output_type="dict",
        ext=variant['buffer_ext_dict']['train'])

    train_embeddings_map = None
    emb_model = None
    emb_dim_kwargs = {}

    if variant['task_embedding'] in ["lang", "demo", "demo_lang", "mcil"]:
        # Create instruction strings
        eval_env_task_lang_list = eval_env.get_task_lang_dict()['instructs']
        eval_task_strs = []
        for task_idx in eval_task_indices:
            eval_task_str = eval_env_task_lang_list[task_idx]
            if isinstance(eval_task_str, list):
                # First item in list is default string.
                # We only want one string for each eval task for
                # evaluation consistency.
                eval_task_str = eval_task_str[0]
            eval_task_strs.append(eval_task_str)

        # May be a list of strings or a list of list of strings.
        buffer_task_strs = [
            eval_env_task_lang_list[task_idx]
            for task_idx in train_task_indices]
        buffer_task_strs = exp_utils.maybe_flatten_list_of_singleton_lists(
            buffer_task_strs)
        if not multiple_strs_per_task:
            assert isinstance(buffer_task_strs[0], str)
        else:
            assert isinstance(buffer_task_strs[0], list)
            assert isinstance(buffer_task_strs[0][0], str)

        variant['eval_task_strs'] = dict(
            zip(eval_task_indices, eval_task_strs))
        variant['buffer_task_strs'] = dict(
            zip(train_task_indices, buffer_task_strs))
        print("eval_task_strs", eval_task_strs)
        print("buffer_task_strs", buffer_task_strs)
        print("len(eval_task_strs)", len(eval_task_strs))
        print("len(buffer_task_strs)", len(buffer_task_strs))

        # Initialize Language Encoder
        clip_wrapper = None
        if variant['vid_enc_cnn_type'] == "clip":
            variant['task_encoder_cnn_params']['task_lang_list'] = (
                eval_env_task_lang_list)
            clip_wrapper = ClipWrapper(**variant['task_encoder_cnn_params'])
            emb_dim_kwargs['emb_model_lang_out_dim'] = clip_wrapper.lang_outdim
            emb_dim_kwargs['emb_model_visual_out_dim'] = (
                clip_wrapper.visual_outdim)
            if variant['lang_emb_model_type'] == "clip":
                emb_model = clip_wrapper
        elif variant['lang_emb_model_type'] == "clip":
            # Only using CLIP for lang_embs, not for its CNN
            variant['clip_lang_encoder_params']['task_lang_list'] = (
                eval_env_task_lang_list)
            emb_model = ClipWrapper(**variant['clip_lang_encoder_params'])
            emb_dim_kwargs['emb_model_lang_out_dim'] = emb_model.lang_outdim
            emb_dim_kwargs['emb_model_visual_out_dim'] = emb_model.lang_outdim
        else:
            if variant['finetune_lang_enc']:
                emb_model_class = LM_STR_TO_CLASS_MAP[
                    variant['lang_emb_model_type']]
                # For ex, DistilBERT
            else:
                emb_model_class = LM_STR_TO_FN_CLASS_MAP[
                    variant['lang_emb_model_type']]
                # For ex, DistilBERTFunctional
            emb_model = emb_model_class(
                l2_unit_normalize=variant['l2_unit_normalize_target_embs'],
                gpu=0)
            emb_dim_kwargs['emb_model_lang_out_dim'] = emb_model.out_dim
            emb_dim_kwargs['emb_model_visual_out_dim'] = emb_model.out_dim

        if variant['task_embedding'] in ["demo", "demo_lang", "mcil"]:
            assert emb_dim_kwargs['emb_model_visual_out_dim'] == (
                variant['latent_out_dim_size'])
            # So that z_demo and z_lang (after potential transforms) are the
            # same dim.

        eval_embeddings_map = dict()
        train_embeddings_map = dict()
        if (variant['vid_enc_cnn_type'] == "clip" and
                not variant['use_cached_embs']):
            pass
        else:
            # includes case where variant['lang_emb_model_type'] == "clip"
            eval_embeddings = get_embeddings_from_str_list(
                emb_model, eval_task_strs, variant, gpu=0)
            # buffer_embeddings is either a 2D tensor or a list of 2D tensors.
            buffer_embeddings = get_embeddings_from_str_list(
                emb_model, buffer_task_strs, variant, gpu=0)
            # np.save("20220909_buffer_embeddings.npy", buffer_embeddings)
            # np.save("20220909_eval_embeddings.npy", eval_embeddings)

            for i, eval_task_idx in enumerate(eval_task_indices):
                eval_embeddings_map[eval_task_idx] = eval_embeddings[i]
            for i, train_task_idx in enumerate(train_task_indices):
                train_embeddings_map[train_task_idx] = buffer_embeddings[i]

        out_dict = init_emb_keys_and_obs_space(
            variant, args, eval_env, num_tasks, **emb_dim_kwargs)
        emb_obs_keys = out_dict['emb_obs_keys']
        policy_film_emb_dim_list = out_dict['policy_film_emb_dim_list']
        vid_enc_film_emb_dim_list = out_dict['vid_enc_film_emb_dim_list']
        emb_obs_keys_to_dim_map = out_dict['emb_obs_keys_to_dim_map']
        emb_key_to_concat = out_dict['emb_key_to_concat']
        latent_fc_dim = out_dict['latent_fc_dim']

        video_encoder = init_video_encoder(
            variant, args, clip_wrapper, vid_enc_film_emb_dim_list)
        target_emb_net = init_target_emb_net(variant, emb_dim_kwargs)

        if variant['task_embedding'] in ["lang"]:
            assert len(emb_obs_keys) == 1
            if variant['finetune_lang_enc']:
                eval_env = TrainableEmbeddingWrapper(
                    eval_env, lang_enc=emb_model, emb_obs_key=emb_obs_keys[0],
                    eval_video_batch_size=variant['eval_video_batch_size'],
                    eval_task_indices=eval_task_indices,
                    eval_task_idx_to_tokens_map=eval_embeddings_map)
            else:
                eval_env = EmbeddingWrapperOffline(
                    eval_env, embeddings=eval_embeddings_map,
                    emb_obs_key=emb_obs_keys[0])
        elif variant['task_embedding'] in ["demo", "demo_lang", "mcil"]:
            env_wrapper_kwargs = {}
            if variant['finetune_lang_enc']:
                env_wrapper_kwargs['lang_enc'] = emb_model

            eval_env = VideoTargetEmbeddingWrapper(
                eval_env,
                task_encoder=video_encoder,
                target_embeddings=eval_embeddings_map,
                emb_obs_keys=emb_obs_keys,
                eval_task_indices=eval_task_indices,
                eval_video_batch_size=variant['eval_video_batch_size'],
                task_embedding=variant['task_embedding'],
                path_len=variant['max_path_length'],
                target_emb_net=target_emb_net,
                policy_num_film_inputs=(
                    variant['trainer_kwargs']['policy_num_film_inputs']),
                task_emb_input_mode=variant['task_emb_input_mode'],
                **env_wrapper_kwargs)
        else:
            raise NotImplementedError
    elif variant['task_embedding'] == "onehot":
        out_dict = init_emb_keys_and_obs_space(
            variant, args, eval_env, num_tasks, **emb_dim_kwargs)
        emb_obs_keys = out_dict['emb_obs_keys']
        policy_film_emb_dim_list = out_dict['policy_film_emb_dim_list']
        vid_enc_film_emb_dim_list = out_dict['vid_enc_film_emb_dim_list']
        emb_obs_keys_to_dim_map = out_dict['emb_obs_keys_to_dim_map']
        emb_key_to_concat = out_dict['emb_key_to_concat']
        latent_fc_dim = out_dict['latent_fc_dim']

        eval_embeddings_map = dict(
            [(eval_task_idx, np.eye(num_tasks)[eval_task_idx])
             for eval_task_idx in eval_task_indices])
        eval_env = EmbeddingWrapperOffline(
            eval_env, embeddings=eval_embeddings_map,
            emb_obs_key=emb_obs_keys[0])
    elif variant['task_embedding'] == "none":
        pass
    else:
        raise NotImplementedError

    if not variant['finetune_lang_enc']:
        del emb_model

    exp_prefix = '{}-BC-image-{}'.format(time.strftime("%y-%m-%d"), args.env)

    setup_logger(logger, exp_prefix, LOCAL_LOG_DIR, variant=variant,
                 snapshot_mode=variant['save_checkpoint_mode'],
                 snapshot_gap=variant['save_checkpoint_freq'],
                 earliest_snapshot_epoch=(
                    variant['save_checkpoint_earliest_epoch']),
                 seed=variant['seed'])

    gripper_idx = eval_env.env.gripper_idx

    policy_and_params_out_dict = init_policy_nets_and_set_params(
        variant, eval_env, emb_obs_keys, emb_obs_keys_to_dim_map,
        policy_film_emb_dim_list, latent_fc_dim, num_tasks)
    policy = policy_and_params_out_dict['policy']
    policy.gripper_idx = eval_env.env.gripper_idx
    observation_keys = policy_and_params_out_dict['observation_keys']
    cnn_params = policy_and_params_out_dict['cnn_params']
    action_dim = policy_and_params_out_dict['action_dim']
    other_nets = policy_and_params_out_dict['other_nets']
    buffer_policy = policy_and_params_out_dict['buffer_policy']

    buffer_task_idxs = train_task_indices
    print("buffer_task_idxs", buffer_task_idxs)

    # Update max_num_demos_per_task
    # Only used for hdf5 formats since npy format has already been truncated
    assert ((variant['num_train_demos_per_task'] is None) or
            (variant['num_train_target_demos_per_task'] is None))
    max_num_demos_per_task = (
        variant['num_train_demos_per_task'] or
        variant['num_train_target_demos_per_task'])

    # Init and fill Replay Buffer
    sep_gripper_action = bool(
        variant['policy_params']['gripper_policy_arch'] == "sep_head")
    internal_keys = []
    replay_buffer = exp_utils.init_filled_buffer(
        buffer_datas, variant, max_replay_buffer_size, eval_env,
        buffer_task_idxs, observation_keys, internal_keys,
        num_tasks, train_embeddings_map,
        gripper_idx=gripper_idx,
        success_only=True, ext=variant['buffer_ext_dict']['train'],
        max_num_demos_per_task=max_num_demos_per_task,
        sep_gripper_action=sep_gripper_action,
        gripper_smear_ds_actions=(
            variant['trainer_kwargs']['gripper_smear_ds_actions']))

    if variant['task_embedding'] in ["demo", "demo_lang", "mcil"]:
        replay_buffer.video_encoder = video_encoder
        replay_buffer_positive = replay_buffer

    target_buffer = None
    if len(variant['target_buffers']) > 0:
        target_buffer_datas = buffer_data_dict['target_buffer_datas']

        max_target_buffer_size = get_buffer_size_multitask(
            target_buffer_datas, success_only=True,
            ext=variant['buffer_ext_dict']['target'])

        variant['target_buffer_task_idxs'] = eval_task_indices
        internal_keys = []
        print("target_buffer_task_idxs", variant['target_buffer_task_idxs'])
        assert video_encoder
        assert isinstance(eval_env, VideoTargetEmbeddingWrapper)
        target_buffer = exp_utils.init_filled_buffer(
            target_buffer_datas, variant, max_target_buffer_size, eval_env,
            variant['target_buffer_task_idxs'], observation_keys,
            internal_keys, num_tasks, buffer_embeddings=None,
            # buffer_embeddings is not used in vid or vid+lang task emb mode.
            success_only=True,
            gripper_idx=gripper_idx,
            video_encoder=video_encoder,
            ext=variant['buffer_ext_dict']['target'],
            max_num_demos_per_task=None,
            sep_gripper_action=sep_gripper_action,
            gripper_smear_ds_actions=(
                variant['trainer_kwargs']['gripper_smear_ds_actions']))
        eval_env.set_replay_buffer(target_buffer)

        # Free the buffer_datas now that stuff is all loaded
        if (variant['buffer_ext_dict']['train'] ==
                variant['buffer_ext_dict']['target'] == "npy"):
            for _, buf_data in buffer_data_dict.items():
                del buf_data
            del buffer_data_dict

        if variant['use_cached_embs']:
            # Precompute the video embeddings to save time.
            video_encoder.create_cached_embs(
                replay_buffer_positive, target_buffer,
                train_task_indices, eval_task_indices, num_embs_per_task=64,
                max_path_length=variant['max_path_length'])

    variant['target_buffer_obj'] = target_buffer

    if variant['task_embedding'] in ["demo", "demo_lang", "mcil"]:
        variant['trainer_kwargs']['task_encoder'] = video_encoder
        variant['trainer_kwargs']['target_emb_net'] = target_emb_net
    if variant['task_embedding'] in [
            "onehot", "lang", "demo", "demo_lang", "mcil"]:
        variant['trainer_kwargs']['task_embedding_type'] = (
            variant['task_embedding'])
        variant['trainer_kwargs']['emb_obs_keys'] = emb_obs_keys

    if variant['finetune_lang_enc']:
        variant['trainer_kwargs'].update({
            "finetune_lang_enc": True,
            "lang_encoder": emb_model,
        })

    print("train_task_indices", train_task_indices)
    print("eval_task_indices", eval_task_indices)

    trainer = BCTrainer(
        env=eval_env,
        policy=policy,
        replay_buffer=replay_buffer,
        multitask=True,
        train_tasks=train_task_indices,
        **variant['trainer_kwargs']
    )

    eval_policy = MakeDeterministic(policy)

    if variant['task_embedding'] in ["demo", "demo_lang", "mcil"]:
        # Add language emb to obs_keys for evals
        observation_keys = list(observation_keys) + emb_obs_keys

    eval_path_collector = ObsDictPathCollector(
        eval_env,
        eval_policy,
        observation_keys=observation_keys,
        emb_obs_keys=emb_obs_keys,
        task_embedding_type=variant['task_embedding'],
        task_emb_input_mode=variant['task_emb_input_mode'],
        emb_key_to_concat=emb_key_to_concat,
        aux_tasks=variant['aux_tasks'],
    )

    algo_kwargs = {}
    algo_kwargs['train_target_task_indices'] = (
        variant['train_target_task_indices'])

    if variant['task_embedding'] in ["demo", "demo_lang", "mcil"]:
        algo_kwargs.update(dict(
            replay_buffer_positive=replay_buffer_positive,
            buffer_embeddings=train_embeddings_map,
            multiple_strs_per_task=multiple_strs_per_task,
            # eval_embeddings=eval_embeddings,
            task_embedding=variant['task_embedding'],
            video_batch_size=variant['video_batch_size'],
            task_emb_noise_std=variant['task_emb_noise_std'],
        ))

    assert args.save_video_freq % args.eval_epoch_freq == 0
    algo_kwargs.update(dict(eval_epoch_freq=args.eval_epoch_freq,))

    variant['algo_kwargs'] = algo_kwargs

    algorithm = TorchBatchRLAlgorithm(
        trainer=trainer,
        evaluation_env=eval_env,
        evaluation_data_collector=eval_path_collector,
        replay_buffer=replay_buffer,
        max_path_length=variant['max_path_length'],
        batch_size=variant['batch_size'],
        meta_batch_size=variant['meta_batch_size'],
        num_epochs=variant['num_epochs'],
        num_eval_steps_per_epoch=variant['num_eval_steps_per_epoch'],
        num_trains_per_train_loop=variant['num_trains_per_train_loop'],
        multi_task=True,
        train_tasks=train_task_indices,
        eval_tasks=eval_task_indices,
        train_task_sample_probs=variant['train_task_sample_probs'],
        log_keys_to_remove=['evaluation/env'],
        **algo_kwargs,
    )

    video_func = VideoSaveFunctionBullet(variant)

    post_train_funcs_to_add = []
    post_train_funcs_to_add.append(video_func)

    if ((variant['task_embedding'] in ["demo", "demo_lang"])
            and (len(variant['target_buffers']) > 0)):
        plot_func = PlotSaveFunctionBullet(variant, eval_env)
        post_train_funcs_to_add.append(plot_func)

    algorithm.post_train_funcs.extend(post_train_funcs_to_add)

    algorithm.to(ptu.device)
    algorithm.train()


def enable_gpus(gpu_str):
    if gpu_str != "":
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_str
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Main Args
    parser.add_argument(
        "--env", type=str, required=True)
    parser.add_argument(
        "--num-tasks", type=int, default=300)
    parser.add_argument(
        "--eval-task-idx-intervals", nargs="+", type=str, default=[],
        help=("Must be in format such as 0-2, 5-9. This means that we "
              "collect data on indices [0, 1, 2, 5, 6, 7, 8, 9]."))
    parser.add_argument(
        "--train-task-idx-intervals", nargs="+", type=str, default=[])
    parser.add_argument(
        "--train-target-task-idx-intervals", nargs="+", type=str, default=[])
    parser.add_argument(
        "--buffers", type=str, nargs="+", default=[])
    parser.add_argument(
        "--train-target-buffers", type=str, nargs="+", default=[],
        help=("Optional buffer that overlaps with some/all of target-buffer "
              "task-idxs. Meant for cases where we eval on the train tasks."))
    parser.add_argument(
        "--num-train-target-demos-per-task", type=int, default=None)
    parser.add_argument(
        "--num-train-demos-per-task", type=int, default=None)
    parser.add_argument(
        "--target-buffers", type=str, nargs="+", default=[],
        help=("Optional buffer containing tasks we want to eval on. "
              "This is the source of the target-task demonstration for "
              "1-shot generalization eval."))

    # Below args only used if args.buffers have ext == hdf5
    parser.add_argument(
        "--hdf5-cache-mode", type=str, default="low_dim",
        choices=["all", "low_dim"])
    parser.add_argument(
        "--hdf5-subseq-len", type=int, default=30, choices=[1, 20, 30, 80],
        help=("In general, should match the max-path-len of the rollouts "
              "so that trajectories are loaded entirely with one disk access "
              "instead of needing to load one transition from disk at a time."))

    # Finetuning args
    parser.add_argument(
        "--policy-ckpt", type=str, default=None)

    # Env args
    parser.add_argument(
        "--image-dim", type=int, default=48)
    parser.add_argument(
        "--eval-image-dim", type=int, default=96)
    parser.add_argument(
        "--distractor-obj-hard-mode-prob", type=float, default=0.0)
    parser.add_argument(
        "--random-target-obj-referent", action="store_true", default=False)

    # If we want to sample some "focus" tasks more than others during training
    parser.add_argument(
        "--focus-train-task-idx-intervals", nargs="+", type=str, default=[])
    parser.add_argument(
        "--focus-train-tasks-sample-prob", type=float, default=None)

    # Policy net args
    parser.add_argument(
        "--policy-cnn-type",
        choices=['plain', 'r3m', 'resnet18', 'resnet34', 'plain_standalone'],
        default="resnet18")
    parser.add_argument(
        "--freeze-policy-cnn", action="store_true", default=False)
    parser.add_argument(
        "--policy-batchnorm", action="store_true", default=False)
    parser.add_argument(
        "--policy-lr", type=float, default=3E-4)
    parser.add_argument(
        "--policy-use-spatial-softmax", action="store_true", default=False)
    parser.add_argument(
        "--policy-std", type=float, default=None)
    parser.add_argument(
        "--policy-resnet-conv-strides", type=str, default="1,1,1,1,1")
    parser.add_argument(
        "--policy-resnet-block-sizes", type=str, default=None)
    parser.add_argument(
        "--policy-cnn-output-stage", type=str, default="",
        choices=["", "last_activations", "conv_channels", "layer1", "layer2"])
    parser.add_argument(
        "--policy-film-attn-mode", type=str, default="none",
        choices=["none", "embs"])
    parser.add_argument(
        "--policy-film-attn-hidden-sizes", nargs="+", type=int, default=[])
    parser.add_argument(
        "--policy-num-film-inputs", type=int, default=0)
    parser.add_argument(
        "--policy-film-input-order", type=str, default="",
        choices=["", "vl", "lv"],
        help=("Only used if args.policy_num_film_inputs == 2. "
              "vl = video, then lang. lv = lang, then video."))
    parser.add_argument(
        "--policy-cnn-num-channels", default=None, type=str)
    parser.add_argument(
        "--bc-policy-loss-type", default=None, type=str,
        choices=[None, "logp", "mse"])
    parser.add_argument(
        "--gripper-policy-arch", default="sep_head", type=str,
        choices=["ac_dim", "sep_head"])
    parser.add_argument(
        "--gripper-loss-weight", default=None, type=float,
        help=("Makes gripper-loss term have weight "
              "bc_weight (1.0 by default) * gripper_loss_weight."))
    parser.add_argument(
        "--gripper-loss-type", default=None, type=str,
        choices=[None, "ce", "mse"])
    parser.add_argument(
        "--gripper-smear-ds-actions", action="store_true", default=False,
        help=("Change dataset so that it always outputs -1's "
              "for closing gripper and 1's for opening gripper. "
              "Empirically, this does not help performance."))

    # Image augmentation (policy and task encoder)
    parser.add_argument(
        "--task-enc-im-augs", nargs="+", default=["pad_crop"],
        help=("Choose a subset of the following: "
              "pad_crop, bright_contr, rgb_shift, erase."))
    parser.add_argument(
        "--policy-im-augs", nargs="+", default=["pad_crop"],
        help=("Choose a subset of the following: "
              "pad_crop, bright_contr, rgb_shift, erase."))
    parser.add_argument(
        "--rnd-erase-prob", type=float, default=0.0,
        help=("Only used when 'erase' is one of the im-augs "
              "chosen in either of the previous two args."))

    # --Aux tasks (subset of Policy net args)
    parser.add_argument(
        "--policy-aux-tasks", nargs="+", type=str, default=[],
        choices=["target_container_pos_xy", "target_object_pos_xy"],
        help=("Add additional prediction heads for auxiliary tasks. "
              "Using the auxiliary predictions for these tasks as additional "
              "network inputs did not empirically help performance."))
    parser.add_argument(
        "--aux-task-weight", type=float, default=0.0,
        help="Should be non-zero if previous arg is used.")
    parser.add_argument(
        "--aux-to-feed-policy-fc", type=str, default="none",
        choices=["none", "preds", "ground_truths"],
        help=("Whether to feed the auxiliary predictions "
              "or the ground truths to the policy's "
              "fully connected layers. Not surprisingly, "
              "feeding the ground truths (cheating) makes "
              "success rates a lot better."))

    # Task Embedding and Task Encoder stuff
    parser.add_argument(
        '--task-embedding',
        choices=['onehot', 'none', 'lang', 'demo', 'demo_lang', 'mcil'],
        default='lang')
    parser.add_argument(
        "--task-emb-input-mode", type=str, default="concat_to_img_embs",
        choices=[
            "concat_to_img_embs", "film", "film_video_concat_lang",
            "film_lang_concat_video"],
        help="Used in ablations (task conditioning architecture)")
    parser.add_argument(
        "--vid-enc-num-film-inputs", type=int, default=0)
    parser.add_argument(
        "--film-hidden-sizes", nargs="+", type=int, default=[])
    parser.add_argument(
        "--film-hidden-activation", type=str, default="identity",
        choices=list(HIDDEN_ACTIVATION_STR_TO_FN_MAP.keys()))
    parser.add_argument(
        "--task-emb-noise-std", type=float, default=0.0,
        help="Only supported for demo and demo_lang task-embedding types")
    parser.add_argument(
        '--task-encoder-weight', type=float, default=0.0,
        help="Loss weight for task encoder.")
    parser.add_argument(
        "--task-encoder-loss-type",
        choices=list(VIDEO_ENCODER_LOSS_TYPE_TO_FN_MAP.keys()), default=None)
    parser.add_argument(
        "--task-encoder-contr-temp", default=None, type=float,
        help=("The temperature (beta) in the paper's "
              "task encoder loss equation."))
    parser.add_argument(
        "--latent-out-dim-size", type=int, default=768,
        help=("Dimension of the task embedding. "
              "Lang and Demo embeddings should match "
              "for contrastive learning."))
    parser.add_argument(
        "--transform-targets", action='store_true', default=False,
        help=("Whether to add trainable FC layers "
              "after the pretrained language embedding."))
    parser.add_argument(
        "--vid-enc-start-frame-range", default=None, type=str,
        help=("Specify this (such as '0,5' without the quotes) "
              "if we want the demo to start at a random frame "
              "within this specified timestep range. "
              "If not specified, we default to taking "
              "the first frame in the trajectory."))
    parser.add_argument(
        "--vid-enc-final-frame-offset-range", default=None, type=str,
        help=("Specify this (such as '25,30' without the quotes) "
              "if we want the demo to end at a random frame within this "
              "specified timestep range. "
              "If not specified, we default to "
              "taking the last frame in the trajectory."))
    parser.add_argument(
        "--vid-enc-cnn-type",
        choices=['none', 'plain', 'resnet18', 'r3m', 'clip'], default="none")
    parser.add_argument(
        "--vid-enc-mosaic-rc", default="1,2", type=str,
        help=("The number of rows and columns of images for the demo format. "
              "This is labeled as (m, n) in the paper."))
    parser.add_argument(
        "--vid-enc-cnn-num-channels", default="16,16,16", type=str)
    parser.add_argument(
        "--vid-enc-cnn-kernel-sizes", default="3,3,3", type=str)
    parser.add_argument(
        "--vid-enc-cnn-strides", default="1,1,1", type=str)
    parser.add_argument(
        "--clip-ckpt", type=str, default=None)
    parser.add_argument(
        "--clip-tokenize-scheme", type=str, default="clip")
    parser.add_argument(
        "--freeze-clip", action="store_true", default=False)
    parser.add_argument(
        "--use-cached-embs", action="store_true", default=False,
        help=("Using this flag assumes that clip is kept frozen, "
              "and that the video and lang embs "
              "can be computed beforehand. Saves lots of training time."))

    # Language Encoder stuff
    parser.add_argument(
        "--lang-emb-model-type",
        choices=["distilbert", "clip", "distilroberta", "minilm"],
        default=None)
    parser.add_argument(
        "--finetune-lang-enc", action="store_true", default=False,
        help=("Allow language encoder to be trained jointly "
              "with the rest of the policy and task encoder nets."))

    parser.add_argument(
        "--max-path-len", type=int, default=30,
        help=("Number of timesteps to run the evaluation rollouts for. "
              "Should match the trajectory lengths of the buffer."))
    parser.add_argument(
        "--batch-size", type=int, default=64,
        help="Number of transitions, per task, to train on.")
    parser.add_argument(
        "--meta-batch-size", type=int, default=8,
        help=("Number of tasks to train on in each iteration. "
              "Each task will have batch-size number of transitions, "
              "for a total of batch_size * "
              "meta_batch_size transitions in each iteration."))
    parser.add_argument(
        "--video-batch-size", type=int, default=64,
        help=("If using demo or demo_lang task embedding type, "
              "how many demos per task to draw for each batch. "
              "This is set to a factor of batch-size because loading demos "
              "from the buffer can be time-consuming."))
    parser.add_argument(
        "--eval-video-batch-size", type=int, default=1,
        help=("Number of demos to average demo embeddings over "
              "during evaluation. Set to 1 for 1-shot learning in the paper."))
    parser.add_argument(
        "--gpu", default='0', type=str)
    parser.add_argument(
        "--save-video-freq", type=int, default=100,
        help="Saves videos every save_video_freq number of epochs.")
    parser.add_argument(
        "--save-checkpoint-freq", type=int, default=1000,
        help=("Saves policy checkpoints every "
              "save_checkpoint_freq number of epochs."))
    parser.add_argument(
        "--eval-epoch-freq", type=int, default=10,
        help=("Collects eval rollouts every eval_epoch_freq number of epochs. "
              "Note this is very time consuming especially "
              "with >100 eval tasks."))
    parser.add_argument(
        "--save-checkpoint-earliest-epoch", type=int, default=0)
    parser.add_argument(
        "--seed", default=0, type=int)
    parser.add_argument(
        "--debug", action='store_true', default=False)
    args = parser.parse_args()

    if args.max_path_len != 30 and args.hdf5_subseq_len == 30:
        args.hdf5_subseq_len = args.max_path_len

    IMAGE_SIZE = (args.image_dim, args.image_dim, 3)  # (h, w, c)

    if args.eval_image_dim is not None:
        assert args.eval_image_dim > args.image_dim, (
            "No need to specify eval_image_dim if it equals image_dim")

    if args.vid_enc_cnn_type in ["none", "plain"]:
        save_checkpoint_mode = "gap_and_last"
    elif args.vid_enc_cnn_type == "clip":
        save_checkpoint_mode = "none"
    else:
        save_checkpoint_mode = "last"

    # Set the lang encoder.
    if args.task_embedding in [
            'lang', 'demo', 'demo_lang', 'mcil']:
        if ((args.task_embedding == "lang" or args.use_cached_embs)
                and args.clip_ckpt is not None):
            args.lang_emb_model_type = "clip"
            args.latent_out_dim_size = 512
        elif args.lang_emb_model_type is None:
            args.lang_emb_model_type = "minilm"

    if args.task_encoder_loss_type in ["contrastive", "cross_ent"]:
        assert args.task_encoder_contr_temp > 0.0

    if args.use_cached_embs:
        assert args.freeze_clip

    if args.finetune_lang_enc:
        assert not args.transform_targets
        assert args.lang_emb_model_type == "distilbert"
        assert not args.use_cached_embs
        assert "film" in args.task_emb_input_mode

    if args.task_embedding == "mcil":
        assert args.task_emb_input_mode == "film"
        assert args.policy_num_film_inputs == 1
        assert args.task_encoder_loss_type is None
        assert args.task_encoder_weight == 0.0
        assert args.transform_targets
        assert not args.finetune_lang_enc

    if args.bc_policy_loss_type is None:
        # Set a default BC loss type
        args.bc_policy_loss_type = "logp"

    if args.gripper_policy_arch == "sep_head":
        if args.gripper_loss_weight is None:
            args.gripper_loss_weight = 30.0
        if args.gripper_loss_type is None:
            args.gripper_loss_type = "mse"

    vid_enc_frame_ranges = []
    args.vid_enc_mosaic_rc = exp_utils.tupify_arg(args.vid_enc_mosaic_rc)
    if args.task_embedding in ["demo", "demo_lang", "mcil"]:
        args.vid_enc_cnn_num_channels = exp_utils.tupify_arg(
            args.vid_enc_cnn_num_channels)
        args.vid_enc_cnn_kernel_sizes = exp_utils.tupify_arg(
            args.vid_enc_cnn_kernel_sizes)
        args.vid_enc_cnn_strides = exp_utils.tupify_arg(args.vid_enc_cnn_strides)
        k = np.prod(args.vid_enc_mosaic_rc)
        vid_enc_frame_ranges = exp_utils.create_vid_enc_frame_ranges(
            args.vid_enc_start_frame_range,
            args.vid_enc_final_frame_offset_range,
            args.max_path_len, k, exp_utils.tupify_arg)

    if args.policy_cnn_type in ["resnet18", "resnet34"]:
        args.policy_resnet_conv_strides = list(
            exp_utils.tupify_arg(args.policy_resnet_conv_strides))

    if args.policy_cnn_num_channels is not None:
        args.policy_cnn_num_channels = list(
            exp_utils.tupify_arg(args.policy_cnn_num_channels))

    args.vid_enc_visual_modalities = ['video']
    if args.task_emb_input_mode in [
            "film", "film_video_concat_lang", "film_lang_concat_video"]:

        allowable_policy_num_film_inputs_by_task_emb = {
            "onehot": [1],
            "lang": [1],
            "demo": [1],
            "demo_lang": [1, 2],
            "mcil": [1]
        }

        if args.vid_enc_num_film_inputs > 0:
            # Language goes into the video encoder as film emb.
            assert args.task_embedding == "demo_lang"
            assert args.vid_enc_num_film_inputs in [1]
            assert args.vid_enc_cnn_type in ["plain", "clip"]
            allowable_policy_num_film_inputs_by_task_emb["demo_lang"] = [1]

        assert (
            args.policy_num_film_inputs in
            allowable_policy_num_film_inputs_by_task_emb[args.task_embedding])

        if args.policy_num_film_inputs >= 2:
            assert (
                len(args.policy_film_input_order) ==
                args.policy_num_film_inputs)

    if args.task_emb_input_mode in [
            "film_video_concat_lang", "film_lang_concat_video"]:
        assert args.task_embedding == "demo_lang"
        assert args.policy_num_film_inputs == 1
        assert args.vid_enc_num_film_inputs == 0

    if args.policy_film_attn_mode == "embs":
        assert args.policy_num_film_inputs > 1

    if args.aux_to_feed_policy_fc == "ground_truths":
        assert args.aux_task_weight == 0.0
    elif len(args.policy_aux_tasks) > 0:
        assert args.aux_to_feed_policy_fc in ["none", "preds"]
        assert args.aux_task_weight > 0.0

    if args.aux_to_feed_policy_fc != "none":
        assert len(args.policy_aux_tasks) > 0

    assert len(args.eval_task_idx_intervals) > 0

    if len(args.focus_train_task_idx_intervals) > 0:
        assert 0.0 < args.focus_train_tasks_sample_prob < 1.0

    if args.random_target_obj_referent:
        assert args.task_embedding in ['demo_lang', 'demo']

    trainer_kwargs = dict(
        policy_lr=args.policy_lr,
        policy_weight_decay=1e-4,
        bc_weight=1.0,
        task_encoder_weight=args.task_encoder_weight,
        video_batch_size=args.video_batch_size,
        aux_task_weight=args.aux_task_weight,
        gripper_loss_weight=args.gripper_loss_weight,
        gripper_loss_type=args.gripper_loss_type,
        gripper_smear_ds_actions=args.gripper_smear_ds_actions,
        bc_batch_size=args.batch_size,
        meta_batch_size=args.meta_batch_size,
        task_emb_input_mode=args.task_emb_input_mode,
        policy_num_film_inputs=args.policy_num_film_inputs,
        policy_film_input_order=args.policy_film_input_order,
        film_hidden_sizes=args.film_hidden_sizes,
        film_hidden_activation=args.film_hidden_activation,
        policy_loss_type=args.bc_policy_loss_type,
    )

    if args.debug:
        num_eval_steps_per_epoch = 0
    elif args.num_tasks <= 16:
        num_eval_steps_per_epoch = 4 * args.max_path_len
    else:
        num_eval_steps_per_epoch = 2 * args.max_path_len

    if args.debug:
        num_trains_per_train_loop = 10
    else:
        num_trains_per_train_loop = 1000

    buffer_data_dict, task_indices_dict, buffer_ext_dict = (
        load_buffer_data_dict(args))

    focus_train_task_indices = (
        exp_utils.create_task_indices_from_task_int_str_list(
            args.focus_train_task_idx_intervals, args.num_tasks))
    train_task_sample_probs = exp_utils.set_train_task_sample_probs(
        task_indices_dict['train'], focus_train_task_indices,
        args.focus_train_tasks_sample_prob)

    variant = dict(
        algorithm="BC-Pixel",

        num_epochs=3000,
        batch_size=args.batch_size,
        video_batch_size=args.video_batch_size,
        eval_video_batch_size=args.eval_video_batch_size,
        meta_batch_size=args.meta_batch_size,
        max_path_length=args.max_path_len,
        num_trains_per_train_loop=num_trains_per_train_loop,
        # num_eval_steps_per_epoch=0,
        image_dim=args.image_dim,
        eval_image_dim=args.eval_image_dim,
        num_eval_steps_per_epoch=num_eval_steps_per_epoch,

        dump_video_kwargs=dict(
            save_video_period=args.save_video_freq,
        ),

        plot_kwargs=dict(
            plot_period=50,
        ),

        env=args.env,
        image_size=IMAGE_SIZE,
        num_tasks=args.num_tasks,
        train_task_indices=task_indices_dict['train'],
        train_target_task_indices=task_indices_dict['train_target'],
        eval_task_indices=task_indices_dict['eval'],
        focus_train_task_indices=focus_train_task_indices,
        focus_train_tasks_sample_prob=args.focus_train_tasks_sample_prob,
        train_task_sample_probs=train_task_sample_probs,
        distractor_obj_hard_mode_prob=args.distractor_obj_hard_mode_prob,
        random_target_obj_referent=args.random_target_obj_referent,
        buffers=args.buffers,
        train_target_buffers=args.train_target_buffers,
        target_buffers=args.target_buffers,
        buffer_ext_dict=buffer_ext_dict,
        num_train_target_demos_per_task=args.num_train_target_demos_per_task,
        num_train_demos_per_task=args.num_train_demos_per_task,
        policy_ckpt=args.policy_ckpt,
        use_robot_state=True,
        policy_cnn_type=args.policy_cnn_type,
        aux_tasks=args.policy_aux_tasks,
        aux_to_feed_policy_fc=args.aux_to_feed_policy_fc,
        task_embedding=args.task_embedding,
        task_emb_input_mode=args.task_emb_input_mode,
        policy_film_attn_mode=args.policy_film_attn_mode,
        policy_film_attn_hidden_sizes=args.policy_film_attn_hidden_sizes,
        task_emb_noise_std=args.task_emb_noise_std,
        task_encoder_loss_type=args.task_encoder_loss_type,
        task_encoder_contr_temp=args.task_encoder_contr_temp,
        latent_out_dim_size=args.latent_out_dim_size,
        l2_unit_normalize_target_embs=bool(
            args.task_encoder_loss_type in
            ["cosine_dist", "contrastive", "cross_ent"]),
        transform_targets=args.transform_targets,
        lang_emb_model_type=args.lang_emb_model_type,
        finetune_lang_enc=args.finetune_lang_enc,
        vid_enc_frame_ranges=vid_enc_frame_ranges,
        vid_enc_cnn_type=args.vid_enc_cnn_type,
        vid_enc_mosaic_rc=args.vid_enc_mosaic_rc,
        vid_enc_visual_modalities=args.vid_enc_visual_modalities,
        vid_enc_num_film_inputs=args.vid_enc_num_film_inputs,
        freeze_clip=args.freeze_clip,
        use_cached_embs=args.use_cached_embs,
        clip_ckpt=args.clip_ckpt,
        clip_tokenize_scheme=args.clip_tokenize_scheme,
        save_checkpoint_freq=args.save_checkpoint_freq,
        save_checkpoint_earliest_epoch=args.save_checkpoint_earliest_epoch,
        save_checkpoint_mode=save_checkpoint_mode,
        seed=args.seed,
        trainer_kwargs=trainer_kwargs,
    )

    if "hdf5" in variant['buffer_ext_dict'].values():
        variant['hdf5_cache_mode'] = args.hdf5_cache_mode
        variant['hdf5_subseq_len'] = args.hdf5_subseq_len
    else:
        variant['hdf5_cache_mode'] = None
        variant['hdf5_subseq_len'] = None

    if variant['policy_ckpt'] is not None:
        assert os.path.exists(variant['policy_ckpt'])
    variant['finetuning'] = variant['policy_ckpt'] is not None

    variant['policy_params'] = dict(
        max_log_std=0,
        min_log_std=-6,
        obs_dim=None,
        std_architecture="values",
        freeze_policy_cnn=args.freeze_policy_cnn,
        lang_emb_obs_is_tokens=args.finetune_lang_enc,
        gripper_policy_arch=args.gripper_policy_arch,
        gripper_loss_type=args.gripper_loss_type,
    )

    if args.policy_cnn_num_channels is None:
        # Use default values
        if args.policy_cnn_type in ["plain", "plain_standalone"]:
            args.policy_cnn_num_channels = [16, 16, 16]
        elif args.policy_cnn_type in ["resnet18", "resnet34"]:
            args.policy_cnn_num_channels = [16, 32, 64, 128]
            # Original resnet18 settings: [64, 128, 256, 512]
        else:
            raise NotImplementedError

    if args.policy_cnn_type in ["resnet18", "resnet34"]:
        if args.policy_resnet_block_sizes is None:
            # Use default values
            if variant['policy_cnn_type'] == "resnet18":
                args.policy_resnet_block_sizes = [2, 2, 2, 2]
                # original resnet18 defaults
            elif variant['policy_cnn_type'] == "resnet34":
                args.policy_resnet_block_sizes = [3, 4, 6, 3]
            else:
                raise ValueError
        else:
            args.policy_resnet_block_sizes = list(
                exp_utils.tupify_arg(args.policy_resnet_block_sizes))
            assert len(args.policy_resnet_block_sizes) == 4

    PLAIN_CNN_PARAMS = dict(
        input_width=IMAGE_SIZE[0],
        input_height=IMAGE_SIZE[1],
        input_channels=IMAGE_SIZE[2],
        kernel_sizes=[3, 3, 3],
        n_channels=args.policy_cnn_num_channels,
        strides=[1, 1, 1],
        hidden_sizes=[1024, 512, 256],
        paddings=[1, 1, 1],
        pool_type='max2d',
        pool_sizes=[2, 2, 1],  # the one at the end means no pool
        pool_strides=[2, 2, 1],
        pool_paddings=[0, 0, 0],
        image_augmentation_padding=4,
        rnd_erase_prob=args.rnd_erase_prob,
    )

    if args.policy_cnn_type == "plain":
        variant['cnn_params'] = PLAIN_CNN_PARAMS
    elif args.policy_cnn_type in ["r3m", "resnet18", "resnet34"]:
        args.policy_use_spatial_softmax = True
        args.policy_cnn_output_stage = "conv_channels"

        variant['cnn_params'] = dict()

        if args.policy_cnn_type in ["resnet18", "resnet34"]:
            variant['cnn_params'].update(dict(
                conv_strides=args.policy_resnet_conv_strides,
                layers=args.policy_resnet_block_sizes,
                num_channels=args.policy_cnn_num_channels,
                maxpool_stride=2,
                fc_layers=[],
            ))

        variant['policy_params'].update(
            aug_transforms=args.policy_im_augs,
            image_augmentation_padding=(
                PLAIN_CNN_PARAMS['image_augmentation_padding']),
            image_size=IMAGE_SIZE,
            batchnorm=args.policy_batchnorm,
            hidden_sizes=PLAIN_CNN_PARAMS['hidden_sizes'],

            std=args.policy_std,
            output_activation=identity,

            cnn_output_stage=args.policy_cnn_output_stage,
            use_spatial_softmax=args.policy_use_spatial_softmax,
        )
    elif args.policy_cnn_type == "plain_standalone":
        variant['cnn_params'] = dict(PLAIN_CNN_PARAMS)
        hidden_sizes = PLAIN_CNN_PARAMS['hidden_sizes']
        variant['cnn_params'].update(
            hidden_sizes=None,
            output_size=0,  # Not using the last_fc layer
            reshape_input_on_forward=False,
        )
        variant['policy_params'].update(
            aug_transforms=args.policy_im_augs,
            image_size=IMAGE_SIZE,
            hidden_sizes=hidden_sizes,
            batchnorm=args.policy_batchnorm,
            cnn_output_stage=args.policy_cnn_output_stage,
            use_spatial_softmax=args.policy_use_spatial_softmax,
        )
    else:
        raise NotImplementedError

    if ((args.task_embedding in ["demo", "demo_lang", "mcil"])
            or (args.clip_ckpt is not None)):
        h_in = args.vid_enc_mosaic_rc[0] * IMAGE_SIZE[0]
        w_in = args.vid_enc_mosaic_rc[1] * IMAGE_SIZE[1]
        if args.vid_enc_cnn_type == "plain":
            variant['task_encoder_cnn_params'] = dict(PLAIN_CNN_PARAMS)
            variant['task_encoder_cnn_params'].update(dict(
                input_width=w_in,
                input_height=h_in,
                output_size=variant['latent_out_dim_size'],
                aug_transforms=args.task_enc_im_augs,
                reshape_input_on_forward=False,
                output_activation=l2_unit_normalize,
                n_channels=list(args.vid_enc_cnn_num_channels),
                num_film_inputs=args.vid_enc_num_film_inputs,
                film_hidden_sizes=args.film_hidden_sizes,
                film_hidden_activation=args.film_hidden_activation,
                kernel_sizes=list(args.vid_enc_cnn_kernel_sizes),
                strides=list(args.vid_enc_cnn_strides),
            ))
            variant['task_encoder_cnn_params']['paddings'] = [
                (ksz - 1) // 2
                for ksz in variant['task_encoder_cnn_params']['kernel_sizes']]
        elif args.vid_enc_cnn_type == "resnet18":
            pass
        elif args.vid_enc_cnn_type == "r3m":
            variant['task_encoder_cnn_params'] = dict(PLAIN_CNN_PARAMS)
            variant['task_encoder_cnn_params'].update(dict(
                input_width=w_in,
                input_height=h_in,
                output_size=variant['latent_out_dim_size'],
                hidden_sizes=[256],
                freeze_cnn=True,
                aug_transforms=args.task_enc_im_augs,
                output_activation=l2_unit_normalize,
                reshape_input_on_forward=False,
            ))
        elif args.vid_enc_cnn_type == "clip":
            if args.vid_enc_cnn_type == "clip":
                assert h_in == w_in, "CLIP must process square images."
                assert args.task_encoder_contr_temp == 1.0, (
                        "CLIP uses logit_scale, not hard-coded temp")
            variant['task_encoder_cnn_params'] = dict(
                clip_checkpoint=args.clip_ckpt,
                freeze=args.freeze_clip,
                image_shape=(h_in, w_in),
                aug_transforms=args.task_enc_im_augs,
                image_augmentation_padding=(
                    PLAIN_CNN_PARAMS['image_augmentation_padding']),
                tokenize_scheme=variant['clip_tokenize_scheme'],
            )
        else:
            assert (
                args.lang_emb_model_type == "clip" and
                args.task_embedding == "lang")

        if args.lang_emb_model_type == "clip":
            variant['clip_lang_encoder_params'] = dict(
                clip_checkpoint=args.clip_ckpt,
                freeze=args.freeze_clip,
                image_shape=None,
                aug_transforms=[],
                image_augmentation_padding=0,
                tokenize_scheme=variant['clip_tokenize_scheme'],
            )

        if args.clip_ckpt is not None:
            assert args.vid_enc_cnn_type == "clip" or variant['lang_emb_model_type'] == "clip"

        if args.vid_enc_cnn_type == "r3m":
            assert (variant['task_encoder_cnn_params']['input_height']
                    == variant['task_encoder_cnn_params']['input_width']), (
                    "Image should be square if using R3M as video encoder.")

    enable_gpus(args.gpu)
    ptu.set_gpu_mode(True)
    experiment(variant, buffer_data_dict)
