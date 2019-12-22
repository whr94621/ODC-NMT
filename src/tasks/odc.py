# MIT License

# Copyright (c) 2018 the NJUNMT-pytorch authors.

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

import collections
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
import itertools
import os
import random
import re
import time
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from tensorboardX import SummaryWriter
from tqdm import tqdm

import src.distributed as dist
from src.data.data_iterator import DataIterator
from src.data.dataset import TextLineDataset, ZipDataset
from src.data.vocabulary import Vocabulary
from src.decoding import beam_search, ensemble_beam_search
from src.metric.bleu_scorer import SacreBLEUScorer
from src.models import build_model
from src.modules.criterions import NMTCriterion
from src.optim import Optimizer
from src.optim.lr_scheduler import ReduceOnPlateauScheduler, NoamScheduler
from src.utils.common_utils import *
from src.utils.configs import default_odc_configs, pretty_configs
from src.utils.logging import *
from src.utils.moving_average import MovingAverage, TwoPhaseExponentialMovingAverage

# try:
#     import horovod.torch as hvd
#     from src.utils import distributed
# except:
#     hvd = None
#     distributed = None

BOS = Vocabulary.BOS
EOS = Vocabulary.EOS
PAD = Vocabulary.PAD


def average_checkpoints(checkpoints_path: List[str]):
    ave_state_dict = collections.OrderedDict()
    param_names = None

    for ii, f in enumerate(checkpoints_path):
        state_dict = load_model_parameters(f, map_location=None)

        if param_names is None:
            param_names = list(state_dict.keys())

        if param_names != list(state_dict.keys()):
            raise KeyError(
                "Checkpoint {0} has inconsistent parameters".format(f)
            )

        for k in param_names:
            if k not in ave_state_dict:
                ave_state_dict[k] = state_dict[k]
            else:
                ave_state_dict[k] = (ave_state_dict[k] * ii + state_dict[k]) / float(ii + 1)

    return ave_state_dict


def ave_best_k_pattern(ave_best_k):
    pattern = re.compile("ave_best_\d+")

    if pattern.match(ave_best_k) is not None:
        ave_best_k = [int(d) for d in ave_best_k.split('_') if d.isdigit()]
        if len(ave_best_k) > 1:
            raise ValueError("Value of ave_best_k should be only a single integer.")
        else:
            return ave_best_k[0]
    else:
        raise ValueError("Value of 'teacher_choice' should be 'ave_best_k'.")


def set_seed(seed):
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    random.seed(seed)

    np.random.seed(seed)

    torch.backends.cudnn.deterministic = True


def combine_from_all_shards(all_gathered_output):
    """Combine all_gathered output split by ```split_shards_iterator```
    """
    output = []
    for items in itertools.zip_longest(*all_gathered_output, fillvalue=None):
        for item in items:
            if item is not None:
                output.append(item)

    return output


def load_model_parameters(path, map_location="cpu"):
    state_dict = torch.load(path, map_location=map_location)

    if "model" in state_dict:
        return state_dict["model"]
    return state_dict


def prepare_data(seqs_x, seqs_y=None, cuda=False, batch_first=True):
    """
    Args:
        eval ('bool'): indicator for eval/infer.

    Returns:

    """

    def _np_pad_batch_2D(samples, pad, batch_first=True, cuda=True):

        batch_size = len(samples)

        sizes = [len(s) for s in samples]
        max_size = max(sizes)

        x_np = np.full((batch_size, max_size), fill_value=pad, dtype='int64')

        for ii in range(batch_size):
            x_np[ii, :sizes[ii]] = samples[ii]

        if batch_first is False:
            x_np = np.transpose(x_np, [1, 0])

        x = torch.tensor(x_np)

        if cuda is True:
            x = x.cuda()
        return x

    seqs_x = list(map(lambda s: [BOS] + s + [EOS], seqs_x))
    x = _np_pad_batch_2D(samples=seqs_x, pad=PAD,
                         cuda=cuda, batch_first=batch_first)

    if seqs_y is None:
        return x

    seqs_y = list(map(lambda s: [BOS] + s + [EOS], seqs_y))
    y = _np_pad_batch_2D(seqs_y, pad=PAD,
                         cuda=cuda, batch_first=batch_first)

    return x, y


def compute_forward(model,
                    teacher_model,
                    critic,
                    seqs_x,
                    seqs_y,
                    eval=False,
                    normalization=1.0,
                    norm_by_words=False,
                    add_kd_loss=False,
                    kd_factor=0.5,
                    hint_loss_type="kl",
                    combine_hint_loss_type="interpolation"
                    ):
    """
    Args:
        model (nn.Module):
        critic (NMTCriterion):
        hint_loss_type (str): Type of hint loss. Can be "kl"(knowledge distillation) or
            "mse" (default "kl).
        combine_hint_loss_type (str): Ways to combine hint loss. Can be "interpolation"
            or "addition"
    """

    def bottle(t):
        return t.view(-1, t.size(-1))

    batch_size = seqs_x.size(0)
    y_inp = seqs_y[:, :-1].contiguous()
    y_label = seqs_y[:, 1:].contiguous()

    y_mask = y_label.ne(PAD).float()
    words_norm = y_mask.sum(1)

    if not eval:
        model.train()
        critic.train()
        teacher_model.eval()

        if add_kd_loss:
            # Use teacher student loss
            total_loss = 0.0

            if hint_loss_type == "kl":

                with torch.enable_grad():
                    log_probs = model(seqs_x, y_inp)

                    # 1. compute kl_div between teacher and student
                    with torch.no_grad():
                        teacher_logprobs = teacher_model(seqs_x, y_inp).detach()
                        teacher_probs = torch.exp(teacher_logprobs)
                    ts_loss = F.kl_div(input=bottle(log_probs), target=bottle(teacher_probs), reduce=False).view(
                        batch_size, -1).sum(-1)
                    if norm_by_words:
                        ts_loss = ts_loss.div(words_norm)
                    ts_loss = ts_loss.div(normalization)
                    ts_loss = ts_loss.sum()
                    torch.autograd.backward(ts_loss * kd_factor, retain_graph=True)
                    ts_loss = ts_loss.item()
                    total_loss += ts_loss

                    # 2. compute NLL loss
                    nll_loss = critic(inputs=log_probs, labels=y_label, reduce=False, normalization=normalization)

            elif hint_loss_type == "mse":

                with torch.enable_grad():
                    enc_output, enc_mask = model.encoder(seqs_x)
                    dec_output, *_ = model.decoder(y_inp, enc_output, enc_mask)
                    logits_s = model.generator.proj(dec_output)

                    # 1. compute logits of teacher model
                    with torch.no_grad():
                        enc_output_t, enc_mask_t = teacher_model.encoder(seqs_x)
                        dec_output_t, *_ = teacher_model.decoder(y_inp, enc_output_t, enc_mask_t)
                        logits_t = teacher_model.generator.proj(dec_output_t)

                    # release memory
                    del enc_output_t
                    del enc_mask_t
                    del dec_output_t

                    hint_loss = F.mse_loss(logits_s, logits_t, reduction="none").sum(-1)  # shape: [B, L]
                    hint_loss = hint_loss * y_mask
                    hint_loss = (hint_loss / words_norm.view(-1, 1)).sum().div(normalization)
                    torch.autograd.backward(hint_loss * kd_factor, retain_graph=True)
                    total_loss += hint_loss.item()
                    del logits_t

                    # 2. compute NLL loss
                    log_probs = F.log_softmax(model.generator._pad_2d(logits_s), dim=-1)
                    nll_loss = critic(inputs=log_probs, labels=y_label, reduce=False, normalization=normalization)
            else:
                error_info = "Unknown hint_loss_type {0}".format(hint_loss_type)
                raise ValueError(error_info)

            if norm_by_words:
                nll_loss = nll_loss.div(words_norm)
            nll_loss = nll_loss.sum()

            if combine_hint_loss_type == "interpolation":
                nll_loss = nll_loss * (1.0 - kd_factor)
            elif combine_hint_loss_type == "addition":
                nll_loss = nll_loss
            else:
                raise ValueError("Unknown combine hint loss type {0}".format(combine_hint_loss_type))

            torch.autograd.backward(nll_loss)
            total_loss += nll_loss.item()
            return total_loss

        else:
            # For training
            with torch.enable_grad():
                log_probs = model(seqs_x, y_inp)
                loss = critic(inputs=log_probs, labels=y_label, reduce=False, normalization=normalization)

                if norm_by_words:
                    loss = loss.div(words_norm).sum()
                else:
                    loss = loss.sum()
            torch.autograd.backward(loss)
            return loss.item()
    else:
        model.eval()
        critic.eval()
        # For compute loss
        with torch.no_grad():
            log_probs = model(seqs_x, y_inp)
            loss = critic(inputs=log_probs, labels=y_label, normalization=normalization, reduce=False)

            if norm_by_words:
                loss = loss.div(words_norm).sum()
            else:
                loss = loss.sum()

        return loss.item()


def inference(valid_iterator,
              model,
              vocab_tgt: Vocabulary,
              batch_size,
              max_steps,
              beam_size=5,
              alpha=-1.0,
              rank=0,
              world_size=1,
              using_numbering_iterator=True,
              ):
    model.eval()
    trans_in_all_beams = [[] for _ in range(beam_size)]

    # assert keep_n_beams <= beam_size

    if using_numbering_iterator:
        numbers = []

    if rank == 0:
        infer_progress_bar = tqdm(total=len(valid_iterator),
                                  desc=' - (Infer)  ',
                                  unit="sents")
    else:
        infer_progress_bar = None

    valid_iter = valid_iterator.build_generator(batch_size=batch_size)

    for batch in valid_iter:

        seq_numbers = batch[0]

        if using_numbering_iterator:
            numbers += seq_numbers

        seqs_x = batch[1]

        if infer_progress_bar is not None:
            infer_progress_bar.update(len(seqs_x) * world_size)

        x = prepare_data(seqs_x, cuda=GlobalNames.USE_GPU)

        with torch.no_grad():
            word_ids = beam_search(nmt_model=model, beam_size=beam_size, max_steps=max_steps, src_seqs=x, alpha=alpha)

        word_ids = word_ids.cpu().numpy().tolist()

        # Append result
        for sent_t in word_ids:
            for ii, sent_ in enumerate(sent_t):
                sent_ = [vocab_tgt.id2token(wid) for wid in sent_ if wid != EOS and wid != PAD]
                if len(sent_) > 0:
                    trans_in_all_beams[ii].append(vocab_tgt.tokenizer.detokenize(sent_))
                else:
                    trans_in_all_beams[ii].append('%s' % vocab_tgt.id2token(EOS))

    if infer_progress_bar is not None:
        infer_progress_bar.close()

    if world_size > 1:

        if using_numbering_iterator:
            numbers = combine_from_all_shards(dist.all_gather_py_with_shared_fs(numbers))

        trans_in_all_beams = [combine_from_all_shards(dist.all_gather_py_with_shared_fs(trans)) for trans in
                              trans_in_all_beams]

    if using_numbering_iterator:
        origin_order = np.argsort(numbers).tolist()

        trans_in_all_beams = [[trans[ii] for ii in origin_order] for trans in trans_in_all_beams]

    return trans_in_all_beams


def ensemble_inference(valid_iterator,
                       models,
                       vocab_tgt: Vocabulary,
                       batch_size,
                       max_steps,
                       beam_size=5,
                       alpha=-1.0,
                       rank=0,
                       world_size=1,
                       using_numbering_iterator=True
                       ):
    for model in models:
        model.eval()

    trans_in_all_beams = [[] for _ in range(beam_size)]

    # assert keep_n_beams <= beam_size

    if using_numbering_iterator:
        numbers = []

    if rank == 0:
        infer_progress_bar = tqdm(total=len(valid_iterator),
                                  desc=' - (Infer)  ',
                                  unit="sents")
    else:
        infer_progress_bar = None

    valid_iter = valid_iterator.build_generator(batch_size=batch_size)

    for batch in valid_iter:

        seq_numbers = batch[0]

        if using_numbering_iterator:
            numbers += seq_numbers

        seqs_x = batch[1]

        if infer_progress_bar is not None:
            infer_progress_bar.update(len(seqs_x) * world_size)

        x = prepare_data(seqs_x, cuda=GlobalNames.USE_GPU)

        with torch.no_grad():
            word_ids = ensemble_beam_search(
                nmt_models=models,
                beam_size=beam_size,
                max_steps=max_steps,
                src_seqs=x,
                alpha=alpha
            )

        word_ids = word_ids.cpu().numpy().tolist()

        # Append result
        for sent_t in word_ids:
            for ii, sent_ in enumerate(sent_t):
                sent_ = [vocab_tgt.id2token(wid) for wid in sent_ if wid != EOS and wid != PAD]
                if len(sent_) > 0:
                    trans_in_all_beams[ii].append(vocab_tgt.tokenizer.detokenize(sent_))
                else:
                    trans_in_all_beams[ii].append('%s' % vocab_tgt.id2token(EOS))

    if infer_progress_bar is not None:
        infer_progress_bar.close()

    if world_size > 1:
        if using_numbering_iterator:
            numbers = dist.all_gather_py_with_shared_fs(numbers)

        trans_in_all_beams = [combine_from_all_shards(trans) for trans in trans_in_all_beams]

    if using_numbering_iterator:
        origin_order = np.argsort(numbers).tolist()
        trans_in_all_beams = [[trans[ii] for ii in origin_order] for trans in trans_in_all_beams]

    return trans_in_all_beams


def loss_validation(model, critic, valid_iterator, norm_by_words=False, rank=0, world_size=1):
    """
    :type model: Transformer

    :type critic: NMTCriterion

    :type valid_iterator: DataIterator
    """

    n_sents = 0
    n_tokens = 0.0

    sum_loss = 0.0

    valid_iter = valid_iterator.build_generator()

    for batch in valid_iter:
        _, seqs_x, seqs_y = batch

        n_sents += len(seqs_x)
        n_tokens += sum(len(s) for s in seqs_y)

        x, y = prepare_data(seqs_x, seqs_y, cuda=GlobalNames.USE_GPU)

        loss = compute_forward(model=model,
                               teacher_model=None,
                               critic=critic,
                               seqs_x=x,
                               seqs_y=y,
                               eval=True, add_kd_loss=False, norm_by_words=norm_by_words)

        if np.isnan(loss):
            WARN("NaN detected!")

        sum_loss += float(loss)

    if world_size > 1:
        sum_loss = dist.all_reduce_py(sum_loss)
        n_sents = dist.all_reduce_py(n_sents)

    return float(sum_loss / n_sents)


def bleu_evaluation(uidx,
                    valid_iterator,
                    model,
                    bleu_scorer,
                    vocab_tgt,
                    batch_size,
                    valid_dir="./valid",
                    max_steps=10,
                    beam_size=5,
                    alpha=-1.0,
                    rank=0,
                    world_size=1,
                    using_numbering_iterator=True
                    ):
    translations_in_all_beams = inference(
        valid_iterator=valid_iterator,
        model=model,
        vocab_tgt=vocab_tgt,
        batch_size=batch_size,
        max_steps=max_steps,
        beam_size=beam_size,
        alpha=alpha,
        rank=rank,
        world_size=world_size,
        using_numbering_iterator=using_numbering_iterator
    )

    if rank == 0:
        if not os.path.exists(valid_dir):
            os.mkdir(valid_dir)

        hyp_path = os.path.join(valid_dir, 'trans.iter{0}.txt'.format(uidx))

        with open(hyp_path, 'w') as f:
            for line in translations_in_all_beams[0]:
                f.write('%s\n' % line)
        with open(hyp_path) as f:
            bleu_v = bleu_scorer.corpus_bleu(f)
    else:
        bleu_v = 0.0

    if world_size > 1:
        bleu_v = dist.broadcast_py(bleu_v, root_rank=0)

    return bleu_v


def load_pretrained_model(nmt_model, pretrain_path, device, exclude_prefix=None):
    """
    Args:
        nmt_model: model.
        pretrain_path ('str'): path to pretrained model.
        map_dict ('dict'): mapping specific parameter names to those names
            in current model.
        exclude_prefix ('dict'): excluding parameters with specific names
            for pretraining.

    Raises:
        ValueError: Size not match, parameter name not match or others.

    """
    if exclude_prefix is None:
        exclude_prefix = []
    if pretrain_path != "":
        INFO("Loading pretrained model from {}".format(pretrain_path))
        pretrain_params = torch.load(pretrain_path, map_location=device)
        for name, params in pretrain_params.items():
            flag = False
            for pp in exclude_prefix:
                if name.startswith(pp):
                    flag = True
                    break
            if flag:
                continue
            INFO("Loading param: {}...".format(name))
            try:
                nmt_model.load_state_dict({name: params}, strict=False)
            except Exception as e:
                WARN("{}: {}".format(str(Exception), e))

        INFO("Pretrained model loaded.")


def train(flags):
    """
    flags:
        saveto: str
        reload: store_true
        config_path: str
        pretrain_path: str, default=""
        model_name: str
        log_path: str
    """

    # ================================================================================== #
    # Initialization for training on different devices
    # - CPU/GPU
    # - Single/Distributed
    GlobalNames.USE_GPU = flags.use_gpu

    # ================================================================================== #
    # Initialization for training on different devices
    # - CPU/GPU
    # - Single/Distributed
    GlobalNames.USE_GPU = flags.use_gpu

    if flags.multi_gpu:
        dist.distributed_init(flags.shared_dir)
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        local_rank = dist.get_local_rank()
    else:
        world_size = 1
        rank = 0
        local_rank = 0

    if GlobalNames.USE_GPU:
        torch.cuda.set_device(local_rank)
        CURRENT_DEVICE = "cuda:{0}".format(local_rank)
    else:
        CURRENT_DEVICE = "cpu"

    # If not root_rank, close logging
    # else write log of training to file.
    if rank == 0:
        write_log_to_file(os.path.join(flags.log_path, "%s.log" % time.strftime("%Y%m%d-%H%M%S")))
    else:
        close_logging()

    # ================================================================================== #
    # Parsing configuration files

    config_path = os.path.abspath(flags.config_path)
    with open(config_path.strip()) as f:
        configs = yaml.load(f)

    INFO(pretty_configs(configs))

    # Add default configs
    configs = default_odc_configs(configs)

    data_configs = configs['data_configs']
    model_configs = configs['model_configs']
    optimizer_configs = configs['optimizer_configs']
    training_configs = configs['training_configs']

    if training_configs['teacher_choice'] == "ma":
        assert training_configs['moving_average_method'] is not None, \
            "Using MA as teacher need choose one kind of moving-average method!"
    elif training_configs['teacher_choice'] != "best":
        # check whether "teacher_choice" match pattern "ave_best_k"
        ave_best_k = ave_best_k_pattern(training_configs['teacher_choice'])
        assert training_configs['num_kept_best_k_checkpoints'] >= ave_best_k, \
            "When using ave_best_k as teacher, we should at least keep k best checkpoints"
    else:
        pass

    GlobalNames.SEED = training_configs['seed']

    set_seed(GlobalNames.SEED)

    timer = Timer()

    # ================================================================================== #
    # Load Data

    INFO('Loading data...')
    timer.tic()

    # Generate target dictionary
    vocab_src = Vocabulary(**data_configs["vocabularies"][0])
    vocab_tgt = Vocabulary(**data_configs["vocabularies"][1])

    actual_buffer_size = training_configs["buffer_size"] * max(1, training_configs["update_cycle"])

    train_bitext_dataset = ZipDataset(
        TextLineDataset(data_path=data_configs['train_data'][0],
                        vocabulary=vocab_src,
                        max_len=data_configs['max_len'][0],
                        ),
        TextLineDataset(data_path=data_configs['train_data'][1],
                        vocabulary=vocab_tgt,
                        max_len=data_configs['max_len'][1],
                        )
    )

    valid_bitext_dataset = ZipDataset(
        TextLineDataset(data_path=data_configs['valid_data'][0],
                        vocabulary=vocab_src,
                        ),
        TextLineDataset(data_path=data_configs['valid_data'][1],
                        vocabulary=vocab_tgt,
                        )
    )

    training_iterator = DataIterator(dataset=train_bitext_dataset,
                                     batch_size=training_configs["batch_size"],
                                     use_bucket=training_configs['use_bucket'],
                                     buffer_size=actual_buffer_size,
                                     batching_func=training_configs['batching_key'],
                                     world_size=world_size,
                                     rank=rank)

    valid_iterator = DataIterator(dataset=valid_bitext_dataset,
                                  batch_size=training_configs['valid_batch_size'],
                                  use_bucket=True, buffer_size=100000, numbering=True,
                                  world_size=world_size, rank=rank)

    bleu_scorer = SacreBLEUScorer(reference_path=data_configs["bleu_valid_reference"],
                                  num_refs=data_configs["num_refs"],
                                  lang_pair=data_configs["lang_pair"],
                                  sacrebleu_args=training_configs["bleu_valid_configs"]['sacrebleu_args'],
                                  postprocess=training_configs["bleu_valid_configs"]['postprocess']
                                  )

    INFO('Done. Elapsed time {0}'.format(timer.toc()))

    lrate = optimizer_configs['learning_rate']
    is_early_stop = False

    # ================================ Begin ======================================== #
    # Build Model & Optimizer
    # We would do steps below on after another
    #     1. build models & criterion
    #     2. move models & criterion to gpu if needed
    #     3. load pre-trained model if needed
    #     4. build optimizer
    #     5. build learning rate scheduler if needed
    #     6. load checkpoints if needed

    # 0. Initial
    model_collections = Collections()
    best_model_prefix = os.path.join(flags.saveto, flags.model_name + GlobalNames.MY_BEST_MODEL_SUFFIX)

    teacher_model_path = os.path.join(flags.saveto, flags.model_name + ".teacher.tpz")

    checkpoint_saver = Saver(save_prefix="{0}.ckpt".format(os.path.join(flags.saveto, flags.model_name)),
                             num_max_keeping=training_configs['num_kept_checkpoints']
                             )

    last_k_saver = LastKSaver(
        save_prefix="{0}.last_k_ckpt".format(os.path.join(flags.saveto, flags.model_name)),
        num_max_keeping=training_configs['num_kept_last_k_checkpoints']
    )

    best_k_saver = BestKSaver(
        save_prefix="{0}.best_k_ckpt".format(os.path.join(flags.saveto, flags.model_name)),
        num_max_keeping=training_configs['num_kept_best_k_checkpoints']
    )

    # 1. Build Model & Criterion
    INFO('Building teacher model...')

    teacher_model = build_model(n_src_vocab=vocab_src.max_n_words,
                                n_tgt_vocab=vocab_tgt.max_n_words,
                                **model_configs)

    INFO('Done.')

    INFO('Building model...')
    timer.tic()
    nmt_model = build_model(n_src_vocab=vocab_src.max_n_words,
                            n_tgt_vocab=vocab_tgt.max_n_words, **model_configs)
    INFO(nmt_model)

    critic = NMTCriterion(label_smoothing=model_configs['label_smoothing'])

    INFO(critic)
    INFO('Done. Elapsed time {0}'.format(timer.toc()))

    # 2. Move to GPU
    if GlobalNames.USE_GPU:
        nmt_model = nmt_model.cuda()
        critic = critic.cuda()
        teacher_model = teacher_model.cuda()

    # 3. Load pretrained model if needed
    load_pretrained_model(nmt_model, flags.pretrain_path, exclude_prefix=None, device=CURRENT_DEVICE)

    # 4. Build optimizer
    INFO('Building Optimizer...')

    if not flags.multi_gpu:
        optim = Optimizer(name=optimizer_configs['optimizer'],
                          model=nmt_model,
                          lr=lrate,
                          grad_clip=optimizer_configs['grad_clip'],
                          optim_args=optimizer_configs['optimizer_params'],
                          update_cycle=training_configs['update_cycle']
                          )
    else:
        optim = dist.DistributedOptimizer(name=optimizer_configs['optimizer'],
                                          model=nmt_model,
                                          lr=lrate,
                                          grad_clip=optimizer_configs['grad_clip'],
                                          optim_args=optimizer_configs['optimizer_params'],
                                          device_id=local_rank
                                          )

    # 5. Build scheduler for optimizer if needed
    if optimizer_configs['schedule_method'] is not None:

        if optimizer_configs['schedule_method'] == "loss":

            scheduler = ReduceOnPlateauScheduler(optimizer=optim,
                                                 **optimizer_configs["scheduler_configs"]
                                                 )

        elif optimizer_configs['schedule_method'] == "noam":
            scheduler = NoamScheduler(optimizer=optim, **optimizer_configs['scheduler_configs'])
        else:
            WARN("Unknown scheduler name {0}. Do not use lr_scheduling.".format(optimizer_configs['schedule_method']))
            scheduler = None
    else:
        scheduler = None

    # 6. build moving average
    if training_configs['moving_average_method'] == "two_phase_ema":
        ma = TwoPhaseExponentialMovingAverage(
            named_params=nmt_model.named_parameters(),
            alpha1=training_configs['moving_average_alpha'],
            alpha2=training_configs['moving_average_alpha_2'],
            warmup_steps=training_configs['moving_average_warmup']
        )
    elif training_configs['moving_average_method'] is not None:
        ma = MovingAverage(moving_average_method=training_configs['moving_average_method'],
                           named_params=nmt_model.named_parameters(),
                           alpha=training_configs['moving_average_alpha'])
    else:
        ma = None

    INFO('Done. Elapsed time {0}'.format(timer.toc()))

    # Reload from latest checkpoint
    if flags.reload:
        checkpoint_saver.load_latest(model=nmt_model, optim=optim, lr_scheduler=scheduler,
                                     collections=model_collections, ma=ma, device=CURRENT_DEVICE)

    if os.path.exists(teacher_model_path):
        teacher_model.load_state_dict(torch.load(teacher_model_path, map_location=CURRENT_DEVICE), strict=False)

    # broadcast parameters and optimizer states
    if world_size > 1:
        INFO("Broadcasting model parameters...")
        dist.broadcast_parameters(params=nmt_model.state_dict())
        INFO("Broadcasting optimizer states...")
        dist.broadcast_optimizer_state(optimizer=optim.optim)
        INFO('Done.')

    # ================================================================================== #
    # Prepare training

    eidx = model_collections.get_collection("eidx", [0])[-1]
    uidx = model_collections.get_collection("uidx", [1])[-1]
    bad_count = model_collections.get_collection("bad_count", [0])[-1]
    oom_count = model_collections.get_collection("oom_count", [0])[-1]
    add_kd_loss = model_collections.get_collection("add_kd_loss", [False])[-1]
    teacher_patience = model_collections.get_collection("teacher_patience", [training_configs['teacher_patience']])[-1]
    cum_n_samples = 0
    cum_n_words = 0
    best_valid_loss = 1.0 * 1e10  # Max Float
    update_cycle = training_configs['update_cycle']
    grad_denom = 0

    if rank == 0:
        summary_writer = SummaryWriter(log_dir=flags.log_path)
    else:
        summary_writer = None

    # Timer for computing speed
    timer_for_speed = Timer()
    timer_for_speed.tic()

    INFO('Begin training...')

    while True:

        if summary_writer is not None:
            summary_writer.add_scalar("Epoch", (eidx + 1), uidx)

        # Build iterator and progress bar
        training_iter = training_iterator.build_generator()

        if rank == 0:
            training_progress_bar = tqdm(desc='  - (Epoch %d)   ' % eidx,
                                         total=len(training_iterator),
                                         unit="sents"
                                         )
        else:
            training_progress_bar = None

        for batch in training_iter:

            seqs_x, seqs_y = batch

            batch_size = len(seqs_x)

            cum_n_samples += batch_size
            cum_n_words += sum(len(s) for s in seqs_y)

            try:
                # Prepare data
                x, y = prepare_data(seqs_x, seqs_y, cuda=GlobalNames.USE_GPU)

                loss = compute_forward(model=nmt_model,
                                       teacher_model=teacher_model,
                                       critic=critic,
                                       seqs_x=x,
                                       seqs_y=y,
                                       eval=False,
                                       normalization=1.0,
                                       norm_by_words=training_configs["norm_by_words"],
                                       add_kd_loss=add_kd_loss,
                                       kd_factor=training_configs["kd_factor"],
                                       hint_loss_type=training_configs['hint_loss_type'],
                                       combine_hint_loss_type=training_configs["combine_hint_loss_type"],
                                       )

                update_cycle -= 1
                grad_denom += batch_size

            except RuntimeError as e:
                if 'out of memory' in str(e):
                    print('| WARNING: ran out of memory, skipping batch')
                    oom_count += 1
                else:
                    raise e

            # When update_cycle becomes 0, it means end of one batch. Several things will be done:
            # - update parameters
            # - reset update_cycle and grad_denom
            # - update uidx
            # - update moving average

            if update_cycle == 0:

                # 0. reduce variables
                if world_size > 1:
                    grad_denom = dist.all_reduce_py(grad_denom)
                    cum_n_words = dist.all_reduce_py(cum_n_words)

                optim.step(denom=grad_denom)
                optim.zero_grad()

                if training_progress_bar is not None:
                    training_progress_bar.update(grad_denom)

                update_cycle = training_configs['update_cycle']
                grad_denom = 0

                uidx += 1

                if scheduler is None:
                    pass
                elif optimizer_configs["schedule_method"] == "loss":
                    scheduler.step(metric=best_valid_loss)
                else:
                    scheduler.step(global_step=uidx)

                if ma is not None and eidx >= training_configs['moving_average_start_epoch']:
                    ma.step()
            else:
                continue

            # ================================================================================== #
            # Display some information
            if should_trigger_by_steps(uidx, eidx, every_n_step=training_configs['disp_freq']):

                if world_size > 1:
                    cum_n_words = dist.all_reduce_py(cum_n_words)
                    cum_n_samples = dist.all_reduce_py(cum_n_samples)

                # words per second and sents per second
                words_per_sec = cum_n_words / (timer.toc(return_seconds=True))
                sents_per_sec = cum_n_samples / (timer.toc(return_seconds=True))
                lrate = list(optim.get_lrate())[0]

                if summary_writer is not None:
                    summary_writer.add_scalar("Speed(words/sec)", scalar_value=words_per_sec, global_step=uidx)
                    summary_writer.add_scalar("Speed(sents/sen)", scalar_value=sents_per_sec, global_step=uidx)
                    summary_writer.add_scalar("lrate", scalar_value=lrate, global_step=uidx)
                    summary_writer.add_scalar("oom_count", scalar_value=oom_count, global_step=uidx)

                # Reset timer
                timer.tic()
                cum_n_words = 0
                cum_n_samples = 0

            # ================================================================================== #
            # Loss Validation & Learning rate annealing
            if should_trigger_by_steps(global_step=uidx, n_epoch=eidx, every_n_step=training_configs['loss_valid_freq'],
                                       debug=flags.debug):

                valid_loss = loss_validation(model=nmt_model,
                                             critic=critic,
                                             valid_iterator=valid_iterator,
                                             rank=rank,
                                             world_size=world_size
                                             )

                model_collections.add_to_collection("history_losses", valid_loss)

                min_history_loss = np.array(model_collections.get_collection("history_losses")).min()

                best_valid_loss = min_history_loss

                if summary_writer is not None:
                    summary_writer.add_scalar("loss", valid_loss, global_step=uidx)
                    summary_writer.add_scalar("best_loss", min_history_loss, global_step=uidx)
                    # summary_writer.add_scalar("add_kd_loss", 1 if add_kd_loss else 0, uidx)

            # ================================================================================== #
            # BLEU Validation & Early Stop

            if should_trigger_by_steps(global_step=uidx, n_epoch=eidx,
                                       every_n_step=training_configs['bleu_valid_freq'],
                                       min_step=training_configs['bleu_valid_warmup'],
                                       debug=flags.debug):

                valid_bleu = bleu_evaluation(uidx=uidx,
                                             valid_iterator=valid_iterator,
                                             batch_size=training_configs["bleu_valid_batch_size"],
                                             model=nmt_model,
                                             bleu_scorer=bleu_scorer,
                                             vocab_tgt=vocab_tgt,
                                             valid_dir=flags.valid_path,
                                             max_steps=training_configs["bleu_valid_configs"]["max_steps"],
                                             beam_size=training_configs["bleu_valid_configs"]["beam_size"],
                                             alpha=training_configs["bleu_valid_configs"]["alpha"],
                                             world_size=world_size,
                                             rank=rank,
                                             )

                model_collections.add_to_collection(key="history_bleus", value=valid_bleu)

                best_valid_bleu = float(np.array(model_collections.get_collection("history_bleus")).max())

                if summary_writer is not None:
                    summary_writer.add_scalar("bleu", valid_bleu, uidx)
                    summary_writer.add_scalar("best_bleu", best_valid_bleu, uidx)

                # If model get new best valid bleu score
                if valid_bleu >= best_valid_bleu:
                    bad_count = 0

                    if is_early_stop is False:
                        if rank == 0:
                            # 1. save the best model
                            torch.save(nmt_model.state_dict(), best_model_prefix + ".final")

                            # 2. Save last-k-checkpoints from scratch
                            last_k_saver.clean_all_checkpoints()

                else:
                    bad_count += 1

                    # At least one epoch should be traversed
                    if bad_count >= training_configs['early_stop_patience'] and eidx > 0:
                        is_early_stop = True
                        WARN("Early Stop!")

                if not is_early_stop:
                    # If not early stop
                    # save best checkpoint

                    if rank == 0:
                        last_k_saver.save(global_step=uidx,
                                          model=nmt_model,
                                          optim=optim,
                                          lr_scheduler=scheduler,
                                          collections=model_collections,
                                          ma=ma)

                        best_k_saver.save(global_step=uidx,
                                          metric=valid_bleu,
                                          model=nmt_model,
                                          optim=optim,
                                          lr_scheduler=scheduler,
                                          collections=model_collections,
                                          ma=ma)

                if summary_writer is not None:
                    summary_writer.add_scalar("bad_count", bad_count, uidx)

                # ODC-BLEU
                if not is_early_stop:
                    if valid_bleu >= best_valid_bleu:

                        # choose method to generate teachers from checkpoints
                        # - best
                        # - ave_k_best
                        # - ma
                        if training_configs['teacher_choice'] == "ma":
                            teacher_params = ma.export_ma_params()
                        elif training_configs['teacher_choice'] == "best":
                            teacher_params = nmt_model.state_dict()
                        else:
                            if best_k_saver.num_saved >= ave_best_k:
                                teacher_params = average_checkpoints(best_k_saver.get_all_ckpt_path()[-ave_best_k:])
                            else:
                                teacher_params = nmt_model.state_dict()

                        torch.save(teacher_params, teacher_model_path)
                        teacher_patience = 0
                        del teacher_params
                        add_kd_loss = False

                        if flags.debug:
                            add_kd_loss = not add_kd_loss

                    else:
                        if eidx >= training_configs["teacher_refresh_warmup"]:
                            teacher_patience += 1

                            # start to as kd_loss
                            if teacher_patience >= training_configs["teacher_patience"]:
                                teacher_params = torch.load(teacher_model_path, map_location=CURRENT_DEVICE)
                                teacher_model.load_state_dict(teacher_params, strict=False)
                                add_kd_loss = True
                                del teacher_params

                if summary_writer is not None:
                    summary_writer.add_scalar("add_kd_loss", 1 if add_kd_loss else 0, uidx)

                INFO("{0} Loss: {1:.2f} BLEU: {2:.2f} lrate: {3:6f} patience: {4} add_kd_loss: {5}".format(
                    uidx, valid_loss, valid_bleu, lrate, bad_count, 1 if add_kd_loss else 0
                ))

            # ================================================================================== #
            # Saving checkpoints
            if should_trigger_by_steps(uidx, eidx, every_n_step=training_configs['save_freq'], debug=flags.debug):
                model_collections.add_to_collection("uidx", uidx)
                model_collections.add_to_collection("eidx", eidx)
                model_collections.add_to_collection("bad_count", bad_count)
                model_collections.add_to_collection("teacher_patience", teacher_patience)
                model_collections.add_to_collection("add_kd_loss", add_kd_loss)

                if not is_early_stop:
                    if rank == 0:
                        checkpoint_saver.save(global_step=uidx,
                                              model=nmt_model,
                                              optim=optim,
                                              lr_scheduler=scheduler,
                                              collections=model_collections,
                                              ma=ma)

        if training_progress_bar is not None:
            training_progress_bar.close()

        eidx += 1
        if eidx > training_configs["max_epochs"]:
            break


def translate(flags):
    GlobalNames.USE_GPU = flags.use_gpu

    if flags.multi_gpu:
        dist.distributed_init(flags.shared_dir)
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        local_rank = dist.get_local_rank()
        torch.cuda.set_device(local_rank)
    else:
        world_size = 1
        rank = 0
        local_rank = 0

    if rank != 0:
        close_logging()

    config_path = os.path.abspath(flags.config_path)

    with open(config_path.strip()) as f:
        configs = yaml.load(f)

    data_configs = configs['data_configs']
    model_configs = configs['model_configs']

    timer = Timer()
    # ================================================================================== #
    # Load Data

    INFO('Loading data...')
    timer.tic()

    # Generate target dictionary
    vocab_src = Vocabulary(**data_configs["vocabularies"][0])
    vocab_tgt = Vocabulary(**data_configs["vocabularies"][1])

    valid_dataset = TextLineDataset(data_path=flags.source_path,
                                    vocabulary=vocab_src)

    valid_iterator = DataIterator(dataset=valid_dataset,
                                  batch_size=flags.batch_size,
                                  use_bucket=True,
                                  buffer_size=100000,
                                  numbering=True,
                                  world_size=world_size,
                                  rank=rank
                                  )

    INFO('Done. Elapsed time {0}'.format(timer.toc()))

    # ================================================================================== #
    # Build Model & Sampler & Validation
    INFO('Building model...')
    timer.tic()
    nmt_model = build_model(n_src_vocab=vocab_src.max_n_words,
                            n_tgt_vocab=vocab_tgt.max_n_words, **model_configs)
    nmt_model.eval()
    INFO('Done. Elapsed time {0}'.format(timer.toc()))

    INFO('Reloading model parameters...')
    timer.tic()

    params = load_model_parameters(flags.model_path, map_location="cpu")

    nmt_model.load_state_dict(params)

    if GlobalNames.USE_GPU:
        nmt_model.cuda()

    INFO('Done. Elapsed time {0}'.format(timer.toc()))

    INFO('Begin...')
    timer.tic()

    translations_in_all_beams = inference(
        valid_iterator=valid_iterator,
        model=nmt_model,
        vocab_tgt=vocab_tgt,
        batch_size=flags.batch_size,
        max_steps=flags.max_steps,
        beam_size=flags.beam_size,
        alpha=flags.alpha,
        rank=rank,
        world_size=world_size,
        using_numbering_iterator=True
    )

    acc_time = timer.toc(return_seconds=True)
    acc_num_tokens = sum([len(line.strip().split()) for line in translations_in_all_beams[0]])

    INFO('Done. Speed: {0:.2f} words/sec'.format(acc_num_tokens / acc_time))

    if rank == 0:
        keep_n = flags.beam_size if flags.keep_n <= 0 else min(flags.beam_size, flags.keep_n)
        outputs = ['%s.%d' % (flags.saveto, i) for i in range(keep_n)]

        with batch_open(outputs, 'w') as handles:
            for ii in range(keep_n):
                for trans in translations_in_all_beams[ii]:
                    handles[ii].write('%s\n' % trans)


def ensemble_translate(flags):
    GlobalNames.USE_GPU = flags.use_gpu

    if flags.multi_gpu:
        dist.distributed_init(flags.shared_dir)
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        local_rank = dist.get_local_rank()
        torch.cuda.set_device(local_rank)
    else:
        world_size = 1
        rank = 0
        local_rank = 0

    if rank != 0:
        close_logging()

    config_path = os.path.abspath(flags.config_path)

    with open(config_path.strip()) as f:
        configs = yaml.load(f)

    data_configs = configs['data_configs']
    model_configs = configs['model_configs']

    timer = Timer()
    # ================================================================================== #
    # Load Data

    INFO('Loading data...')
    timer.tic()

    # Generate target dictionary
    vocab_src = Vocabulary(**data_configs["vocabularies"][0])
    vocab_tgt = Vocabulary(**data_configs["vocabularies"][1])

    valid_dataset = TextLineDataset(data_path=flags.source_path,
                                    vocabulary=vocab_src)

    valid_iterator = DataIterator(dataset=valid_dataset,
                                  batch_size=flags.batch_size,
                                  use_bucket=True,
                                  buffer_size=100000,
                                  numbering=True,
                                  world_size=world_size,
                                  rank=rank
                                  )

    INFO('Done. Elapsed time {0}'.format(timer.toc()))

    # ================================================================================== #
    # Build Model & Sampler & Validation
    INFO('Building model...')
    timer.tic()

    nmt_models = []

    model_path = flags.model_path

    for ii in range(len(model_path)):

        nmt_model = build_model(n_src_vocab=vocab_src.max_n_words,
                                n_tgt_vocab=vocab_tgt.max_n_words, **model_configs)
        nmt_model.eval()
        INFO('Done. Elapsed time {0}'.format(timer.toc()))

        INFO('Reloading model parameters...')
        timer.tic()

        params = load_model_parameters(model_path[ii], map_location="cpu")

        nmt_model.load_state_dict(params)

        if GlobalNames.USE_GPU:
            nmt_model.cuda()

        nmt_models.append(nmt_model)

    INFO('Done. Elapsed time {0}'.format(timer.toc()))

    INFO('Begin...')
    timer.tic()

    translations_in_all_beams = ensemble_inference(
        valid_iterator=valid_iterator,
        models=nmt_models,
        vocab_tgt=vocab_tgt,
        batch_size=flags.batch_size,
        max_steps=flags.max_steps,
        beam_size=flags.beam_size,
        alpha=flags.alpha,
        rank=rank,
        world_size=world_size,
        using_numbering_iterator=True
    )

    acc_time = timer.toc(return_seconds=True)
    acc_num_tokens = sum([len(line.strip().split()) for line in translations_in_all_beams[0]])

    INFO('Done. Speed: {0:.2f} words/sec'.format(acc_num_tokens / acc_time))

    if rank == 0:
        keep_n = flags.beam_size if flags.keep_n <= 0 else min(flags.beam_size, flags.keep_n)
        outputs = ['%s.%d' % (flags.saveto, i) for i in range(keep_n)]

        with batch_open(outputs, 'w') as handles:
            for ii in range(keep_n):
                for trans in translations_in_all_beams[ii]:
                    handles[ii].write('%s\n' % trans)


def evalute_sharpness(flags):
    GlobalNames.USE_GPU = flags.use_gpu

    if flags.multi_gpu:
        dist.distributed_init(flags.shared_dir)
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        local_rank = dist.get_local_rank()
        torch.cuda.set_device(local_rank)
    else:
        world_size = 1
        rank = 0
        local_rank = 0

    if rank != 0:
        close_logging()

    config_path = os.path.abspath(flags.config_path)

    with open(config_path.strip()) as f:
        configs = yaml.load(f)

    data_configs = configs['data_configs']
    model_configs = configs['model_configs']
    training_configs = configs['training_configs']

    timer = Timer()

    # ================================================================================== #
    # Load Data

    INFO('Loading data...')
    timer.tic()

    # Generate target dictionary
    vocab_src = Vocabulary(**data_configs["vocabularies"][0])
    vocab_tgt = Vocabulary(**data_configs["vocabularies"][1])

    valid_dataset = ZipDataset(
        TextLineDataset(data_path=flags.source_path,
                        vocabulary=vocab_src,
                        ),
        TextLineDataset(data_path=flags.target_path,
                        vocabulary=vocab_tgt,
                        )
    )

    valid_iterator = DataIterator(dataset=valid_dataset,
                                  batch_size=flags.batch_size,
                                  use_bucket=True,
                                  buffer_size=100000,
                                  numbering=True,
                                  world_size=world_size,
                                  rank=rank
                                  )

    INFO('Done. Elapsed time {0}'.format(timer.toc()))
    # ================================================================================== #

    # ================================================================================== #
    # Build Model & Sampler & Validation
    INFO('Building model...')
    timer.tic()

    nmt_models = []

    model_path = flags.model_path

    for ii in range(len(model_path)):

        nmt_model = build_model(n_src_vocab=vocab_src.max_n_words,
                                n_tgt_vocab=vocab_tgt.max_n_words, **model_configs)
        nmt_model.eval()
        INFO('Done. Elapsed time {0}'.format(timer.toc()))

        INFO('Reloading model parameters...')
        timer.tic()

        params = load_model_parameters(model_path[ii], map_location="cpu")

        nmt_model.load_state_dict(params)

        if GlobalNames.USE_GPU:
            nmt_model.cuda()

        nmt_models.append(nmt_model)

    # Build a model for new model

    new_nmt_model = build_model(n_src_vocab=vocab_src.max_n_words,
                                n_tgt_vocab=vocab_tgt.max_n_words, **model_configs)
    new_nmt_model.eval()

    if GlobalNames.USE_GPU:
        new_nmt_model.cuda()

    critic = NMTCriterion(label_smoothing=0.0)

    if GlobalNames.USE_GPU:
        critic.cuda()

    INFO('Done. Elapsed time {0}'.format(timer.toc()))

    lower_bound = - 1.0
    upper_bound = 2.0
    interval = 0.02

    def gen_state_dict(model_a: nn.Module, model_b: nn.Module, alpha):
        named_params_a = collections.OrderedDict(model_a.named_parameters())
        named_params_b = collections.OrderedDict(model_b.named_parameters())

        named_params = collections.OrderedDict()
        with torch.no_grad():
            for name in named_params_a.keys():
                named_params[name] = alpha * named_params_a[name] + (1.0 - alpha) * named_params_b[name]

        return named_params

    with open("%s.csv" % flags.saveto, "w") as f:
        for i in range(1000000):

            alpha = lower_bound + i * interval

            if alpha > upper_bound:
                break

            state_dict = gen_state_dict(nmt_models[0], nmt_models[1], alpha=alpha)
            new_nmt_model.load_state_dict(state_dict=state_dict, strict=False)

            valid_loss = loss_validation(model=new_nmt_model,
                                         critic=critic,
                                         valid_iterator=valid_iterator,
                                         rank=rank,
                                         world_size=world_size,
                                         norm_by_words=True)

            print(alpha, valid_loss)
            f.write("%.4f, %.4f\n" % (alpha, valid_loss))
