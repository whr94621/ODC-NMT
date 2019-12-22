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
import time

import numpy as np
import torch
import yaml
from tensorboardX import SummaryWriter
from tqdm import tqdm

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
from src.utils.configs import default_baseline_configs, pretty_configs
from src.utils.logging import *
from src.utils.moving_average import MovingAverage

try:
    import horovod.torch as hvd
    from src.utils import distributed
except:
    hvd = None
    distributed = None

BOS = Vocabulary.BOS
EOS = Vocabulary.EOS
PAD = Vocabulary.PAD


def set_seed(seed):
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    random.seed(seed)

    np.random.seed(seed)

    torch.backends.cudnn.deterministic = True


def load_model_parameters(path, map_location="cpu"):
    state_dict = torch.load(path, map_location=map_location)

    if "model" in state_dict:
        return state_dict["model"]
    return state_dict


def split_shard(*inputs, split_size=1):
    if split_size <= 1:
        yield inputs
    else:

        lengths = [len(s) for s in inputs[-1]]  #
        sorted_indices = np.argsort(lengths)

        # sorting inputs

        inputs = [
            [inp[ii] for ii in sorted_indices]
            for inp in inputs
        ]

        # split shards
        total_batch = sorted_indices.shape[0]  # total number of batches

        if split_size >= total_batch:
            yield inputs
        else:
            shard_size = total_batch // split_size

            _indices = list(range(total_batch))[::shard_size] + [total_batch]

            for beg, end in zip(_indices[:-1], _indices[1:]):
                yield (inp[beg:end] for inp in inputs)


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
                    critic,
                    seqs_x,
                    seqs_y,
                    eval=False,
                    normalization=1.0,
                    norm_by_words=False
                    ):
    """
    :type model: nn.Module

    :type critic: NMTCriterion
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
            loss = critic(inputs=log_probs, labels=y_label, normalization=normalization, reduce=True)
        return loss.item()


def loss_validation(model, critic, valid_iterator, rank=0, world_size=1):
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
                               critic=critic,
                               seqs_x=x,
                               seqs_y=y,
                               eval=True)

        if np.isnan(loss):
            WARN("NaN detected!")

        sum_loss += float(loss)

    if world_size > 1:
        sum_loss = sum(distributed.all_gather(sum_loss))
        n_sents = sum(distributed.all_gather(n_sents))

    return float(sum_loss / n_sents)


def bleu_validation(uidx,
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
                    world_size=1
                    ):
    model.eval()

    numbers = []
    trans = []

    if rank == 0:
        infer_progress_bar = tqdm(total=len(valid_iterator),
                                  desc=' - (Infer)  ',
                                  unit="sents")
    else:
        infer_progress_bar = None

    valid_iter = valid_iterator.build_generator(batch_size=batch_size)

    for batch in valid_iter:

        seq_nums = batch[0]
        numbers += seq_nums

        seqs_x = batch[1]

        if infer_progress_bar is not None:
            infer_progress_bar.update(len(seqs_x) * world_size)

        x = prepare_data(seqs_x, cuda=GlobalNames.USE_GPU)

        with torch.no_grad():
            word_ids = beam_search(nmt_model=model, beam_size=beam_size, max_steps=max_steps, src_seqs=x, alpha=alpha)

        word_ids = word_ids.cpu().numpy().tolist()

        # Append result
        for sent_t in word_ids:
            sent_t = [[wid for wid in line if wid != PAD] for line in sent_t]
            x_tokens = []

            for wid in sent_t[0]:
                if wid == EOS:
                    break
                x_tokens.append(vocab_tgt.id2token(wid))

            if len(x_tokens) > 0:
                trans.append(vocab_tgt.tokenizer.detokenize(x_tokens))
            else:
                trans.append('%s' % vocab_tgt.id2token(EOS))

    if infer_progress_bar is not None:
        infer_progress_bar.close()

    if world_size > 1:

        trans_gathered = distributed.all_gather_with_shared_fs(trans)

        trans = []

        for lines in itertools.zip_longest(*trans_gathered, fillvalue=None):
            for line in lines:
                if line is not None:
                    trans.append(line)

        numbers_gathered = distributed.all_gather_with_shared_fs(numbers)

        numbers = []

        for numbers_ in itertools.zip_longest(*numbers_gathered, fillvalue=None):
            for num in numbers_:
                if num is not None:
                    numbers.append(num)

    origin_order = np.argsort(numbers).tolist()
    trans = [trans[ii] for ii in origin_order]

    if rank == 0:
        if not os.path.exists(valid_dir):
            os.mkdir(valid_dir)

        hyp_path = os.path.join(valid_dir, 'trans.iter{0}.txt'.format(uidx))

        with open(hyp_path, 'w') as f:
            for line in trans:
                f.write('%s\n' % line)

        with open(hyp_path) as f:
            bleu_v = bleu_scorer.corpus_bleu(f)
    else:
        bleu_v = 0.0

    if world_size > 1:
        bleu_v = distributed.broadcast(bleu_v, root_rank=0)

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


def train(FLAGS):
    """
    FLAGS:
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
    GlobalNames.USE_GPU = FLAGS.use_gpu

    if FLAGS.multi_gpu:

        if hvd is None or distributed is None:
            ERROR("Distributed training is disable. Please check the installation of Horovod.")

        hvd.init()
        world_size = hvd.size()
        rank = hvd.rank()
        local_rank = hvd.local_rank()
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
    if rank != 0:
        close_logging()

    # write log of training to file.
    if rank == 0:
        write_log_to_file(os.path.join(FLAGS.log_path, "%s.log" % time.strftime("%Y%m%d-%H%M%S")))

    # ================================================================================== #
    # Parsing configuration files

    config_path = os.path.abspath(FLAGS.config_path)
    with open(config_path.strip()) as f:
        configs = yaml.load(f)

    INFO(pretty_configs(configs))

    # Add default configs
    configs = default_baseline_configs(configs)
    data_configs = configs['data_configs']
    model_configs = configs['model_configs']
    optimizer_configs = configs['optimizer_configs']
    training_configs = configs['training_configs']

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
    best_model_prefix = os.path.join(FLAGS.saveto, FLAGS.model_name + GlobalNames.MY_BEST_MODEL_SUFFIX)

    checkpoint_saver = Saver(save_prefix="{0}.ckpt".format(os.path.join(FLAGS.saveto, FLAGS.model_name)),
                             num_max_keeping=training_configs['num_kept_checkpoints']
                             )
    best_model_saver = Saver(save_prefix=best_model_prefix, num_max_keeping=training_configs['num_kept_best_model'])

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

    # 3. Load pretrained model if needed
    load_pretrained_model(nmt_model, FLAGS.pretrain_path, exclude_prefix=None, device=CURRENT_DEVICE)

    # 4. Build optimizer
    INFO('Building Optimizer...')
    optim = Optimizer(name=optimizer_configs['optimizer'],
                      model=nmt_model,
                      lr=lrate,
                      grad_clip=optimizer_configs['grad_clip'],
                      optim_args=optimizer_configs['optimizer_params'],
                      distributed=True if world_size > 1 else False,
                      update_cycle=training_configs['update_cycle']
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

    if training_configs['moving_average_method'] is not None:
        ma = MovingAverage(moving_average_method=training_configs['moving_average_method'],
                           named_params=nmt_model.named_parameters(),
                           alpha=training_configs['moving_average_alpha'])
    else:
        ma = None

    INFO('Done. Elapsed time {0}'.format(timer.toc()))

    # Reload from latest checkpoint
    if FLAGS.reload:
        checkpoint_saver.load_latest(model=nmt_model, optim=optim, lr_scheduler=scheduler,
                                     collections=model_collections, ma=ma)

    # broadcast parameters and optimizer states
    if world_size > 1:
        hvd.broadcast_parameters(params=nmt_model.state_dict(), root_rank=0)
        hvd.broadcast_optimizer_state(optimizer=optim.optim, root_rank=0)

    # ================================================================================== #
    # Prepare training

    eidx = model_collections.get_collection("eidx", [0])[-1]
    uidx = model_collections.get_collection("uidx", [1])[-1]
    bad_count = model_collections.get_collection("bad_count", [0])[-1]
    oom_count = model_collections.get_collection("oom_count", [0])[-1]
    cum_n_samples = 0
    cum_n_words = 0
    best_valid_loss = 1.0 * 1e10  # Max Float
    update_cycle = training_configs['update_cycle']
    grad_denom = 0

    if rank == 0:
        summary_writer = SummaryWriter(log_dir=FLAGS.log_path)
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
                                       critic=critic,
                                       seqs_x=x,
                                       seqs_y=y,
                                       eval=False,
                                       normalization=1.0,
                                       norm_by_words=training_configs["norm_by_words"])

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
                if world_size > 1:
                    grad_denom = distributed.all_reduce(grad_denom)

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
                    cum_n_words = sum(distributed.all_gather(cum_n_words))
                    cum_n_samples = sum(distributed.all_gather(cum_n_samples))

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
                                       debug=FLAGS.debug):

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

            # ================================================================================== #
            # BLEU Validation & Early Stop

            if should_trigger_by_steps(global_step=uidx, n_epoch=eidx,
                                       every_n_step=training_configs['bleu_valid_freq'],
                                       min_step=training_configs['bleu_valid_warmup'],
                                       debug=FLAGS.debug):

                valid_bleu = bleu_validation(uidx=uidx,
                                             valid_iterator=valid_iterator,
                                             batch_size=training_configs["bleu_valid_batch_size"],
                                             model=nmt_model,
                                             bleu_scorer=bleu_scorer,
                                             vocab_tgt=vocab_tgt,
                                             valid_dir=FLAGS.valid_path,
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

                            # 2. record all several best models
                            best_model_saver.save(global_step=uidx, model=nmt_model, ma=ma)
                else:
                    bad_count += 1

                    # At least one epoch should be traversed
                    if bad_count >= training_configs['early_stop_patience'] and eidx > 0:
                        is_early_stop = True
                        WARN("Early Stop!")

                if summary_writer is not None:
                    summary_writer.add_scalar("bad_count", bad_count, uidx)

                INFO("{0} Loss: {1:.2f} BLEU: {2:.2f} lrate: {3:6f} patience: {4}".format(
                    uidx, valid_loss, valid_bleu, lrate, bad_count
                ))

            # ================================================================================== #
            # Saving checkpoints
            if should_trigger_by_steps(uidx, eidx, every_n_step=training_configs['save_freq'], debug=FLAGS.debug):
                model_collections.add_to_collection("uidx", uidx)
                model_collections.add_to_collection("eidx", eidx)
                model_collections.add_to_collection("bad_count", bad_count)

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


def translate(FLAGS):
    GlobalNames.USE_GPU = FLAGS.use_gpu

    if FLAGS.multi_gpu:

        if hvd is None or distributed is None:
            ERROR("Distributed training is disable. Please check the installation of Horovod.")

        hvd.init()
        world_size = hvd.size()
        rank = hvd.rank()

        if GlobalNames.USE_GPU:
            torch.cuda.set_device(hvd.local_rank())
    else:
        world_size = 1
        rank = 0

    if rank != 0:
        close_logging()

    config_path = os.path.abspath(FLAGS.config_path)

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

    valid_dataset = TextLineDataset(data_path=FLAGS.source_path,
                                    vocabulary=vocab_src)

    valid_iterator = DataIterator(dataset=valid_dataset,
                                  batch_size=FLAGS.batch_size,
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

    params = load_model_parameters(FLAGS.model_path, map_location="cpu")

    nmt_model.load_state_dict(params, strict=False)

    if GlobalNames.USE_GPU:
        nmt_model.cuda()

    INFO('Done. Elapsed time {0}'.format(timer.toc()))

    INFO('Begin...')

    result_numbers = []
    result = []
    n_words = 0

    timer.tic()

    if rank == 0:
        infer_progress_bar = tqdm(total=len(valid_iterator),
                                  desc=' - (Infer)  ',
                                  unit="sents")
    else:
        infer_progress_bar = None

    valid_iter = valid_iterator.build_generator()

    for batch in valid_iter:

        numbers, seqs_x = batch

        batch_size_t = len(seqs_x)

        x = prepare_data(seqs_x=seqs_x, cuda=GlobalNames.USE_GPU)

        with torch.no_grad():
            word_ids = beam_search(nmt_model=nmt_model, beam_size=FLAGS.beam_size, max_steps=FLAGS.max_steps,
                                   src_seqs=x, alpha=FLAGS.alpha)

        word_ids = word_ids.cpu().numpy().tolist()

        # Append result
        for sent_t in word_ids:
            sent_t = [[wid for wid in line if wid != PAD] for line in sent_t]
            result.append(sent_t)

            n_words += len(sent_t[0])

        result_numbers += numbers

        if rank == 0:
            infer_progress_bar.update(batch_size_t * world_size)

    if rank == 0:
        infer_progress_bar.close()

    if FLAGS.multi_gpu:
        n_words = sum(distributed.all_gather(n_words))

    INFO('Done. Speed: {0:.2f} words/sec'.format(n_words / (timer.toc(return_seconds=True))))

    if FLAGS.multi_gpu:

        result_gathered = distributed.all_gather_with_shared_fs(result)

        result = []

        for lines in itertools.zip_longest(*result_gathered, fillvalue=None):
            for line in lines:
                if line is not None:
                    result.append(line)

        result_numbers_gathered = distributed.all_gather_with_shared_fs(result_numbers)

        result_numbers = []

        for numbers in itertools.zip_longest(*result_numbers_gathered, fillvalue=None):
            for num in numbers:
                if num is not None:
                    result_numbers.append(num)

    if rank == 0:
        translation = []
        for sent in result:
            samples = []
            for trans in sent:
                sample = []
                for w in trans:
                    if w == vocab_tgt.EOS:
                        break
                    sample.append(vocab_tgt.id2token(w))
                samples.append(vocab_tgt.tokenizer.detokenize(sample))
            translation.append(samples)

        # resume the ordering
        origin_order = np.argsort(result_numbers).tolist()
        translation = [translation[ii] for ii in origin_order]

        keep_n = FLAGS.beam_size if FLAGS.keep_n <= 0 else min(FLAGS.beam_size, FLAGS.keep_n)
        outputs = ['%s.%d' % (FLAGS.saveto, i) for i in range(keep_n)]

        with batch_open(outputs, 'w') as handles:
            for trans in translation:
                for i in range(keep_n):
                    if i < len(trans):
                        handles[i].write('%s\n' % trans[i])
                    else:
                        handles[i].write('%s\n' % 'eos')


def ensemble_translate(FLAGS):
    GlobalNames.USE_GPU = FLAGS.use_gpu

    if FLAGS.multi_gpu:

        if hvd is None or distributed is None:
            ERROR("Distributed training is disable. Please check the installation of Horovod.")

        hvd.init()
        world_size = hvd.size()
        rank = hvd.rank()

        if GlobalNames.USE_GPU:
            torch.cuda.set_device(hvd.local_rank())
    else:
        world_size = 1
        rank = 0

    if rank != 0:
        close_logging()

    config_path = os.path.abspath(FLAGS.config_path)

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

    valid_dataset = TextLineDataset(data_path=FLAGS.source_path,
                                    vocabulary=vocab_src)

    valid_iterator = DataIterator(dataset=valid_dataset,
                                  batch_size=FLAGS.batch_size,
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

    model_path = FLAGS.model_path

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
    result_numbers = []
    result = []
    n_words = 0

    timer.tic()

    if rank == 0:
        infer_progress_bar = tqdm(total=len(valid_iterator),
                                  desc=' - (Infer)  ',
                                  unit="sents")
    else:
        infer_progress_bar = None

    valid_iter = valid_iterator.build_generator()
    for batch in valid_iter:

        numbers, seqs_x = batch

        batch_size_t = len(seqs_x)

        x = prepare_data(seqs_x=seqs_x, cuda=GlobalNames.USE_GPU)

        with torch.no_grad():
            word_ids = ensemble_beam_search(nmt_models=nmt_models,
                                            beam_size=FLAGS.beam_size,
                                            max_steps=FLAGS.max_steps,
                                            src_seqs=x, alpha=FLAGS.alpha)

        word_ids = word_ids.cpu().numpy().tolist()

        # Append result
        for sent_t in word_ids:
            sent_t = [[wid for wid in line if wid != PAD] for line in sent_t]
            result.append(sent_t)

            n_words += len(sent_t[0])

        if rank == 0:
            infer_progress_bar.update(batch_size_t * world_size)

    if rank == 0:
        infer_progress_bar.close()

    if FLAGS.multi_gpu:
        n_words = sum(distributed.all_gather(n_words))

    INFO('Done. Speed: {0:.2f} words/sec'.format(n_words / (timer.toc(return_seconds=True))))

    if FLAGS.multi_gpu:

        result_gathered = distributed.all_gather_with_shared_fs(result)

        result = []

        for lines in itertools.zip_longest(*result_gathered, fillvalue=None):
            for line in lines:
                if line is not None:
                    result.append(line)

        result_numbers_gathered = distributed.all_gather_with_shared_fs(result_numbers)

        result_numbers = []

        for numbers in itertools.zip_longest(*result_numbers_gathered, fillvalue=None):
            for num in numbers:
                if num is not None:
                    result_numbers.append(num)

    if rank == 0:
        translation = []
        for sent in result:
            samples = []
            for trans in sent:
                sample = []
                for w in trans:
                    if w == vocab_tgt.EOS:
                        break
                    sample.append(vocab_tgt.id2token(w))
                samples.append(vocab_tgt.tokenizer.detokenize(sample))
            translation.append(samples)

        # resume the ordering
        origin_order = np.argsort(result_numbers).tolist()
        translation = [translation[ii] for ii in origin_order]

        keep_n = FLAGS.beam_size if FLAGS.keep_n <= 0 else min(FLAGS.beam_size, FLAGS.keep_n)
        outputs = ['%s.%d' % (FLAGS.saveto, i) for i in range(keep_n)]

        with batch_open(outputs, 'w') as handles:
            for trans in translation:
                for i in range(keep_n):
                    if i < len(trans):
                        handles[i].write('%s\n' % trans[i])
                    else:
                        handles[i].write('%s\n' % 'eos')
