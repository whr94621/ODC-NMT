# ODC-NMT

This repository is for our paper *Online Distilling from Checkpoints for Neural Machine Translation*. For basic usage of this codebase, please refer to the dev branch [NJUNMT-pytorch](https:/github.com/whr94621/NJUNMT-pytorch/tree/dev).

If you want to run experiments of ODC, please add an option in `src.bin.train`
``` bash 
python -m src.bin.train \
    ... \
    --task "odc"
```

There are some comments about the configuration of ODC (they are all in the `training_configs`):
- teacher_choice: The choice of the teacher model. There are several options
  - best: use the best checkpoint as teacher
  - ma: use the EMA (exponential moving average) as the teacher
  - ave_best_k: 'k' should be an integer. This means we use the average of best-k checkpoints.
- moving_average_method: The choice of moving average method. 
  - ema (recommended)
  - sma
  - two_phase_ema
  - none: do not use moving average
- teacher patience: Integer. This value controls the tolerance that we start to use ODC when current model is inferior to the best checkpoint. I use 1 in the paper.
- teacher_refresh_warmup: Integer. After how many epoches we start to use ODC. I use 1 in the paper.
- kd_factor: The value of factor before the knowledge distillation loss.
- combine_hint_loss_type: The way to combine NLL loss and KD loss. Let `kd_factor` be $alpha$, it can be
  - "add": nll_loss + $alpha$ * kd_loss 
  - "interpolation": (1.0 - $alpha$) * nll_loss + $alpha$ * kd_loss
- hint_loss_type: The type of knowledge distillation loss. It can be
  - kl (recommended): word-level knowledge distillation
  - mse: MSE of decoder hidden states between teacher and student.

If you have any questions, you can contact me by email whr94621@gmail.com. If my work can help you somehow, please cite 
``` bibtex
@inproceedings{wei-etal-2019-online,
    title = "Online Distilling from Checkpoints for Neural Machine Translation",
    author = "Wei, Hao-Ran  and
      Huang, Shujian  and
      Wang, Ran  and
      Dai, Xin-yu  and
      Chen, Jiajun",
    booktitle = "Proceedings of the 2019 Conference of the North {A}merican Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers)",
    month = jun,
    year = "2019",
    address = "Minneapolis, Minnesota",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/N19-1192",
    doi = "10.18653/v1/N19-1192",
    pages = "1932--1941",
}
```