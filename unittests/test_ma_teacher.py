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

import os

import unittests.test_utils as test_utils
from src.utils.logging import INFO


def test_transformer_train(test_dir, use_gpu=False, multi_gpu=False):
    from src.bin import train

    config_path = "./unittests/configs/test_ma_teacher.yaml"
    model_name = test_utils.get_model_name(config_path)

    saveto = os.path.join(test_dir, "save")
    log_path = os.path.join(test_dir, "log")
    valid_path = os.path.join(test_dir, "valid")

    train.run(model_name=model_name,
              config_path=config_path,
              saveto=saveto,
              log_path=log_path,
              valid_path=valid_path,
              debug=True,
              use_gpu=use_gpu,
              multi_gpu=multi_gpu,
              task="ocd")


def test_transformer_inference(test_dir, use_gpu=False, multi_gpu=False):
    from src.bin import translate
    from src.utils.common_utils import GlobalNames
    config_path = "./unittests/configs/test_ma_teacher.yaml"

    saveto = os.path.join(test_dir, "save")
    model_name = test_utils.get_model_name(config_path)
    model_path = os.path.join(saveto, model_name + GlobalNames.MY_BEST_MODEL_SUFFIX + ".final")
    source_path = "./unittests/data/dev/zh.0"
    batch_size = 3
    beam_size = 3

    translate.run(model_name=model_name,
                  source_path=source_path,
                  batch_size=batch_size,
                  beam_size=beam_size,
                  model_path=model_path,
                  config_path=config_path,
                  saveto=saveto,
                  max_steps=20,
                  use_gpu=use_gpu,
                  multi_gpu=multi_gpu
                  )


def test_transformer_ensemble_inference(test_dir, use_gpu=False, multi_gpu=False):
    from src.bin import ensemble_translate
    from src.utils.common_utils import GlobalNames
    config_path = "./unittests/configs/test_ocd_loss.yaml"

    saveto = os.path.join(test_dir, "save")
    model_name = test_utils.get_model_name(config_path)

    model_path = os.path.join(saveto, model_name + GlobalNames.MY_BEST_MODEL_SUFFIX + ".final")
    model_path = [model_path for _ in range(3)]

    source_path = "./unittests/data/dev/zh.0"
    batch_size = 3
    beam_size = 3

    ensemble_translate.run(model_name=model_name,
                           source_path=source_path,
                           batch_size=batch_size,
                           beam_size=beam_size,
                           model_path=model_path,
                           config_path=config_path,
                           saveto=saveto,
                           max_steps=20,
                           use_gpu=use_gpu, multi_gpu=multi_gpu)


if __name__ == '__main__':

    test_dir = "./tmp"
    parser = test_utils.build_test_argparser()
    args = parser.parse_args()

    if not os.path.exists(test_dir):
        os.makedirs(test_dir, exist_ok=True)

    INFO("=" * 20)
    INFO("Test transformer training...")
    test_transformer_train(test_dir, use_gpu=args.use_gpu, multi_gpu=args.multi_gpu)
    INFO("Done.")
    INFO("=" * 20)

    INFO("=" * 20)
    INFO("Test transformer inference...")
    test_transformer_inference(test_dir, use_gpu=args.use_gpu, multi_gpu=args.multi_gpu)
    INFO("Done.")
    INFO("=" * 20)

    # INFO("=" * 20)
    # INFO("Test ensemble inference...")
    # test_transformer_ensemble_inference(test_dir, use_gpu=args.use_gpu, multi_gpu=args.multi_gpu)
    # INFO("Done.")
    # INFO("=" * 20)

    test_utils.rm_tmp_dir(test_dir)
