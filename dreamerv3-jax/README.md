# Harmony DreamerV3

This repository is an exact copy of the [official DreamerV3 repository](https://github.com/danijar/dreamerv3/tree/8fa35f83eee1ce7e10f3dee0b766587d0a713a60), with the lightweight HarmonyDream method applied to it. 
The original README can be found [here](https://github.com/danijar/dreamerv3/blob/8fa35f83eee1ce7e10f3dee0b766587d0a713a60/README.md).

## Modifications to the Original Codebase
We added the HarmonyDream method to the DreamerV3 codebase, which consists of the following changes:
- Added the `harmony` flag to the `dreamerv3/configs.yaml` file (line 108) to enable HarmonyDream.
- Implemented the `Harmonizer` class in `dreamerv3/nets.py` (line 612-622).
- Applied the Harmonizer to the DreamerV3 model in `dreamerv3/agent.py` (line 151-152, 178-192).

## Dependencies
To install JAX, run the following command:
```zsh
pip install "jax[cuda11_cudnn82]"==0.4.6 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```
Or alternatively, follow the official instructions [here](https://github.com/google/jax?tab=readme-ov-file#installation).

Other dependencies can be installed using the following command:
```zsh
pip install -r requirements.txt
```
To run experiments on Atari, additionally install the Atari env by running the following command:
```zsh
sh dreamerv3/embodied/scripts/install_atari.sh
```
Meta-world installation instructions can be found in our [Paper README](../README.md).

## Experiments
### Atari100K
```zsh
python dreamerv3/train.py --logdir {LOGDIR} --configs atari100k --seed 0 --task atari_bank_heist --harmony True
```
We made a minor modification to the Chopper Command task by setting the maximum episode length from 108000 to 5400. This is because the agent is easy to get stuck due to environmental reasons, which causes training to fail. A similar problem can occur in Breakout as well.
### Meta-world
```zsh
python dreamerv3/train.py --logdir {LOGDIR} --configs metaworld --seed 0 --task metaworld_handle_pull_side --harmony True
```

## Tips
- When running multiple experiments on a single GPU, set the `--jax.prealloc` flag to `False` to avoid running out of memory.
- We notice experiments on the Atari100K benchmark have high variance. Reproducing the exact same results can be difficult. We provide our experiment results of Harmony DreamerV3 and the reproduced DreamerV3 in the `scores` folder for reference.
- We encountered numerical instability issues when running experiments on RTX 3090 GPUs. This could be due to a JAX problem. We used a hack in the `dreamerv3/agent.py` file (line 136, 194) to solve this problem by adding an identity module.
