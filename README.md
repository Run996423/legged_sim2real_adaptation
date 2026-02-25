<h1 align="center"> Real-Time Dynamic Parameter Estimation for Legged Robot Sim-to-Real Adaptation </h1>

<div align="center">

[[Website]](https://anonymous.4open.science/w/legged-sim2real-adaptation-20B4/)
[[Video]](https://www.youtube.com/playlist?list=PLG-5CzJKb16SnrZxh_hAvW1XI_GIh05Qk)

</div>

The sim-to-real gap remains a critical barrier preventing learning-based policies from achieving high
performance in real-world tasks. To address this challenge, we propose a real-time adaptation framework
for legged robots. Our approach explicitly estimates dynamic parameters online using real-world data,
with the policy conditioned on these estimated parameters to achieve optimal performance in current
scenarios. Extensive experiments demonstrate that our method outperforms the baseline by 27% in velocity
tracking, 84% in base orientation stability, and 39% in robustness on average. Moreover, the explicit
estimates can be directly utilized for parameter-dependent downstream tasks such as payload mass
identification, leash-guided quadruped control, and force-aware locomotion.

---

## Installation

This codebase is built following the structure of [legged_gym](https://github.com/leggedrobotics/legged_gym) codebases.

### Prerequisites

- Ubuntu 22.04 LTS (recommended)
- NVIDIA GPU with CUDA support
- Python 3.8

### Environment Setup

Create conda env:

```bash
conda create -n ls2ra python=3.8
conda activate ls2ra
```

Download [IsaacGym](https://developer.nvidia.com/isaac-gym/download) and extract:

```bash
wget https://developer.nvidia.com/isaac-gym-preview-4
tar -xvzf isaac-gym-preview-4
```

Install IsaacGym Python API:

```bash
pip install -e isaacgym/python
```

Install legged_gym:

```bash
cd path/to/legged_sim2real_adaptation
pip install -e .
```

Install rsl_rl:

```bash
cd path/to/legged_sim2real_adaptation/rsl_rl-2.3.3
pip install -e .
```

Install tensorboard:

```bash
pip install tensorboard
```

For libpython error:

- Check conda path:

```bash
conda info -e
```

- Then set LD_LIBRARY_PATH:

```bash
export LD_LIBRARY_PATH=</path/to/conda/envs/your_env>/lib:$LD_LIBRARY_PATH
```

For `np.float` deprecated error in `isaacgym/python/isaacgym/torch_utils.py`, replace `np.float` by `float`.

## Train the policy from scratch

```bash
cd legged_gym
python scripts/train.py --task=go1_adaptation
```
