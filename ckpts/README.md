---
license: mit
language:
- en
library_name: transformers
tags:
- video-generation
- robotics
- embodied-ai
- physical-reasoning
- causal-reasoning
- inverse-dynamics
- wow
- arxiv:2509.22642
datasets:
- WoW-world-model/WoW-1-Benchmark-Samples
pipeline_tag: video-generation
base_model: wan
---

# 🤖 WoW-1-Wan-14B-2M

**WoW-1-Wan-14B** is a 14-billion-parameter generative world model trained on **2 million real-world robot interaction trajectories**. It is designed to imagine, reason, and act in physically consistent environments, powered by SOPHIA-guided refinement and a co-trained **Inverse Dynamics Model**.

This model is part of the [WoW (World-Omniscient World Model)](https://github.com/wow-world-model/wow-world-model) project, introduced in the paper:

> **[WoW: Towards a World omniscient World model Through Embodied Interaction](https://arxiv.org/abs/2509.22642)**  
> *Chi et al., 2025 – arXiv:2509.22642*

## 🧠 Key Features

- **14B parameters** trained on **2M robot interaction samples**
- Learns **causal physical reasoning** from embodied action
- Generates physically consistent video and robotic action plans
- Uses **SOPHIA**, a vision-language critic, to refine outputs
- Paired with an **Inverse Dynamics Model** to complete imagination-to-action loop

## 🧪 Training Data

<!-- - Dataset: [WoW-1-Benchmark-Samples](https://huggingface.co/datasets/WoW-world-model/WoW-1-Benchmark-Samples) -->
- **2M** Real-world robot interaction trajectories
- Multimodal scenes including vision, action, and language
- Diverse **mixture captions** for better generalization
### 🧠 Mixture Caption Strategy

- **Prompt Lengths**:
  - Short: *"The Franka robot, grasp the red bottle on the table"*
  - Long: *"The scene... open the drawer, take the screwdriver, place it on the table..."*

- **Robot Model Mixing**:
  - Captions reference various robot types
  - Example: *"grasp with the Franka Panda arm"*, *"use end-effector to align"*

- **Action Granularity**:
  - Coarse: *"move to object"*
  - Fine: *"rotate wrist 30° before grasping"*


## 🔄 Continuous Updates

This dataset will be **continuously updated** with:
- More trajectories
- Richer language
- Finer multimodal annotations

## 🧩 Applications

- Zero-shot video generation in robotics
- Causal reasoning and physics simulation
- Long-horizon manipulation planning
- Forward and inverse control prediction

## 📄 Citation

```bibtex
@article{chi2025wow,
  title={WoW: Towards a World omniscient World model Through Embodied Interaction},
  author={Chi, Xiaowei and Jia, Peidong and Fan, Chun-Kai and Ju, Xiaozhu and Mi, Weishi and Qin, Zhiyuan and Zhang, Kevin and Tian, Wanxin and Ge, Kuangzhi and Li, Hao and others},
  journal={arXiv preprint arXiv:2509.22642},
  year={2025}
}
```

## 🔗 Resources

- 🧠 Project page: [wow-world-model.github.io](https://wow-world-model.github.io/)
- 💻 GitHub repo: [wow-world-model/wow-world-model](https://github.com/wow-world-model/wow-world-model)
- 📊 Dataset: [WoW-1 Benchmark Samples](https://huggingface.co/datasets/WoW-world-model/WoW-1-Benchmark-Samples)

---