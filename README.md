# MIND-V: Hierarchical Video Generation for Long-Horizon Robotic Manipulation with RL-based Physical Alignment

**Version**

Ruicheng Zhang¹\*, Mingyang Zhang²\*, Jun Zhou¹†, Zhangrui Guo³, Xiaofan Liu³,
Zunnan Xu¹, Zhizhou Zhong⁴, Puxin Yan⁴, Haocheng Luo¹, Xiu Li¹†

¹Tsinghua University ²China University of Geosciences
³Sun Yat-sen University ⁴Hong Kong University of Science and Technology

\*: Equal contribution  †: Corresponding authors


## 🌟 Introduction
🔥 **MIND-V** is a hierarchical framework that synthesizes physically plausible and logically coherent long-horizon robotic manipulation videos. Inspired by cognitive science, MIND-V bridges high-level reasoning with pixel-level synthesis through three core components: a Semantic Reasoning Hub (SRH), a Behavioral Semantic Bridge (BSB), and a Motor Video Generator (MVG). The framework employs Staged Visual Future Rollouts and RL-based Physical Foresight Coherence (PFC) optimization using V-JEPA world models for enhanced physical plausibility.

![demo_video](docs/videos/teaser.mp4)



## ⚙️ Quick Start

### 1. Environment Setup
Our environment setup is compatible with CogVideoX. You can follow their configuration to complete the setup.

```bash
conda create -n mindv python=3.10
conda activate mindv
pip install -r requirements.txt
bash setup_MIND-V_env.sh
```

Download models from [Model Download Guide](MODEL_DOWNLOAD_GUIDE.md) and place them under the base root. The checkpoints should be organized as follows:

```
├── ckpts
    ├── CogVideoX-Fun-V1.5-5b-InP   (pretrained model base)
    ├── MIND-V                      (fine-tuned transformer)
    ├── sam2                        (segmentation model)
    ├── vjepa2                      (world models)
    └── affordance-r1               (semantic reasoning model)
```

### 2. Long-Horizon Video Generation

**Diverse Object Manipulation**

```bash
python long_horizon_video_pipeline.py \
    --image "demos/diverse_ood_objs/avocado.png" \
    --instruction "pick up the avocado" \
    --output "output/diverse_objects" \
    --seed 42
```

**Complex Multi-Step Tasks**

```bash
python long_horizon_video_pipeline.py \
    --image "demos/long_video/bridge1_s1.png" \
    --instruction "First put the towel into the metal pot, then put the spoon into the metal pot" \
    --output "output/long_horizon" \
    --num_inference_steps 20 \
    --transition_frames 5 \
    --seed 42
```



### 3. Training

#### Pre-training (Injector Training)

```bash
cd scripts
bash train_injector.sh
```

We fine-tune the base model on robotic manipulation videos with a resolution of 720×1280 and 49 frames using 8 H200 GPUs. The training process includes:


#### Post-training with GRPO

```bash
python flow_grpo/scripts/mindv_post_training.py \
    --config flow_grpo/config/base.py \
    --dataset "path/to/grpo/dataset" \
    --output_dir "flow_grpo/checkpoints" \
    --main_gpu 0 \
    --reward_gpu 1 \
    --num_epochs 50
```


## 🔗 Citation

If you find this work helpful, please consider citing:

```bibtex
@article{zhang2025mindv,
  title={MIND-V: Hierarchical Video Generation for Long-Horizon Robotic Manipulation with RL-based Physical Alignment},
  author={Zhang, Ruicheng and Zhang, Mingyang and Zhou, Jun and Guo, Zhangrui and Liu, Xiaofan and Xu, Zunnan and Zhong, Zhizhou and Yan, Puxin and Luo, Haocheng and Li, Xiu},
  journal={arXiv preprint arXiv:2506.09985},
  year={2025}
}
```

### Acknowledgments

We sincerely thank the **RoboMaster** team for their pioneering work in robotic video generation. Our implementation builds upon and extends the excellent codebase from:

**https://github.com/KlingTeam/RoboMaster/tree/main**

### Additional References

- **CogVideoX**: https://github.com/THUDM/CogVideo
- **V-JEPA2**: https://github.com/facebookresearch/vjepa2
- **SAM2**: https://github.com/facebookresearch/segment-anything-2
- **Affordance-R1**: https://github.com/hq-King/Affordance-R1

---

