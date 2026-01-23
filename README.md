<div align="center">

# MIND-V: Hierarchical Video Generation for Long-Horizon Robotic Manipulation with RL-based Physical Alignment


**Ruicheng Zhang**¹\*&nbsp;&nbsp;&nbsp;&nbsp;
**Mingyang Zhang**²\*&nbsp;&nbsp;&nbsp;&nbsp;
**Jun Zhou**¹†&nbsp;&nbsp;&nbsp;&nbsp;
Zhangrui Guo³&nbsp;&nbsp;&nbsp;&nbsp;
Xiaofan Liu³  

Zunnan Xu¹˒⁴<sup>‡</sup>&nbsp;&nbsp;&nbsp;&nbsp;
Zhizhou Zhong⁵&nbsp;&nbsp;&nbsp;&nbsp;
Puxin Yan⁵&nbsp;&nbsp;&nbsp;&nbsp;
Haocheng Luo¹˒⁶&nbsp;&nbsp;&nbsp;&nbsp;
**Xiu Li**¹†

<sup>1</sup>Tsinghua University 
<sup>2</sup>China University of Geosciences 
<sup>3</sup>Sun Yat-sen University  
<sup>4</sup>X Square Robot 
<sup>5</sup>Hong Kong University of Science and Technology 
<sup>6</sup>Central South University  

\* Equal contribution † Corresponding authors ‡ Project Lead  

<br>

[![arXiv](https://img.shields.io/badge/arXiv-2512.06628-b31b1b.svg)](https://arxiv.org/pdf/2512.06628)
[![Model](https://img.shields.io/badge/%F0%9F%A4%97_Model-MIND--V-FF6C37)](https://huggingface.co/Richard-ZZZZZ/MIND-V)


### Abstract

Embodied imitation learning is constrained by the scarcity of diverse, long-horizon robotic manipulation data. Existing video generation models for this domain are limited to synthesizing short clips of simple actions and often rely on manually defined trajectories. To this end, we introduce MIND-V, a hierarchical framework designed to synthesize physically plausible and logically coherent videos of long-horizon robotic manipulation. Inspired by cognitive science, MIND-V bridges high-level reasoning with pixel-level synthesis through three core components: a Semantic Reasoning Hub (SRH) that leverages a pre-trained vision-language model for task planning; a Behavioral Semantic Bridge (BSB) that translates abstract instructions into domain-invariant representations; and a Motor Video Generator (MVG) for conditional video rendering. MIND-V employs Staged Visual Future Rollouts, a test-time optimization strategy to enhance long-horizon robustness. To align the generated videos with physical laws, we introduce a GRPO reinforcement learning post-training phase guided by a novel Physical Foresight Coherence (PFC) reward. PFC leverages the V-JEPA world model to enforce physical plausibility by aligning the predicted and actual dynamic evolutions in the feature space. MIND-V demonstrates state-of-the-art performance in long-horizon robotic manipulation video generation, establishing a scalable and controllable paradigm for embodied data synthesis.

### Comprehensive comparison of MIND-V against SOTA models for long-horizon robotic video generation

<img src="assets/rada.png" width="88%"/>

<br>

### Long-Horizon Manipulation Demos

<!-- 强制两个 GIF 完全等大对齐 -->
<div align="center">
  <img src="assets/long1.gif" width="48%" style="margin:0; padding:0; border:none;"/>
  <img src="assets/long2.gif" width="48%" style="margin:0; padding:0; border:none;"/>
</div>

<br>

### Overview of our hierarchical framework for long-horizon robotic manipulation video generation

<img src="assets/pipeline.png" width="100%"/>

<div align="center">
  Beginning in the cognitive core, the <b>Semantic Reasoning Hub (SRH)</b> decomposes a high-level instruction into atomic sub-tasks and plans a detailed trajectory for each. These plans are then encapsulated into our novel <b>Behavioral Semantic Bridge (BSB)</b>, a structured, domain-invariant intermediate representation that serves as a precise blueprint for the <b>Motor Video Generator (MVG)</b>. The MVG, a conditional diffusion model, renders photorealistic videos that strictly adhere to the kinematic constraints defined in the BSB. At inference time, <b>Staged Visual Future Rollouts</b> provide a “propose-verify-refine” loop for self-correction, ensuring local optimality at each stage to mitigate error accumulation.
</div>

<br>

</div>




## ⚙️ Quick Start

### 1. Setup
Our environment setup is compatible with [CogVideoX 5B](https://github.com/THUDM/CogVideo). You can follow their configuration to complete the setup.

```bash
conda create -n mindv python=3.10
conda activate mindv
pip install -r requirements.txt
bash setup_MIND-V_env.sh
```

Download models from [download_models.sh](download_models.sh) and place them under the base root. The checkpoints should be organized as follows:

```
├── ckpts
    ├── CogVideoX-Fun-V1.5-5b-InP   (pretrained model base)
    ├── MIND-V                      (fine-tuned transformer)
    ├── sam2                        (segmentation model)
    ├── vjepa2                      (world models)
    └── affordance-r1               (semantic reasoning model)
```
The finetuned MIND-V checkpoints are available at https://huggingface.co/Richard-ZZZZZ/MIND-V/tree/main.
Download models using huggingface-cli:
```
pip install "huggingface_hub[cli]"
huggingface-cli download Richard-ZZZZZ/MIND-V --local-dir ./ckpts/MIND-V
```

**Required:** Configure your own Gemini API keyThe project uses [Google Gemini 2.5 Pro](https://blog.google/technology/google-deepmind/gemini-model-thinking-updates-march-2025/#gemini-2-5-thinking) (via service account) for visual captioning. Create a Google Cloud project and enable the Gemini API:  
```
Create a service account → Create Key → JSON  
Save the downloaded JSON as vlm_api/captioner.json
```
Example content (replace with your own values):
```
{
  "type": "service_account",
  "project_id": "your-project-id",
  "private_key_id": "your-key-id",
  "private_key": "-----BEGIN PRIVATE KEY-----\nYOUR_PRIVATE_KEY_HERE\n-----END PRIVATE KEY-----\n",
  "client_email": "xxx@your-project.iam.gserviceaccount.com",
  "client_id": "your-client-id",
  "auth_uri": "https://accounts.google.com/o/oauth2/auth",
  "token_uri": "https://oauth2.googleapis.com/token",
  "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
  "client_x509_cert_url": "https://www.googleapis.com/robot/v1/metadata/x509/xxx%40your-project.iam.gserviceaccount.com",
  "universe_domain": "googleapis.com"
}
```


### 2. Long-Horizon Video Generation

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

#### Pre-training (Supervised Fine-Tuning)

```bash
cd scripts
bash train_sft.sh
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
@misc{zhang2025mindvhierarchicalvideogeneration,
      title={MIND-V: Hierarchical Video Generation for Long-Horizon Robotic Manipulation with RL-based Physical Alignment}, 
      author={Ruicheng Zhang and Mingyang Zhang and Jun Zhou and Zhangrui Guo and Xiaofan Liu and Zunnan Xu and Zhizhou Zhong and Puxin Yan and Haocheng Luo and Xiu Li},
      year={2025},
      eprint={2512.06628},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2512.06628}, 
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





