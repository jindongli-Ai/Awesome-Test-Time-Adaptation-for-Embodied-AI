# Awesome Test-Time Methods for Embodied AI 🤖

A curated collection of papers, methods, and benchmarks on **test-time learning, adaptation, policy steering, and scaling for embodied AI**.

**Companion survey (in preparation): _Test-Time Methods for Embodied AI: A Survey_.**

Maintained by [Jindong Li](https://github.com/jindongli-Ai). Contributions and corrections are welcome.

**Paper format:** `Year_Venue_MethodName_Full Paper Title.` Titles are plain text. Official publication links, arXiv abstract pages, source-code repositories, model weights, and project websites are listed separately when known. Unverified links are omitted.

> **Research question:** Once an embodied policy has been trained, how can it improve its behavior at test time through deployment observations, demonstrations, interaction feedback, or additional computation?

**Focus:** robotic manipulation and vision-language-action (VLA) policies, with relevant research from navigation and locomotion. **Literature cutoff:** October 2026.

## Contents

- [1. Scope and Definitions](#1-scope-and-definitions)
- [2. Taxonomy and Survey Outline](#2-taxonomy-and-survey-outline)
- [3. Test-Time Learning and Adaptation](#3-test-time-learning-and-adaptation)
- [4. Test-Time Policy Steering](#4-test-time-policy-steering)
- [5. Test-Time Scaling](#5-test-time-scaling)
- [6. Benchmarks and Evaluation](#6-benchmarks-and-evaluation)
- [7. Open Challenges](#7-open-challenges)
- [8. Related Collections and Surveys](#8-related-collections-and-surveys)
- [9. Contributing and Citation](#9-contributing-and-citation)

## 1. Scope and Definitions

This collection studies **deployment-stage mechanisms** that improve the behavior of a pretrained or previously trained embodied policy. A method may update model parameters, adapters or fast weights; use new context; steer action generation; or allocate additional inference-time computation.

| Term | Working definition |
| --- | --- |
| **Test-Time Learning (TTL)** | Uses deployment-time information to learn or acquire an adapted state; broadly encompasses some parametric and non-parametric adaptation mechanisms. |
| **Test-Time Adaptation (TTA)** | Adapts behavior to a target task, environment or distribution using information encountered at test time. |
| **Test-Time Training (TTT)** | Performs learning updates at test time, including gradient-based updates and fast-weight mechanisms; overlaps with TTA but is not synonymous with it. |
| **In-Context Adaptation** | Conditions an embodied policy on test-time demonstrations, trajectories or interaction history, commonly without updating its base weights. |
| **Policy Steering** | Guides the action-generation mechanism of an existing policy with rewards, value functions, dynamics, constraints or other policies. |
| **Test-Time Scaling (TTS)** | Increases or dynamically allocates inference-time compute for sampling, search, verification, refinement or reasoning; not necessarily a learning method. |

**Included:** test-time parameter/fast-weight adaptation, feedback-driven adaptation, in-context imitation, retrieval-augmented adaptation, inference-time action guidance, verification, search, and adaptive computation.

**Excluded from the core taxonomy:** offline SFT or RL fine-tuning completed before deployment, generic computer-vision TTA with no embodied policy, and classical planning not aimed at enhancing a learned embodied policy at inference time. Such work may be cited as background.

**Classification policy:** Papers are organized by **primary technical contribution**, not by a claim of mutual exclusivity. In particular, steering and scaling frequently overlap. Venue names are conservative: **arXiv** indicates preprint status unless a publication venue has been independently established.

## 2. Taxonomy and Survey Outline

Our three research themes are complementary:

| Research theme | Central question | Representative mechanisms |
| --- | --- | --- |
| **Test-Time Learning and Adaptation** | What information can change an embodied policy's deployed behavior? | Model adaptation/training; in-context learning/adaptation. |
| **Test-Time Policy Steering** | How can the action generation of a trained policy be guided? | Value/reward guidance; dynamics/constraint guidance; policy composition. |
| **Test-Time Scaling** | How can additional inference-time compute improve decisions? | Parallel sampling and verification; sequential search/refinement; adaptive compute. |

### Proposed paper outline

1. **Introduction**
2. **Background and Preliminaries**
   - 2.1. Embodied Policies and Test-Time Settings
   - 2.2. Test-Time Learning, Adaptation, and Scaling
   - 2.3. Scope and Taxonomy
3. **Test-Time Learning and Adaptation**
   - 3.1. Model Adaptation and Training
      - 3.1.1. Self-Supervised Adaptation
      - 3.1.2. Feedback-Driven Adaptation
      - 3.1.3. Fast-Weight Test-Time Training
   - 3.2. In-Context Learning and Adaptation
      - 3.2.1. Demonstration-Based Adaptation
      - 3.2.2. Interaction-History Adaptation
      - 3.2.3. Retrieval-Augmented Adaptation
4. **Test-Time Policy Steering**
   - 4.1. Value- and Reward-Guided Steering
   - 4.2. Dynamics- and Constraint-Guided Steering
   - 4.3. Policy Composition
5. **Test-Time Scaling**
   - 5.1. Parallel Sampling and Verification
   - 5.2. Sequential Search and Refinement
   - 5.3. Adaptive Compute Allocation
6. **Benchmarks and Evaluation**
   - 6.1. Test-Time Evaluation Settings
   - 6.2. Benchmarks and Tasks
   - 6.3. Evaluation Metrics and Protocols
   - 6.4. Performance–Efficiency Trade-offs
7. **Challenges and Future Directions**
8. **Conclusion**

The subsections are a working taxonomy and may be refined after a complete paper-by-paper screening.

---

## 3. Test-Time Learning and Adaptation

### 3.1. Model Adaptation and Training

Methods that update parameters, adapters, fast weights or learned internal representations using deployment information.

#### 3.1.1. Self-Supervised Adaptation

1. 2021_ICLR_PAD_Self-Supervised Policy Adaptation during Deployment.
   [[arXiv]](https://arxiv.org/abs/2007.04309)

2. 2023_NeurIPS_MoVie_MoVie: Visual Model-Based Policy Adaptation for View Generalization.
   [[arXiv]](https://arxiv.org/abs/2307.00972)

3. 2024_ICML_FSTTA_Fast-Slow Test-Time Adaptation for Online Vision-and-Language Navigation.
   [[arXiv]](https://arxiv.org/abs/2311.13209)

4. 2026_arXiv_TTT-VLA_TTT-VLA: Test-Time Latent Prompt Optimization for Vision-Language-Action Models.
   [[arXiv]](https://arxiv.org/abs/2606.03127)

5. 2026_arXiv_VANE_VANE: Reliable Test-Time Training for Vision-Language-Action Models.
   [[arXiv]](https://arxiv.org/abs/2608.09448)


#### 3.1.2. Feedback-Driven Adaptation

1. 2025_CoLLAs_ROAM_Adapt On-the-Go: Behavior Modulation for Single-Life Robot Deployment.
   [[arXiv]](https://arxiv.org/abs/2311.01059)

2. 2025_ICML_Method_Test-Time Adaptation for Online Vision-Language Navigation with Feedback-based Reinforcement Learning.
   [[Official]](https://proceedings.mlr.press/v267/kim25ad.html)

3. 2026_arXiv_TT-VLA_On-the-Fly VLA Adaptation via Test-Time Reinforcement Learning.
   [[arXiv]](https://arxiv.org/abs/2601.06748)

4. 2026_arXiv_TTT-Parkour_TTT-Parkour: Rapid Test-Time Training for Perceptive Robot Parkour.
   [[arXiv]](https://arxiv.org/abs/2602.02331)


#### 3.1.3. Fast-Weight Test-Time Training

1. 2026_arXiv_RoboTTT_RoboTTT: Context Scaling for Robot Policies.
   [[arXiv]](https://arxiv.org/abs/2607.15275)

2. 2026_arXiv_WAM-TTT_WAM-TTT: Steering World-Action Models by Watching Human Play at Test Time.
   [[arXiv]](https://arxiv.org/abs/2607.06988)


### 3.2. In-Context Learning and Adaptation

Methods that condition behavior on demonstrations or temporally accumulated evidence, typically without updating base policy weights.

#### 3.2.1. Demonstration-Based Adaptation

1. 2025_ICRA_ICRT_In-Context Imitation Learning via Next-Token Prediction.
   [[arXiv]](https://arxiv.org/abs/2408.15980)  [[GitHub]](https://github.com/Max-Fu/icrt)  [[Hugging Face]](https://huggingface.co/mlfu7/ICRT)  [[Project]](https://icrt.dev/)

2. 2025_ICLR_Instant-Policy_Instant Policy: In-Context Imitation Learning via Graph Diffusion.
   [[arXiv]](https://arxiv.org/abs/2411.12633)  [[Project]](https://www.robot-learning.uk/instant-policy)

3. 2025_arXiv_RICL_RICL: Adding In-Context Adaptability to Pre-Trained Vision-Language-Action Models.
   [[arXiv]](https://arxiv.org/abs/2508.02062)

4. 2026_arXiv_BPP_Behavior Prompting Policy: Demonstrations as Prompts for Manipulation.
   [[arXiv]](https://arxiv.org/abs/2606.30457)


#### 3.2.2. Interaction-History Adaptation

1. 2021_RSS_RMA_Rapid Motor Adaptation for Legged Robots.
   [[arXiv]](https://arxiv.org/abs/2107.04034)

2. 2023_ICML_Human-Timescale_Human-Timescale Adaptation in an Open-Ended Task Space.
   [[arXiv]](https://arxiv.org/abs/2301.07608)

3. 2025_CoRL_LocoFormer_LocoFormer: Generalist Locomotion via Long-context Adaptation.
   [[arXiv]](https://arxiv.org/abs/2509.23745)


#### 3.2.3. Retrieval-Augmented Adaptation

1. 2026_arXiv_RA-VLA_RA-VLA: Retrieval-Augmented VLA for Test-Time Adaptation.
   [[ICML]](https://proceedings.mlr.press/v306/jang26c.html)  [[arXiv]](https://arxiv.org/abs/2608.25585)


**Scope note:** Ordinary recurrent history or an unchanged task instruction does not automatically constitute test-time learning. We include work whose newly available context demonstrably supports behavioral adaptation.

---

## 4. Test-Time Policy Steering

Policy steering modifies **how actions are generated** from a pretrained policy, often through an auxiliary critic, value function, dynamics model or constrained action representation.

### 4.1. Value- and Reward-Guided Steering

1. 2024_CoRL_V-GPS_Steering Your Generalists: Improving Robotic Foundation Models via Value Guidance.
   [[arXiv]](https://arxiv.org/abs/2410.13816)

2. 2026_arXiv_VLS_VLS: Steering Pretrained Robot Policies via Vision-Language Models.
   [[arXiv]](https://arxiv.org/abs/2602.03973)

3. 2026_arXiv_Guided-Action-Flow_Guided Action Flow: Q-Guided Inference for Flow-Matching Vision-Language-Action Policies.
   [[arXiv]](https://arxiv.org/abs/2607.02092)


### 4.2. Dynamics- and Constraint-Guided Steering

1. 2025_NeurIPS_DynaGuide_DynaGuide: Steering Diffusion Policies with Active Dynamic Guidance.
   [[NeurIPS]](https://proceedings.neurips.cc/paper_files/paper/2025/hash/3ec6c6fc9065aa57785eb05dffe7c3db-Abstract-Conference.html)  [[arXiv]](https://arxiv.org/abs/2506.13922)  [[GitHub]](https://github.com/MaxDu17/DynaGuide)  [[Project]](https://dynaguide.github.io/)

2. 2025_NeurIPS_Latent-Policy-Barrier_Latent Policy Barrier: Learning Robust Visuomotor Policies by Staying In-Distribution.
   [[arXiv]](https://arxiv.org/abs/2508.05941)

3. 2026_arXiv_OmniGuide_OmniGuide: Universal Guidance Fields for Enhancing Generalist Robot Policies.
   [[arXiv]](https://arxiv.org/abs/2603.10052)

4. 2026_CoRL_Task-Structure-Steering_Inference-Time Robot Behavior Steering through Physically-Aware Reconfiguration of Task-Structure.
   [[arXiv]](https://arxiv.org/abs/2606.26588)


### 4.3. Policy Composition

1. 2026_ICLR_GPC_Compose Your Policies! Improving Diffusion-based or Flow-based Robot Policies via Test-time Distribution-level Composition.
   [[ICLR]](https://proceedings.iclr.cc/paper_files/paper/2026/hash/f7fc38fdd95fd146a471791b93ff9f12-Abstract-Conference.html)  [[arXiv]](https://arxiv.org/abs/2510.01068)  [[GitHub]](https://github.com/SageCao1125/GPC)  [[Project]](https://sagecao1125.github.io/GPC-Site/)

2. 2026_arXiv_PriGo_PriGo: Test-Time Primitive Guidance to Diffusion and Flow Policies for Adaptive Robotic Manipulation.
   [[arXiv]](https://arxiv.org/abs/2607.07076)

3. 2026_CoRL_Proxy-Policy-Steering_Proxy Policy Steering.
   [[arXiv]](https://arxiv.org/abs/2609.09148)


**Boundary:** Steering changes the action generation mechanism. Some steering methods also spend additional compute, so the boundary with scaling is not strictly exclusive.

---

## 5. Test-Time Scaling

Scaling improves decisions through more or adaptively allocated **inference computation**, without necessarily learning or updating weights.

### 5.1. Parallel Sampling and Verification

1. 2025_CoRL_RoboMonkey_RoboMonkey: Scaling Test-Time Sampling and Verification for Vision-Language-Action Models.
   [[CoRL]](https://proceedings.mlr.press/v305/kwok25a.html)  [[arXiv]](https://arxiv.org/abs/2506.17811)  [[Project]](https://robomonkey-vla.github.io/)

2. 2026_ICLR_MG-Select_Verifier-free Test-Time Sampling for Vision-Language-Action Models.
   [[ICLR]](https://openreview.net/forum?id=UD4Rw8MOEK)  [[arXiv]](https://arxiv.org/abs/2510.05681)  [[GitHub]](https://github.com/suhyeok-jang/mg-select)  [[Project]](https://suhyeok-jang.github.io/mg-select/)

3. 2025_arXiv_RoVer_RoVer: Robot Reward Model as Test-Time Verifier for Vision-Language-Action Model.
   [[arXiv]](https://arxiv.org/abs/2510.10975)

4. 2026_arXiv_CoVer_Scaling Verification Can Be More Effective than Scaling Policy Learning for Vision-Language-Action Alignment.
   [[arXiv]](https://arxiv.org/abs/2602.12281)


### 5.2. Sequential Search and Refinement

1. 2026_ICRA_VLA-Reasoner_VLA-Reasoner: Empowering Vision-Language-Action Models with Reasoning via Online Monte Carlo Tree Search.
   [[arXiv]](https://arxiv.org/abs/2509.22643)  [[Project]](https://vla-reasoner.github.io/)

2. 2026_arXiv_SAIL_SAIL: Test-Time Scaling for In-Context Imitation Learning with VLM.
   [[arXiv]](https://arxiv.org/abs/2603.08269)

3. 2026_arXiv_E-TTS_E-TTS: A New Embodied Test-Time Scaling Framework for Robotic Manipulation.
   [[arXiv]](https://arxiv.org/abs/2606.27268)

4. 2026_arXiv_tau0-WM_τ0-WM: A Unified Video-Action World Model for Robotic Manipulation.
   [[arXiv]](https://arxiv.org/abs/2606.01027)


### 5.3. Adaptive Compute Allocation

1. 2026_arXiv_VLA-ATTC_VLA-ATTC: Adaptive Test-Time Compute for VLA Models with Relative Action Critic Model.
   [[arXiv]](https://arxiv.org/abs/2605.01194)

2. 2026_CoRL_ELASTIC_ELASTIC: Efficiently Learning to Adaptively Scale Test-Time Compute for Generative Control Policies.
   [[arXiv]](https://arxiv.org/abs/2606.31132)

3. 2026_arXiv_tau0-VLA_τ0-VLA: A Hierarchical Robot Foundation Model with World-Model-Guided Test-Time Computation.
   [[arXiv]](https://arxiv.org/abs/2608.16885)


---

## 6. Benchmarks and Evaluation

### 6.1. Representative Environments

| Area | Example benchmarks / platforms | Main use |
| --- | --- | --- |
| Manipulation | [LIBERO](https://libero-project.github.io/), [SimplerEnv](https://simpler-env.github.io/), [RLBench](https://github.com/stepjam/RLBench), [robomimic](https://robomimic.github.io/), [RoboTwin](https://robotwin-platform.github.io/) | Task success under object, visual and environment changes. |
| Long-horizon manipulation | [CALVIN](https://github.com/mees/calvin) | Chained tasks and extended execution. |
| Navigation | [Habitat](https://aihabitat.org/), [VLN-CE](https://github.com/jacobkrantz/VLN-CE) | Unseen environments and episode adaptation. |
| Locomotion | [Isaac Lab](https://isaac-sim.github.io/IsaacLab/) | Dynamics changes and terrain robustness. |

*These are evaluation resources and platforms; they are not all test-time methods, nor is every benchmark used by every paper in Sections 3–5.*

### 6.2. Evaluation Dimensions

| Dimension | Suggested measurements |
| --- | --- |
| Task performance | Success rate, return, completion, performance gain over frozen baseline. |
| Generalization | Unseen environments, objects, camera shifts, dynamics changes, new embodiments. |
| Adaptation cost | Number of test-time updates, demonstrations, robot interactions, rewards, resets. |
| Compute and latency | Action inference time, control frequency, GPU time, candidates sampled, search steps, verifier calls. |
| Reliability | Failure rate, unsafe actions, drift, forgetting, rollback rate. |
| Fairness | Same initial policy, access to feedback, interaction/compute budgets, reset protocol. |

**Reporting principle:** compare both *performance versus deployment interaction budget* (learning/adaptation) and *performance versus test-time computation or latency* (scaling). Do not conflate offline pretraining costs with deployment costs.

### 6.3. Benchmark and Evaluation Papers

The following papers introduce or substantiate the evaluation platforms above. Paper titles are plain text; venue, arXiv abstract, code, and project links are provided separately.

1. 2023_NeurIPS_LIBERO_LIBERO: Benchmarking Knowledge Transfer for Lifelong Robot Learning.
   [[NeurIPS]](https://proceedings.neurips.cc/paper_files/paper/2023/hash/8c3c666820ea055a77726d66fc7d447f-Abstract-Datasets_and_Benchmarks.html)  [[arXiv]](https://arxiv.org/abs/2306.03310)  [[GitHub]](https://github.com/Lifelong-Robot-Learning/LIBERO)  [[Project]](https://libero-project.github.io/)

2. 2024_CoRL_SIMPLER_Evaluating Real-World Robot Manipulation Policies in Simulation.
   [[CoRL]](https://proceedings.mlr.press/v270/li25c.html)  [[arXiv]](https://arxiv.org/abs/2405.05941)  [[GitHub]](https://github.com/simpler-env/SimplerEnv)  [[Project]](https://simpler-env.github.io/)

3. 2020_RA-L_RLBench_RLBench: The Robot Learning Benchmark & Learning Environment.
   [[IEEE RA-L]](https://doi.org/10.1109/LRA.2020.2974707)  [[arXiv]](https://arxiv.org/abs/1909.12271)  [[GitHub]](https://github.com/stepjam/RLBench)

4. 2021_CoRL_robomimic_What Matters in Learning from Offline Human Demonstrations for Robot Manipulation.
   [[CoRL]](https://proceedings.mlr.press/v164/mandlekar22a.html)  [[arXiv]](https://arxiv.org/abs/2108.03298)  [[Project]](https://robomimic.github.io/)

5. 2025_CVPR_RoboTwin_RoboTwin: Dual-Arm Robot Benchmark with Generative Digital Twins.
   [[CVPR]](https://openaccess.thecvf.com/content/CVPR2025/html/Mu_RoboTwin_Dual-Arm_Robot_Benchmark_with_Generative_Digital_Twins_CVPR_2025_paper.html)  [[arXiv]](https://arxiv.org/abs/2504.13059)  [[GitHub]](https://github.com/RoboTwin-Platform/RoboTwin)  [[Project]](https://robotwin-platform.github.io/)

6. 2022_RA-L_CALVIN_CALVIN: A Benchmark for Language-Conditioned Policy Learning for Long-Horizon Robot Manipulation Tasks.
   [[IEEE RA-L]](https://ieeexplore.ieee.org/document/9788026)  [[arXiv]](https://arxiv.org/abs/2112.03227)  [[GitHub]](https://github.com/mees/calvin)

7. 2019_ICCV_Habitat_Habitat: A Platform for Embodied AI Research.
   [[ICCV]](https://openaccess.thecvf.com/content_ICCV_2019/html/Savva_Habitat_A_Platform_for_Embodied_AI_Research_ICCV_2019_paper.html)  [[arXiv]](https://arxiv.org/abs/1904.01201)  [[Project]](https://aihabitat.org/)

8. 2025_arXiv_Isaac-Lab_Isaac Lab: A GPU-Accelerated Simulation Framework for Multi-Modal Robot Learning.
   [[arXiv]](https://arxiv.org/abs/2511.04831)  [[GitHub]](https://github.com/isaac-sim/IsaacLab)  [[Project]](https://research.nvidia.com/labs/prl/publication/isaaclab2025/)

## 7. Open Challenges

1. **When to adapt?** Distinguishing meaningful deployment shifts from temporary observation noise or recoverable mistakes.
2. **What to adapt?** Selecting weights, latent states, contexts or inference-time guidance mechanisms.
3. **How to learn from weak feedback?** Handling delayed reward, incorrect success labels, imperfect critics and world models.
4. **How to preserve stability?** Avoiding policy drift, catastrophic forgetting, negative transfer and dangerous experiments.
5. **When does scaling pay off?** Balancing improved decisions against real-time latency, compute and control stability.
6. **How to evaluate fairly?** Accounting for extra demonstration access, interaction resets, computation and privileged feedback.

### 7.1. Related Papers on Reliability and Failure Monitoring

These studies provide concrete evidence on runtime monitoring, uncertainty, and safety-related challenges. Failure detectors are **supporting research**, not automatically test-time adaptation algorithms.

1. 2024_CoRL_Sentinel_Unpacking Failure Modes of Generative Policies: Runtime Monitoring of Consistency and Progress.
   [[CoRL]](https://proceedings.mlr.press/v270/agia25a.html)  [[arXiv]](https://arxiv.org/abs/2410.04640)  [[GitHub]](https://github.com/agiachris/sentinel)  [[Project]](https://sites.google.com/stanford.edu/sentinel)

2. 2025_RSS_FAIL-Detect_Can We Detect Failures Without Failure Data? Uncertainty-Aware Runtime Failure Detection for Imitation Learning Policies.
   [[RSS]](https://www.roboticsproceedings.org/rss21/p073.pdf)  [[arXiv]](https://arxiv.org/abs/2503.08558)  [[GitHub]](https://github.com/CXU-TRI/FAIL-Detect)  [[Project]](https://cxu-tri.github.io/FAIL-Detect-Website/)

3. 2025_NeurIPS_SAFE_SAFE: Multitask Failure Detection for Vision-Language-Action Models.
   [[NeurIPS]](https://proceedings.neurips.cc/paper_files/paper/2025/hash/392d0d05e2f514063e6ce6f8b370834c-Abstract-Conference.html)  [[arXiv]](https://arxiv.org/abs/2506.09937)  [[GitHub]](https://github.com/vla-safe/SAFE)  [[Project]](https://vla-safe.github.io/)

## 8. Related Collections and Surveys

### 8.1. Community Paper Collections

These items are GitHub literature collections, **not research papers**, so their repository names remain linked directly.

- [Awesome Test-Time Robot Learning](https://github.com/Oliverbansk/Awesome-Test-Time-Robot-Learning) — deployment-stage robot RL, adaptation, steering, in-context learning and scaling.
- [awesome_robo_ttt](https://github.com/asimfish/awesome_robo_ttt) — TTA/TTT foundations, fast-weight learning and robot policy deployment.

### 8.2. Related Survey Papers

1. 2024_arXiv_TTA-Survey_Beyond Model Adaptation at Test Time: A Survey.
   [[arXiv]](https://arxiv.org/abs/2411.03687)

2. 2025_IJCV_OTTA-Survey_In Search of Lost Online Test-Time Adaptation: A Survey.
   [[IJCV]](https://link.springer.com/article/10.1007/s11263-024-02213-5)  [[arXiv]](https://arxiv.org/abs/2310.20199)  [[GitHub]](https://github.com/Jo-wang/OTTA_ViT_survey)

3. 2026_Expert-Systems_TTL-Survey_From Static to Adaptive Inference: A Survey on Test-Time Learning.
   [[Expert Systems]](https://doi.org/10.1111/exsy.70446)

4. 2026_arXiv_TTI-Survey_A Survey on Self-Improving Test-Time Intelligence: Feedback-Driven Adapting, Learning, and Scaling at Inference.
   [[arXiv]](https://arxiv.org/abs/2609.01679)

This collection focuses on **embodied policy execution** and deployment-time interventions, whereas these surveys cover broader test-time methods in vision, language, multimodal AI, or general machine learning.

## 9. Contributing and Citation

Paper additions, verified publication updates, reproducibility results, and taxonomy corrections are welcome through [GitHub issues](https://github.com/jindongli-Ai/Awesome-Test-Time-Methods-for-Embodied-AI/issues) or pull requests. Please provide the **paper title, stable public URL, release year, verified venue, deployment-time mechanism, and its reason for inclusion**.

**Citation:** The companion survey is in preparation. Please cite original papers when discussing their methods and link to this repository when referring to this collection. We will provide a BibTeX entry when a public survey preprint is available.

---

Maintained by [Jindong Li](https://github.com/jindongli-Ai). This is an evolving literature map; categorization and publication metadata will be updated as the survey progresses.
