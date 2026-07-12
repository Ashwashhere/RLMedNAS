<section id="overview">
  <h2>🎯 Problem Statement & Solution</h2>
  <p>
    Standard Neural Architecture Search (NAS) frameworks driven by Reinforcement Learning face a significant computational bottleneck: the agent routinely wastes massive GPU cycles exploring invalid, broken, or mathematically unviable layer configurations. 
  </p>
  <p>
    This project addresses that constraint by engineering an autonomous **Reinforcement Learning Controller (RNN)** to automate CNN architecture design for medical imaging workflows.By integrating domain-specific structural constraints directly into a **"Masked" Agent**, invalid network paths are systematically pruned *before* child network compilation.This targeted search space reduction forces the controller to focus exclusively on high-performance paths, dramatically accelerating convergence efficiency.</p>
</section>

<hr />

<section id="benchmarks">
  <h2>📊 Performance Benchmarks (MedMNIST)</h2>
  <p>The performance benefits of search-space masking compared to traditional baseline approaches:</p>
  
  <table style="width:100%; border-collapse: collapse; margin: 20px 0; font-family: sans-serif; text-align: left;">
    <thead>
      <tr style="border-bottom: 2px solid #30363d; color: #8b949e;">
        <th style="padding: 12px;">Search Strategy</th>
        <th style="padding: 12px; text-align: center;">Peak Validation Accuracy</th>
        <th style="padding: 12px; text-align: center;">Parameter Efficiency</th>
        <th style="padding: 12px;">Search Space Constraints</th>
      </tr>
    </thead>
    <tbody>
      <tr style="border-bottom: 1px solid #21262d;">
        <td style="padding: 12px; font-weight: bold;">Random Search Baseline</td>
        <td style="padding: 12px; text-align: center; color: #8b949e;">~72.5% - 87.5%</td>
        <td style="padding: 12px; text-align: center;">Baseline (1x)</td>
        <td style="padding: 12px; color: #8b949e;">Unconstrained (High GPU waste)</td>
      </tr>
      <tr style="border-bottom: 1px solid #21262d;">
        <td style="padding: 12px; font-weight: bold;">Standard RL Controller</td>
        <td style="padding: 12px; text-align: center; color: #8b949e;">~80.0% - 90.0%</td>
        <td style="padding: 12px; text-align: center;">Baseline (1x)</td>
        <td style="padding: 12px; color: #8b949e;">Unconstrained (Explores invalid paths)</td>
      </tr>
      <tr style="border-bottom: 1px solid #21262d; background-color: rgba(56, 139, 253, 0.1);">
        <td style="padding: 12px; font-weight: bold; color: #58a6ff;">Masked RL Controller (Ours)</td>
        <td style="padding: 12px; text-align: center; font-weight: bold; color: #3fb950;">96.7%</td>
        <td style="padding: 12px; text-align: center; font-weight: bold; color: #58a6ff;">3x Reduction</td>
        <td style="padding: 12px; font-weight: 500;">Enforced Validity Mask (Prunes invalid paths)</td>
      </tr>
    </tbody>
  </table>
</section>

<hr />

<section id="architecture">
  <h2>📁 Codebase Architecture</h2>
  <p>The framework features a highly modular, decoupled design splitting the policy controller from the network translation layer:</p>
  
  <pre style="background-color: #161b22; color: #e6edf3; padding: 16px; border-radius: 6px; overflow-x: auto; font-family: monospace; line-height: 1.4;">
├── controllers/
│   ├── rnn_agent.py          # RNN controller policy network configuration│   └── validity_mask.py      # Search space constraint masking logic tensor operations├── environment/
│   ├── child_network.py      # Dynamic CNN generation & compilation script
│   └── evaluator.py          # Reward calculation engine and MedMNIST data hub
├── utils/
│   └── plotting.py           # Evaluation visualisation & rolling mean scripts
├── requirements.txt          # Python dependency definitions
├── generate_preview.py       # Custom repository graphic generator
└── README.md</pre>
</section>

<hr />

<section id="quickstart">
  <h2>🚀 Quick Start & Evaluation</h2>
  <p>Execute a rapid-fire training run with the validity-masked agent using the steps below:</p>
  
  <pre style="background-color: #161b22; color: #e6edf3; padding: 16px; border-radius: 6px; overflow-x: auto; font-family: monospace; line-height: 1.4;">
<span style="color: #8b949e;"># Clone the repository and navigate to work directory</span>
git clone https://github.com/yourusername/RLMedNAS-S1.git
cd RLMedNAS-S1

<span style="color: #8b949e;"># Install core graphics and machine learning frameworks</span>
pip install -r requirements.txt

<span style="color: #8b949e;"># Run training loop with the validity-masked agent active</span>
python main.py --mode masked --episodes 100 --dataset medmnist</pre>
</section>

<hr />

<section id="evolution">
  <h2>📈 Project Evolution & Next Steps</h2>
  <blockquote style="margin: 20px 0; padding: 0 1em; color: #8b949e; border-left: .25em solid #30363d;">
    This repository details the foundational <strong>Semester 1</strong> design framework tailored specifically around constraint masking logic inside discrete search spaces.This codebase serves as the core architectural baseline for my ongoing <strong>MSc Graduate Thesis Project: RLMedNAS for Liver Cirrhosis Classification</strong>.The extended production framework scales the search environment into complex Directed Acyclic Graphs (DAGs) utilising <strong>PyTorch Geometric (PyG)</strong>and integrates highly specialised 3D imaging pipelines utilising <strong>MONAI</strong>.</blockquote>
</section>
