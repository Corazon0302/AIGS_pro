# Multi-Agent Design: Optimizing Agents with Better Prompts and Topologies  

Han Zhou1 2 \*, Xingchen Wan1, Ruoxi $\mathsf{s u n}^{1}$ , Hamid Palangi1, Shariq Iqbal1, Ivan Vulić1 2, Anna Korhonen2   
and Sercan Ö. Arık1   
1Google, 2University of Cambridge  

Large language models, employed as multiple agents that interact and collaborate with each other, have excelled at solving complex tasks. The agents are programmed with prompts that declare their functionality, along with the topologies that orchestrate interactions across agents. Designing prompts and topologies for multi-agent systems (MAS) is inherently complex. To automate the entire design process, we first conduct an in-depth analysis of the design space aiming to understand the factors behind building effective MAS. We reveal that prompts together with topologies play critical roles in enabling more effective MAS design. Based on the insights, we propose Multi-Agent System Search (Mass), a MAS optimization framework that efficiently exploits the complex MAS design space by interleaving its optimization stages, from local to global, from prompts to topologies, over three stages: 1) block-level (local) prompt optimization; 2) workflow topology optimization; 3) workflow-level (global) prompt optimization, where each stage is conditioned on the iteratively optimized prompts/topologies from former stages. We show that Mass-optimized multi-agent systems outperform a spectrum of existing alternatives by a substantial margin. Based on the Mass-found systems, we finally propose design principles behind building effective multi-agent systems.  

## 1. Introduction  

Large language models (LLMs) have showcased extraordinary capabilities in understanding, reasoning, and generating coherent responses based on user prompts, revolutionizing a wide range of applications (Kojima et al., 2022; Ouyang et al., 2022). LLM-based agents enhance usability by autonomously handling complex tasks across diverse domains, including code generation and debugging (Jimenez et al., 2023), retrievalaugmented generation (Singh et al., 2025; Wang et al., 2024a), data analysis (Guo et al., 2024; Hu et al., 2024b), and interactive decision-making (Li et al., 2025; Su et al., 2025). These agents are typically programmed with prompts that reinforce them to interact with the environment, utilizing available tools, and approach their objectives over multiple turns (Yao et al., 2023). Beyond individual agents, LLMs can be orchestrated within complex topologies that coordinate multiple agents  

![](https://cdn-mineru.openxlab.org.cn/extract/480bcfb1-0272-4d7a-b502-2d4631527442/131e64f679a5f47c583ac2fb81f030bf3ba2dfb3eb5a0a57c6e17129d4a0a03c.jpg)  
Figure 1 | Proposed Multi-Agent System Search (Mass) framework discovers effective multiagent system designs (with both optimized topology and optimized prompts, right) via interleaved prompt optimization and topology optimization in a customizable multi-agent design space (key components illustrated on the left).  

toward a shared objective. This type of multi-agent system (MAS) typically outperforms its singleagent counterpart by involving more diverse agentic perspectives or role profiles, such as agents as verifiers (Shinn et al., 2024) and multi-agent debate (Qian et al., 2024; Wang et al., 2024b).  

However, designing effective MAS for new domains often proves to be challenging. First, the single agent might suffer from prompt sensitivity (Verma et al., 2024), where simple modifications in the prompt can already exert significant but unexpected degradation of performance (Liu et al., 2024a; Zhou et al., 2024b). In MAS, when sensitive agents are cascaded, the compounding effect due to prompt sensitivity may be amplified. Together with the prompt design, crafting an effective topology might demand a substantial amount of manual experimentation, based on trial and error. The problem complexity is exacerbated by the overall combinatorial search space, over not only the unbounded space of prompt design but also the design decisions of what agent to integrate into the topology.  

Although recent research has explored automating various aspects of agentic designs, there is still a gap in understanding of what matters most regarding improved MAS performance. For example, DSPy (Khattab et al., 2024) automates the process of designing exemplars for improved prompt programming. Li et al. (2024a) proposes to optimize MAS by scaling up the number of agents in majority voting. ADAS (Hu et al., 2024a) programs new topologies expressed in code via an LLM-based meta-agent. AFlow (Zhang et al., 2024b) searches better topologies using Monte Carlo Tree Search within a set of predefined operators. However, the interplay between multiple design spaces, including prompts and topologies, remains unclear.  

In this paper, we first conduct in-depth analyses of common design spaces in MAS, examining the influence of various aspects such as optimizing the prompts, scaling the number of agents, and involving different types of topologies. Our analyses reveal that prompts frequently form an influential design component that yields strong-performing MAS, and influential topologies only represent a small fraction of the full search space. Based on these insights, we aim to distill the essence of influential MAS components into a pruned search space, thereby lowering the complexity of the overall search process. We propose Multi-Agent System Search (Mass), a novel multi-stage optimization framework that automates the optimization for MAS over an efficient search space. Mass integrates a plugand-play prompt optimizer and workflow optimizer over a configurable topology space. It overcomes the complexity of joint optimization on MAS by interleaving the optimization stages, from local to global, from prompts to topologies, over three stages: 1) block-level (local) prompt ‘warm-up’ for each topology block; 2) workflow topology optimization in a pruned set of topology space; 3) workflow-level (global) prompt optimization given the best-found topology.  

By optimizing over the identified influential components, Mass yields optimized MAS that achieves state-of-the-art performance, outperforming existing manually-crafted MAS baselines and automatically-generated alternatives, by a substantial margin, demonstrated across an extensive selection of tasks, including reasoning, multi-hop understanding, and code generation. Based on the strongest MAS found by Mass, we provide further insights and guidelines behind building effective MAS. Overall, our contributions can be summarized as follows: 1) we provide an in-depth analysis of the design factors that influence the performance of LLM-based MAS, highlighting the importance of prompts and identifying the influential topologies; 2) we propose Mass, a novel multi-stage optimizer that automates the MAS design by interleaving the optimization of prompts and topologies in an influential search space; 3) Mass shows significant performance improvement on various evaluation benchmarks, delivering guidelines for building effective multi-agent systems for the future.  

## 2. Designing Multi-Agent Systems  

In this section, we provide a formulation for designing MAS, followed by analyzing the influence of prompt and topology designs. We refer to the structural arrangements of agents (or equivalently, building blocks) as the topology of agents and define workflow $\mathcal{W}$ as the logical sequence across different topologies that builds the MAS. The design of a MAS can thus be broadly divided into two levels: block-level design and workflow-level orchestration. At the block level, we aim to design effective individual agents that best perform their intended role with better prompt design. On the other hand, at the workflow level, the optimization involves determining the types and quantities of agents to include and how to arrange them in the most effective way, referred to as the topology optimization. Formally, given a search space $\mathcal{A}$ that defines all valid configurations 𝑎 over the blocks (see Fig. 4), workflow topology optimization can be expressed as the following optimization problem with an objective function $f(\cdot,\cdot)$ on a target input and output set $(x,y)\sim\mathcal{D}$ :  

$$
\mathcal{W}^{*}(a)=\underset{a\sim\mathcal{A}}{\arg\operatorname*{max}}\mathbb{E}_{(x,y)\sim\mathcal{D}}[f(\mathcal{W}(a(x)),y)].
$$  

In the rest of this section, we provide an in-depth analysis of each component of MAS design.  

### 2.1. Block-level: Prompt Design for Agents  

At the block level, the primary “optimizable component” that significantly influences downstream performance is the prompt, which defines the role of the agent (e.g., “You are an expert in reflecting on errors...”), provides additional instructions to shape its behavior (e.g., “You should think step by step...”) and optionally, contains few-shot demonstrations (in-context examples) to guide the agent’s responses (Wan et al., 2024, 2025). For instance, a state-of-the-art prompt optimizer searches both instructions and few-shot demonstrations, where demonstrations are bootstrapped from the model’s own, correct predictions on the validation set based on a validation metric. Conditioned on the demonstrations, the prompt optimizer then proposes a few candidates for the instruction with a dataset summary or various hints to improve candidate diversity (Opsahl-Ong et al., 2024). The instructions and demonstrations are then jointly optimized.  

![](https://cdn-mineru.openxlab.org.cn/extract/480bcfb1-0272-4d7a-b502-2d4631527442/5cb21af4bf5147ee803f2772726bf0e21631e6a22e0ed629f69e5e7f97ade7c8.jpg)  
Figure 2 | Accuracy vs. the total token counts for prompt-optimized agents per question on MATH by Gemini $1.5\mathrm{~Pro~}$ compared to scaling agents with self-consistency (SC), self-refine (reflect), and multi-agent debate (debate) only. The error bar indicates 1 standard deviation. We show that by utilizing more compute, better accuracy can be obtained via more effective prompting.  

Although it is well known that LLMs are sensitive to prompts (Verma et al., 2024; Zhou et al., 2024a), applying automatic prompt optimization (APO) techniques to MAS is rather non-trivial. Unlike single-turn tasks where APO can be easily performed by treating prompts as optimizable variables and performance over a validation set as the target. In MAS, APO becomes more complex due to the interdependence across agents (e.g., the output of one agent may be the input of another agent in a cascade with ground-truth responses for intermediate outputs not being available) and exponentially increasing complexity for combinatorial optimization with more number of agents 𝑛 involved; The reward signals also become more sparse when 𝑛 increases, preventing us for implementing APO directly on MAS in any manageable budget; as such, many prior works (Xia et al., 2024; Zhang et al., 2024f) in MAS still primarily use handcrafted prompts instead of including the prompts as optimizable components in the MAS design.  

To systematically understand the influence of prompt design in MAS, we specifically and quantitatively analyze the effect of prompt optimization and compare its effectiveness to other operations common in MAS literature, such as scaling with more agents but with default prompts. We conduct  

APO on a chain-of-thought (Kojima et al., 2022) agent with both instruction optimization and 1-shot exemplar optimization via MIPRO (Opsahl-Ong et al., 2024), and fairly compare the total inference token cost with self-consistency (Kojima et al., 2022), self-refine (Madaan et al., 2024), and multi-agent debate (Du et al., 2024), where the specifications are provided in App. $\S\mathrm{B}$ . In Fig. 2, prompting, which equips agents with more informative instructions and exemplars, demonstrates significant advantages in its token-effectiveness over other building blocks. Furthermore, by applying self-consistency on top of the prompt-optimized agent, we observe an improved scaling performance on the token cost, whereas standard approaches in scaling the number of agents (e.g. SC, or Reflect) saturate much earlier. This empirical observation sheds light on the importance of prompting while providing early evidence for designing effective MAS – optimize agents locally before scaling their topology.  

### 2.2. Workflow-level Search Space Design  

At the workflow level, the primary focus is on orchestrating agents to achieve the best performance effectively. As a relatively new concept specific to MAS, topology optimization has recently garnered significant attention (Li et al., 2024c; Zhang et al., 2024b). However, while much of the existing research emphasizes search methods—such as discovering the most efficient and effective way to identify the optimal configuration—there has been less focus on the design of search spaces, which determines the perimeter and the scope of any search algorithm. This imbalance draws a parallel to the historical development of neural architecture search (NAS) (White et al., 2023). Initially, the field concentrated on sophisticated search methods, such as Bayesian optimization (Kandasamy et al., 2018; Ru et al.,  

![](https://cdn-mineru.openxlab.org.cn/extract/480bcfb1-0272-4d7a-b502-2d4631527442/95c6a49f3c6acbf68f05b3eae2d0fe1007900cc89af2fc764dc400f60fccb636.jpg)  
Figure 3 | The performance of different topologies with Gemini 1.5 Pro compared to the base agent with each topology being optimized with APO, where Sum. (Summarize) and Exe. (Executor) are task-specific topologies as illustrated in Fig. 4. We observe that not all topologies have a positive influence on the MAS design.  

2021) and differentiable search (Liu et al., 2018). Follow-up works have highlighted the oftenoverlooked importance of search space design, arguing that it can be equally, if not more, critical (Wan et al., 2022; Zhou et al., 2023, 2024c). Inspired by this insight, we hypothesize that manually crafted topologies might be sub-optimal, and automatic topology optimization (potentially framed as a rigorous optimization problem) can play a similarly pivotal role via judiciously designing search space for MAS. To achieve so, we first define an expressive search space, similar to prior works, that consists of the connections between the following building blocks:  

• Aggregate: Agents can collaborate in parallel with diversified predictions, which is then followed by an aggregation operator that obtains the most consistent prediction. The aggregate block can be parameterized by $N_{a}$ agents acting in parallel. Majority vote (Li et al., 2024a) and self-consistency (Chen et al., 2024c) sits within this topology.  

• Reflect: Agents can act as verifiers, providing critics and improvement suggestions based on former predictions. The feedback is then fed into the predictor or the reflector itself for an iterative improvement. Similarly, reflect can be parameterized by $N_{r}$ that defines the number of rounds for self-reflection. The self-refine (Madaan et al., 2024) and Reflexion (Shinn et al., 2024) represent this block.  

• Debate: Agents in debate can elicit more truthful predictions than single-agent prediction (Du et al., 2024; Liang et al., 2024), where each debating agent would collect opinions from all other agents and provides an updated response. This topology would involve a mixture of agents, and $N_{d}$ defines the number of rounds for debating.  

![](https://cdn-mineru.openxlab.org.cn/extract/480bcfb1-0272-4d7a-b502-2d4631527442/d11f35d74e81b956e3bfa148e5d21f1e2855b04885ea2d3be63e84f064e9e92f.jpg)  
Figure 4 | Illustration of the Mass framework with its search space and the multi-stage optimization. The search space combines both prompts (Instruction, Demo) and configurable agentic building blocks (Aggregate, Reflect, Debate, Summarize, and Tool-use). 1) Block-level Prompt Optimization: we conduct block-level prompt optimization for each agentic module individually (denoted by $</>\cdot$ ); 2) Workflow Topology Optimization: conditioned on the best prompts found in Stage 1 on each agent block, Mass samples valid configurations from an influence-weighted design space while fusing the prompts of each building block from Stage 1; 3) Workflow-level Prompt Optimization: conditioned on the best workflow found in the Stage 2, we again conduct workflow-level prompt optimization on the best-found MAS (topologies visualized for illustration only).  

• Custom Agents: While the former three forms of agents represent the vast majority of agent topologies constructed as multiple parallel, serial, and mixture of agents, more versatile definitions of agents can be inserted into the MAS design space. For example, for task-specific use cases, we introduce an agent as summarize to improve the long-context capability in the customizable design space.  

• Tool-use: Building towards an effective MAS, enabling agents to leverage tools to access external information is critical for system performance, such as using retriever for RAG (Lewis et al., 2020) and executor with test cases in coding (Chen et al., 2024d). We introduce tool-use as an optimizable binary ‘insertion’ decision $N_{T}\in\{0,1\}$ .  

To understand the influence of individual topology, we report the performance of various topologies in Fig. 3. It is noticeable that not all topologies are beneficial to MAS design, whereas positively influenced topologies only represent a small fraction of the overall set, such that, in HotpotQA (Yang et al., 2018), only debate brings $3\%$ gain while others fail to improve or even degrade systematic performance. We again observe similar trends in the test-output-prediction subtask of LiveCodeBench (Jain et al., 2024). It highlights the importance of searching in the influential set of search space, whereas including decremental building blocks may not only result in higher search complexity but also degrade the performance.  

## 3. Mass: Multi-Agent System Search  

Our analyses in Sec. 2 underscore the importance of well-designed prompts for individual agents and the careful definition of the search space to achieve effective MAS performance. Building on these, we propose a multistage optimization algorithm, Multi-Agent System Search (Mass), that surpasses prior arts that focused solely on optimizing workflow topology without appropriate prompt designs. Instead, our approach demonstrates the greater effectiveness of MAS design with properly optimized prompts and thoughtfully designed search spaces. Mass framework is illustrated in Algorithm 1 and Fig. 4, following an intuition from local to global, from block-level to workflow-level, that conquers the complexity of combinatorial optimization with effective per-stage optimization detailed below.  

1) Block-level prompt optimization. Before composing agents, we first ensure that individual agents are thoroughly optimized at the block level, as highlighted in Sec. 2.1 and Fig. 2 – this step ensures that each agent is primed for its role with the most effective instructions in the most manageable computation budget. To further overcome the complexity of joint optimization on a large MAS space, we first warm up the initial predictor with single-agent APO, $a_{0}^{*}\leftarrow O_{\mathcal{D}}(a_{0})$ , where both instruction and exemplars are jointly optimized with the modular prompt optimizer $o$ . Followed by conditioning on the warmed predictor, we continue optimizing each topology with a minimum number of agents, $a_{i}^{*}\leftarrow O_{\mathcal{D}}(a_{i}|a_{0}^{*})$ , such that, 2 predictors paired with 1 debator form the minimum building block as the debate topology, thereby lowering the complexity for optimization, and this topology can be scaled up later with more predictors and debators but all equipped with optimized prompts. To measure the influence of each building block, we store the validation performance once the optimization is completed. It is important that though Stage (1) serves as the warm-up stage per building block, it is still a critical stage that guarantees the follow-up topology  

### Algorithm 1 Mass: Multi-Agent System Search  

1: Input: Agentic modules in the search space $a_{i}\in\mathcal{A}$ , workflow of agents $\textstyle{\mathcal{W}}(a)$ , prompt optimizer $o$ , evaluator $\varepsilon$ , validation set $\mathcal{D}$ , temperature $t_{.}$ , number of candidates $N$ , budget $B$ .   
2: Output: Optimized multi-agent system $\mathbf{\boldsymbol{\mathcal{W}}^{*}}$ .   
3: [Block-level Prompt Optimization]   
4: Prompt optimization for the initial agent $a_{0}^{*}\leftarrow O_{\mathcal{D}}(a_{0})$ .   
5: for $a_{i}$ in $\mathcal{A}\setminus\{a_{0}\}$ do   
6: Local prompt optimization for each building block in the design space: $a_{i}^{*}\leftarrow O_{\mathcal{D}}(a_{i}|a_{0}^{*})$   
7: Obtain incremental Influence $I_{a_{i}}\leftarrow\mathcal{E}(a_{i}^{*})/\mathcal{E}(a_{0}^{*})$ .   
8: end for   
9: [Workflow Topology Optimization]   
10: Obtain the selection probability $p_{a}\gets S o f t m a x(I_{a},t)$   
11: while $n<N$ do   
12: Reject invalid configurations 𝑐 and cap a budget $B$ . The design space is pruned by the selection probability $p_{a}$ , $\mathcal{W}_{c}\gets(a_{i}^{*}(\cdot),a_{i+1}^{*}(\cdot),\cdot\cdot\cdot)$ with optimized prompts.   
13: Store evaluations $\mathcal{E}_{\mathcal{D}}(\mathcal{W}_{\mathscr{C}})$ and propose new $\scriptscriptstyle{\mathcal{W}}$ .   
14: end while   
15: Obtain the best-performing $\mathcal{W}_{c}^{*}$ arg $\mathrm{max}_{c\in C}\mathcal{E}_{\mathcal{D}}(\mathcal{W}_{c})$ .   
16: [Workflow-level Prompt Optimization]   
17: Workflow-level prompt optimization for the bestperforming topology: ${\mathcal{W}}^{*}\gets O_{\mathcal{D}}({\mathcal{W}}_{c}^{*})$ .   
18: Return optimized multi-agent system $\mathbf{\boldsymbol{\mathbf{\mathit{\sigma}}}}\mathbf{\boldsymbol{\mathbf{\mathit{W}}}}^{*}$ .  

optimization is searching in an effective space, composing well-performing agents instead of suffering from the compounding impact from any ill-formed agents with manual prompts.  

2) Workflow topology optimization. In this stage, we focus on optimizing the overall MAS structure, determining the most effective arrangement and connectivity between agents. The analysis in Fig. 3 shows that beneficial topologies only represent a small fraction of the full design space. Therefore, we aim to distill the essence of strong-performing topologies into a pruned space, thereby making the workflow-level topology search more efficient. Here, we propose to measure the incremental influence $I_{a_{i}}=\mathcal{E}(a_{i}^{*})/\mathcal{E}(a_{0}^{*})$ that quantifies the relative gain for integrating the topology $a_{i}$ over the initial agent $a_{0}$ . Following the intuition that influential dimension comes with higher selection probability, we activate the corresponding topology dimension $a$ if $u>p_{a}$ , given $u\sim\mathcal{U}(0,1)$ and $p_{a}=\mathbf{S}\mathbf{oftmax}(I_{a},t)$ . To compose diverse topologies into a unified space, we constrain the workflow with a rule-based order to reduce the optimization complexity, following a predefined sequence, such that [summarize, reflect, debate, aggregate]. We integrate rejection sampling over the pre-defined design space that rejects any deactivated dimension, or invalid topology compositions exceeding a maximum budget $B$ on the number of agents. We refer to App. §B for the detailed search space per task.  

3) Workflow-level prompt optimization. As a final step, we treat the entire MAS design as an integrated entity and run an additional round of prompt optimization, conditioned on the best topology discovered in Stage (2), $\mathcal{W}^{*}=\mathcal{O}_{\mathcal{D}}(\mathcal{W}_{c}^{*})$ . It is worth noting that although prompts were optimized at the individual level in Stage (1), this stage acts as an adaptation or fine-tuning process, ensuring that prompts are tailored for orchestration within the MAS and that the interdependence between agents is optimized appropriately. Our experiments (Fig. 5 & 6) demonstrate that this stage often yields practical benefits.  

## 4. Related Work  

Forms of LLM-based agentic systems. The simplest form of an LLM-based agentic system involves a single agent that can dynamically interact and respond to the environment (Yao et al., 2023). Recent advances endow agents with diverse roles and tools (Wu et al., 2023), orchestrating multiple agents to cooperate with each other (Chen et al., 2024b). Standard forms of agent cooperation (i.e., topology) often involve parallel and serial flows of information. The parallel form usually diversifies the exploration among many agents in parallel (Li et al., 2024a), and self-consistency (SC) (Wang et al., 2023) is a representative way for scaling agents in parallel. The serial form aims to advance the exploitation of a task via a chain of agents, where LLMs can serve as reflective agents to self-justify and refine former predictions (Madaan et al., 2024; Shinn et al., 2024). Later, the opinions from multiple agents can be summarized to retrieve the most consistent answer by an aggregation agent (Chen et al., 2024c; Lin et al., 2024). Moreover, multi-agent debate consists of a more complex flow of information (Chen et al., 2024a; Wang et al., 2024c; Zhang et al., 2024c), and recent research shows that debating can elicit more truthful predictions (Du et al., 2024; Khan et al., 2024). Recent agent topology extends beyond the above connections (Qian et al., 2024; Wang et al., 2024b), and Mass can automatically search the best topology among the aforementioned spaces.  

Automatic optimization for MAS. Recent research starts automating agent design by interpreting agent functions as learnable policies (Zhang et al., 2024d,e) and synthesizing trajectories for agent fine-tuning (Qiao et al., 2024). Going further from a single agent, automatic multi-agent optimization faces a higher level of complexity, thereby requiring a more sophisticated design of search space and algorithms. Among all recent advances in multi-agent optimization, the optimization space has spanned prompts (Khattab et al., 2024), tools (Zhou et al., 2024d), workflows (Li et al., 2024c), and thinking strategies (Shang et al., 2024). Aligning closer to our topology search space, DyLAN (Liu et al., 2024b) dynamically activates the composition of agents, and Archon (Saad-Falcon et al., 2024) frames MAS as a hyperparameter optimization problem. Neither of them has taken the important prompt space into account, where we demonstrated the importance of prompt optimization in Sec. 2.2. In addition, GPTSwarm (Zhuge et al., 2024) optimizes the connections between agentic nodes using a policy gradient algorithm. State-of-the-art automatic agent design methods, ADAS (Hu et al., 2024a) and AFlow (Zhang et al., 2024b), also attempt to optimize agentic workflows with advanced search algorithms and LLM as optimizers. However, we observe that the importance of proper prompt designs has been relatively under-studied in these prior works.  

## 5. Experiments  

Models and evaluation data. Aside from the common benchmarks used for automating MAS (Hu et al., 2024a; Zhang et al., 2024b), we conduct experiments on an extensive collection of tasks: 1) Hendryck’s MATH (Hendrycks et al., 2021) and DROP (Dua et al., 2019) for reasoning; HotpotQA (Yang et al., 2018), MuSiQue (Trivedi et al., 2022), 2WikiMultiHopQA (Ho et al., 2020) from LongBench (Bai et al., 2024) for long-context understanding; 3) MBPP (Austin et al., 2021),  

Table 1 | Results on the evaluation set with Gemini $1.5\mathrm{Pro}$ and Gemini 1.5 Flash. We report the mean and standard deviation for all results with 3 runs of evaluations. We report the accuracy $(\%)$ for MATH and the test-output-prediction subtask of LiveCodeBench (LCB), F1 score for DROP, HotpotQA, MuSiQue, and 2WikiMQA, and pass $@1$ for MBPP and HumanEval. We note that the meta-prompt of AFlow\* only works properly with Claude 3.5 Sonnet. Therefore, we reproduce AFlow with Gemini $1.5\mathrm{Pro}$ as the executor and Claude 3.5 Sonnet as the optimizer, where \* indicates the results are only for reference. Number of agents in inference for all methods are below 10.  

<html><body><table><tr><td colspan="10">Gemini-1.5-pro-002</td></tr><tr><td>Task</td><td colspan="2">Reasoning</td><td colspan="3">Multi-hop Long-context</td><td colspan="3">Coding</td><td></td></tr><tr><td>Method</td><td>MATH</td><td>DROP</td><td>HotpotQA</td><td>MuSiQue</td><td>2WikiMQA</td><td>MBPP</td><td>HumanEval</td><td>LCB</td><td>Avg.</td></tr><tr><td>CoT</td><td>71.673.30</td><td>70.591.67</td><td>57.430.52</td><td>37.811.43</td><td>63.391.12</td><td>68.330.47</td><td>86.670.94</td><td>66.330.62</td><td>65.28</td></tr><tr><td>Self-Consistency</td><td>77.331.25</td><td>74.060.90</td><td>58.602.19</td><td>41.811.00</td><td>67.791.19</td><td>69.500.71</td><td>86.000.82</td><td>70.330.94</td><td>68.18</td></tr><tr><td>Self-Refine</td><td>79.672.36</td><td>71.031.31</td><td>60.623.33</td><td>42.151.34</td><td>66.742.43</td><td>63.670.24</td><td>84.001.63</td><td>67.331.31</td><td>66.90</td></tr><tr><td>Multi-Agent Debate</td><td>78.670.94</td><td>71.780.71</td><td>64.870.23</td><td>46.000.80</td><td>71.780.63</td><td>68.670.85</td><td>86.671.25</td><td>73.671.65</td><td>70.26</td></tr><tr><td>ADAS</td><td>80.000.82</td><td>72.960.90</td><td>65.881.29</td><td>41.951.24</td><td>71.140.66</td><td>73.001.08</td><td>87.671.70</td><td>65.171.25</td><td>69.72</td></tr><tr><td>AFlow*</td><td>76.000.82</td><td>88.920.63</td><td>68.620.47</td><td>32.051.29</td><td>76.511.05</td><td></td><td>88.000.00</td><td></td><td></td></tr><tr><td>MASS (Ours)</td><td>84.670.47</td><td>90.520.64</td><td>69.911.11</td><td>51.400.42</td><td>73.340.67</td><td>86.500.41</td><td>91.670.47</td><td>82.330.85</td><td>78.79</td></tr><tr><td colspan="10">Gemini-1.5-flash-002</td></tr><tr><td>CoT</td><td>66.672.36</td><td>71.790.69</td><td>57.821.10</td><td>37.101.35</td><td>63.400.68</td><td>63.331.25</td><td>75.671.89</td><td>51.170.24</td><td>60.87</td></tr><tr><td>Self-Consistency</td><td>69.331.25</td><td>73.420.19</td><td>60.191.01</td><td>41.940.93</td><td>67.980.72</td><td>63.670.62</td><td>77.671.89</td><td>53.831.18</td><td>63.50</td></tr><tr><td>Self-Refine</td><td>71.330.94</td><td>73.711.09</td><td>58.843.04</td><td>41.211.99</td><td>65.561.57</td><td>63.331.25</td><td>81.671.89</td><td>52.001.41</td><td>63.46</td></tr><tr><td>Multi-Agent Debate</td><td>71.670.94</td><td>74.790.87</td><td>64.171.69</td><td>46.271.33</td><td>72.190.54</td><td>63.000.71</td><td>79.671.25</td><td>55.500.41</td><td>65.91</td></tr><tr><td>ADAS</td><td>68.001.41</td><td>75.951.18</td><td>61.362.89</td><td>48.811.03</td><td>66.901.00</td><td>65.830.24</td><td>80.672.49</td><td>50.501.63</td><td>64.75</td></tr><tr><td>MASS (Ours)</td><td>81.002.45</td><td>91.680.14</td><td>66.530.38</td><td>43.671.21</td><td>76.690.50</td><td>78.000.82</td><td>84.670.47</td><td>72.170.85</td><td>74.30</td></tr></table></body></html>  

HumanEval (Chen et al., 2021), and LiveCodeBench (LCB) ‘test output prediction’ (Jain et al., 2024) for coding. We refer to App. $\S\mathrm{B}$ & $\S\mathrm{D}$ for details on data splits and prompt templates. We run all experiments primarily on two Gemini 1.5 model sizes (Reid et al., 2024) (gemini-1.5-pro-002 and (gemini-1.5-flash-002) and further validate key findings on Claude 3.5 Sonnet (@20240620) (Anthropic, 2024).  

Baselines. We consider the following baselines: 1) CoT (Kojima et al., 2022): direct chain-of-thought reasoning via zero-shot prompting; 2) CoT-SC (Wang et al., 2023): with self-consistency to find the most consistent answers from diversified reasoning traces; 3) Self-Refine (Madaan et al., 2024; Shinn et al., 2024): reflective agents to verify and self-refine predictions; 4) Multi-Agent Debate (Du et al., 2024; Liang et al., 2024): with agent justifying answers and aggregating information from other agents; 5) ADAS (Hu et al., 2024a): an automatic agent design framework, where an LLM-based meta-agent iteratively proposes new agents based on former evaluations; 6) AFlow (Zhang et al., 2024b): automatic workflow design via Monte-Carto Tree Search over a set of pre-defined operators. We fairly compare all baselines by limiting the maximum number of agents to 10. We refer to App. §B for all specifications.  

Setup. Mass integrates the state-of-the-art prompt optimizer, MIPRO (Opsahl-Ong et al., 2024), which optimizes both instructions and demonstrations for each agent via a Bayesian surrogate model. We limit the number of bootstrapped demonstrations to 3 and instruction candidates to 10, per agent in 10 rounds. In topology optimization for all tasks, we search for 10 different topologies via rejection sampling. Along with topology optimization, each topology is evaluated on the validation set 3 times to stabilize the prediction. The optimized MAS is then reported on the held-out test set over three runs. We set model temperature $T$ at 0.7, maximum output tokens at 4096, and the 𝑡 in Softmax at 0.05 for sharpening the selection probability $p_{a}$ for each search dimension. We implement the same LLM backbone as both evaluator and optimizer in all phases.  

![](https://cdn-mineru.openxlab.org.cn/extract/480bcfb1-0272-4d7a-b502-2d4631527442/ac3d5916b7713c3ba518b64db5ab3f4c2f63860bbd3a33f05e95864064df6644.jpg)  
Figure 5 | Left: average performance per optimization stage of Mass over 8 evaluation tasks on Gemini 1.5 Pro. We compare Mass with a single agent (CoT) starting point as the reference and an APO baseline that optimizes over the single agent by MIPROv2 (Opsahl-Ong et al., 2024). Refer to App. §C for the detailed ablation per task. Right: a comparative ablation study on topology optimization (2TO) without pruning and without the former stage of prompt optimization (1PO) evaluated on HotpotQA.  

![](https://cdn-mineru.openxlab.org.cn/extract/480bcfb1-0272-4d7a-b502-2d4631527442/da2d955d6b5e8a4178331f988380940683040969ce40faf3942f04807c862ef9.jpg)  
Figure 6 | The optimization trajectories of Mass compared to automatic agent design baselines per validation round on DROP. We note that, as a distinct advantage of Mass, the optimization within stages (1) & (2) of Mass can be completely parallelized, whereas ADAS and AFlow are iterative algorithms that have to wait to propose new agents until finishing earlier trajectories.  

Main results. We present the main results of Mass compared to the baselines on the evaluation set in Table 1. Mass yields substantial gains over common forms of multi-agent systems, (e.g. selfconsistency, self-refine, and multi-agent debate), that scale up without optimizing prompts for agents in collaboration. Mass leads to high-performing MAS: $78.8\%$ and $74.3\%$ on average on Gemini 1.5 Pro and Flash, respectively, where we observe consistent improvements on Claude 3.5 Sonnet as reported in Table 4. By comparing Mass with state-of-the-art automatic agent design baselines, ADAS and AFlow, we first notice that ADAS only brings subtle gains even by already conditioning its metaagent generation based on the common forms of agents. The meta-agent keeps proposing complex topologies but without optimizing the prompt design. AFlow, on the other hand, demonstrates a competitive performance to Mass, especially on 2WikiMQA and HumanEval.  

We attribute the performance of AFlow to: 1) its ‘expansion’ phase that generates new nodes based on an error log that contrasts the predictions with the ground truth, which provides implicit textual gradients (Pryzant et al., 2023) to reflect on any formatting errors in prompt design; 2) a more refined search space within a pre-defined set of operators. Though AFlow draws similar inspirations on the importance of search space design as Mass, it still lacks a phase of prompt optimization to optimize its pre-defined operators properly, resulting in under-performance for MAS search results at MATH and MuSiQue. Different from these baselines, the consistent improvements brought by Mass highlight the importance of searching in both prompt and topology design space.  

Ablating optimization stages. To understand the incremental gain per Mass optimization stage, we provide a stage-by-stage ablation study in Fig. 5. We list the average performance of Mass from block-level to workflow-level optimization and compare it with a single agent APO baseline, where the block-level optimization performance indicates the best-performing building block $a\in{\mathcal{A}}$ after APO. First, we notice that there is a large gain, $6\%$ on average, between block-level optimization and single-agent optimization, showing that MAS benefits substantially from having its agents optimized inside the building block. In addition, going from Stage (1) to (2), another $3\%$ gain can be achieved by composing influential topologies while searching the optimal configurations. Here, we provide an additional ablation on conducting Stage (2) without prompt optimization beforehand or without search space pruning. Fig. 5 (right) shows that both of them are critical for effective search space exploration. Lastly, Mass obtains further gains $(\sim2\%)$ by conducting workflow-level prompt optimization on the best-found topology, which indicates that optimizing the prompts towards modeling the interdependence of agents is beneficial in the MAS design.  

![](https://cdn-mineru.openxlab.org.cn/extract/480bcfb1-0272-4d7a-b502-2d4631527442/f7d2a3ebc755c824f46aaa449a23cebb8ef61601d78a26894131570b77f52e8c.jpg)  
Figure 7 | A demonstration of the optimization trajectory of Mass on MATH. In (1) block-level optimization: multi-agent debate serves as the best-performing topology. In (2) workflow topology optimization, aggregating with more parallel agents outweighs the performance of agents in debate. Lastly, (3) workflow-level optimization discovers the optimal prompt conditioned on the best topology.  

Cost-effectiveness of Mass. We conduct analysis on the cost-effectiveness of Mass. In particular, we visualize the optimization trajectory of Mass as shown in Fig. 6. Mass’s trajectory demonstrates a steady trend of optimization that gradually improves the validation performance via interleaving the search towards better prompts and topologies. However, when it comes to automatic design baselines without explicit prompt optimization stages, AFlow is exposed to a larger variance in its optimization due to the nature of MCTS, whereas ADAS gets trapped in discovering over-complex topologies that appear to be less effective than the prompt design space. Overall, the optimization trajectory of Mass highlights the importance of optimizing in an effective design space, where interleaved optimization further resolves the complexity with more consecutive rewards. Following Sec. 2.1, Mass also demonstrated advanced token-effectiveness, which we refer to Fig. 9.  

Best-found MAS architectures & Design principles. We further inspect an example of optimized prompts and the trajectory of Mass in discovering more effective topologies in Fig. 7. The optimization starts from a zero-shot CoT agent, and soon Mass in Stage (1) identifies the high-performing topology in debate with its optimized prompt. However, as found in Stage (2), aggregating with more parallel agents actually outweighs the multi-agent debate. Workflow-level prompt optimization then leads to the best-performing predictor for aggregation. The overall optimization flow sheds light on our guidelines for building effective MAS: 1) optimizing individual agents properly is important before composing them into an MAS; 2) more effective MAS can be built by composing influential topologies; and 3) modeling the interdependence between agents is beneficial, and can be achieved via workflowlevel joint optimization.  

## 6. Conclusion  

We approach designing effective MAS by first conducting a thorough analysis of the massive design space, revealing the crucial role of prompts, and identifying an influential subset of search space. Building on these findings, we introduce Mass, a novel multi-stage optimization framework that searches within a pruned design space, interleaving prompt and topology optimization to efficiently generate high-performing MAS. Our experiments demonstrate that Mass-optimized MAS significantly outperforms existing manual and automated approaches across an extensive set of tasks. Finally, based on the optimized systems discovered by Mass, we extract valuable design principles to guide the development of future effective LLM-based MAS.  

### Acknowledgment  

We thank Jinsung Yoon and all other colleagues from Google Cloud AI Research for their valuable feedback.  

### References  

Anthropic. The claude 3 model family: Opus, sonnet, haiku. 2024.   
J. Austin, A. Odena, M. Nye, M. Bosma, H. Michalewski, D. Dohan, E. Jiang, C. Cai, M. Terry, Q. Le, et al. Program synthesis with large language models. arXiv preprint arXiv:2108.07732, 2021.   
Y. Bai, X. Lv, J. Zhang, H. Lyu, J. Tang, Z. Huang, Z. Du, X. Liu, A. Zeng, L. Hou, Y. Dong, J. Tang, and J. Li. LongBench: A bilingual, multitask benchmark for long context understanding. In L.-W. Ku, A. Martins, and V. Srikumar, editors, Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 3119–3137, Bangkok, Thailand, Aug. 2024. Association for Computational Linguistics. doi: 10.18653/v1/2024.acl-long.172. URL https://aclanthology.org/2024.acl-long.172/.   
J. Chen, S. Saha, and M. Bansal. ReConcile: Round-table conference improves reasoning via consensus among diverse LLMs. In L.-W. Ku, A. Martins, and V. Srikumar, editors, Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 7066–7085, Bangkok, Thailand, Aug. 2024a. Association for Computational Linguistics. doi: 10.18653/v1/2024.acl-long.381. URL https://aclanthology.org/2024.acl-long.381/.   
M. Chen, J. Tworek, H. Jun, Q. Yuan, H. P. D. O. Pinto, J. Kaplan, H. Edwards, Y. Burda, N. Joseph, G. Brockman, et al. Evaluating large language models trained on code. arXiv preprint arXiv:2107.03374, 2021.   
W. Chen, Y. Su, J. Zuo, C. Yang, C. Yuan, C.-M. Chan, H. Yu, Y. Lu, Y.-H. Hung, C. Qian, Y. Qin, X. Cong, R. Xie, Z. Liu, M. Sun, and J. Zhou. Agentverse: Facilitating multi-agent collaboration and exploring emergent behaviors. In The Twelfth International Conference on Learning Representations, 2024b. URL https://openreview.net/forum?id $\risingdotseq$ EHg5GDnyq1.   
X. Chen, R. Aksitov, U. Alon, J. Ren, K. Xiao, P. Yin, S. Prakash, C. Sutton, X. Wang, and D. Zhou. Universal self-consistency for large language models. In ICML 2024 Workshop on In-Context Learning, 2024c. URL https://openreview.net/forum?id $\risingdotseq$ LjsjHF7nAN.   
X. Chen, M. Lin, N. Schärli, and D. Zhou. Teaching large language models to self-debug. In The Twelfth International Conference on Learning Representations, 2024d. URL https://openreview. net/forum?id $\risingdotseq$ KuPixIqPiq.   
Y. Du, S. Li, A. Torralba, J. B. Tenenbaum, and I. Mordatch. Improving factuality and reasoning in language models through multiagent debate. In Forty-first International Conference on Machine Learning, ICML 2024, Vienna, Austria, July 21-27, 2024. OpenReview.net, 2024. URL https: //openreview.net/forum?id $\cdot^{=}$ zj7YuTE4t8.   
D. Dua, Y. Wang, P. Dasigi, G. Stanovsky, S. Singh, and M. Gardner. DROP: A reading comprehension benchmark requiring discrete reasoning over paragraphs. In J. Burstein, C. Doran, and T. Solorio, editors, Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pages 2368–2378, Minneapolis, Minnesota, June 2019. Association for Computational Linguistics. doi: 10.18653/v1/N19-1246. URL https://aclanthology.org/N19-1246/.   
S. Guo, C. Deng, Y. Wen, H. Chen, Y. Chang, and J. Wang. Ds-agent: Automated data science by empowering large language models with case-based reasoning, 2024. URL https://arxiv.org/ abs/2402.17453.   
D. Hendrycks, C. Burns, S. Kadavath, A. Arora, S. Basart, E. Tang, D. Song, and J. Steinhardt. Measuring mathematical problem solving with the math dataset. NeurIPS, 2021. URL https: //openreview.net/forum?id $\l=$ 7Bywt2mQsCe.   
X. Ho, A.-K. Duong Nguyen, S. Sugawara, and A. Aizawa. Constructing a multi-hop QA dataset for comprehensive evaluation of reasoning steps. In D. Scott, N. Bel, and C. Zong, editors, Proceedings of the 28th International Conference on Computational Linguistics, pages 6609–6625, Barcelona, Spain (Online), Dec. 2020. International Committee on Computational Linguistics. doi: 10.18653/ v1/2020.coling-main.580. URL https://aclanthology.org/2020.coling-main.580/.   
S. Hu, C. Lu, and J. Clune. Automated design of agentic systems. arXiv preprint arXiv:2408.08435, 2024a.   
X. Hu, Z. Zhao, S. Wei, Z. Chai, Q. Ma, G. Wang, X. Wang, J. Su, J. Xu, M. Zhu, Y. Cheng, J. Yuan, J. Li, K. Kuang, Y. Yang, H. Yang, and F. Wu. Infiagent-dabench: Evaluating agents on data analysis tasks, 2024b. URL https://arxiv.org/abs/2401.05507.   
N. Jain, K. Han, A. Gu, W.-D. Li, F. Yan, T. Zhang, S. Wang, A. Solar-Lezama, K. Sen, and I. Stoica. Livecodebench: Holistic and contamination free evaluation of large language models for code. arXiv preprint arXiv:2403.07974, 2024.   
C. E. Jimenez, J. Yang, A. Wettig, S. Yao, K. Pei, O. Press, and K. Narasimhan. Swe-bench: Can language models resolve real-world github issues? arXiv preprint arXiv:2310.06770, 2023.   
K. Kandasamy, W. Neiswanger, J. Schneider, B. Poczos, and E. P. Xing. Neural architecture search with bayesian optimisation and optimal transport. Advances in neural information processing systems, 31, 2018.   
A. Khan, J. Hughes, D. Valentine, L. Ruis, K. Sachan, A. Radhakrishnan, E. Grefenstette, S. R. Bowman, T. Rocktäschel, and E. Perez. Debating with more persuasive LLMs leads to more truthful answers. In Forty-first International Conference on Machine Learning, 2024. URL https://openreview. net/forum?id $\risingdotseq$ iLCZtl7FTa.   
O. Khattab, A. Singhvi, P. Maheshwari, Z. Zhang, K. Santhanam, S. V. A, S. Haq, A. Sharma, T. T. Joshi, H. Moazam, H. Miller, M. Zaharia, and C. Potts. DSPy: Compiling declarative language model calls into state-of-the-art pipelines. In The Twelfth International Conference on Learning Representations, 2024. URL https://openreview.net/forum?id=sY5N0zY5Od.   
T. Kojima, S. S. Gu, M. Reid, Y. Matsuo, and Y. Iwasawa. Large language models are zero-shot reasoners. Advances in neural information processing systems, 35:22199–22213, 2022.   
P. Lewis, E. Perez, A. Piktus, F. Petroni, V. Karpukhin, N. Goyal, H. Küttler, M. Lewis, W.-t. Yih, T. Rocktäschel, et al. Retrieval-augmented generation for knowledge-intensive nlp tasks. Advances in Neural Information Processing Systems, 33:9459–9474, 2020.   
J. Li, Q. Zhang, Y. Yu, Q. FU, and D. Ye. More agents is all you need. Transactions on Machine Learning Research, 2024a. ISSN 2835-8856. URL https://openreview.net/forum?id $\risingdotseq$ bgzUSZ8aeg.   
M. Li, S. Zhao, Q. Wang, K. Wang, Y. Zhou, S. Srivastava, C. Gokmen, T. Lee, L. E. Li, R. Zhang, W. Liu, P. Liang, L. Fei-Fei, J. Mao, and J. Wu. Embodied agent interface: Benchmarking llms for embodied decision making, 2025. URL https://arxiv.org/abs/2410.07166.   
Y. Li, Y. Du, J. Zhang, L. Hou, P. Grabowski, Y. Li, and E. Ie. Improving multi-agent debate with sparse communication topology. In Y. Al-Onaizan, M. Bansal, and Y.-N. Chen, editors, Findings of the Association for Computational Linguistics: EMNLP 2024, pages 7281–7294, Miami, Florida, USA, Nov. 2024b. Association for Computational Linguistics. doi: 10.18653/v1/2024.findings-emnlp.427. URL https://aclanthology.org/2024.findings-emnlp.427/.   
Z. Li, S. Xu, K. Mei, W. Hua, B. Rama, O. Raheja, H. Wang, H. Zhu, and Y. Zhang. Autoflow: Automated workflow generation for large language model agents. arXiv preprint arXiv:2407.12821, 2024c.   
T. Liang, Z. He, W. Jiao, X. Wang, Y. Wang, R. Wang, Y. Yang, S. Shi, and Z. Tu. Encouraging divergent thinking in large language models through multi-agent debate. In Y. Al-Onaizan, M. Bansal, and Y.-N. Chen, editors, Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pages 17889–17904, Miami, Florida, USA, Nov. 2024. Association for Computational Linguistics. doi: 10.18653/v1/2024.emnlp-main.992. URL https://aclanthology.org/2024. emnlp-main.992/.   
L. Lin, J. Fu, P. Liu, Q. Li, Y. Gong, J. Wan, F. Zhang, Z. Wang, D. Zhang, and K. Gai. Just ask one more time! self-agreement improves reasoning of language models in (almost) all scenarios. In L.-W. Ku, A. Martins, and V. Srikumar, editors, Findings of the Association for Computational Linguistics: ACL 2024, pages 3829–3852, Bangkok, Thailand, Aug. 2024. Association for Computational Linguistics. doi: 10.18653/v1/2024.findings-acl.230. URL https://aclanthology.org/2024. findings-acl.230/.   
F. Liu, N. AlDahoul, G. Eady, Y. Zaki, B. AlShebli, and T. Rahwan. Self-reflection outcome is sensitive to prompt construction. arXiv preprint arXiv:2406.10400, 2024a.   
H. Liu, K. Simonyan, and Y. Yang. Darts: Differentiable architecture search. arXiv preprint arXiv:1806.09055, 2018.   
Z. Liu, Y. Zhang, P. Li, Y. Liu, and D. Yang. A dynamic LLM-powered agent network for taskoriented agent collaboration. In First Conference on Language Modeling, 2024b. URL https: //openreview.net/forum?id $\cdot^{=}$ XII0Wp1XA9.   
A. Madaan, N. Tandon, P. Gupta, S. Hallinan, L. Gao, S. Wiegreffe, U. Alon, N. Dziri, S. Prabhumoye, Y. Yang, et al. Self-refine: Iterative refinement with self-feedback. Advances in Neural Information Processing Systems, 36, 2024.   
K. Opsahl-Ong, M. J. Ryan, J. Purtell, D. Broman, C. Potts, M. Zaharia, and O. Khattab. Optimizing instructions and demonstrations for multi-stage language model programs. In Y. AlOnaizan, M. Bansal, and Y.-N. Chen, editors, Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pages 9340–9366, Miami, Florida, USA, Nov. 2024. Association for Computational Linguistics. doi: 10.18653/v1/2024.emnlp-main.525. URL https://aclanthology.org/2024.emnlp-main.525/.   
L. Ouyang, J. Wu, X. Jiang, D. Almeida, C. Wainwright, P. Mishkin, C. Zhang, S. Agarwal, K. Slama, A. Ray, et al. Training language models to follow instructions with human feedback. Advances in neural information processing systems, 35:27730–27744, 2022.   
R. Pryzant, D. Iter, J. Li, Y. Lee, C. Zhu, and M. Zeng. Automatic prompt optimization with “gradient descent” and beam search. In H. Bouamor, J. Pino, and K. Bali, editors, Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 7957–7968, Singapore, Dec. 2023. Association for Computational Linguistics. doi: 10.18653/v1/2023.emnlp-main.494. URL https://aclanthology.org/2023.emnlp-main.494/.   
C. Qian, Z. Xie, Y. Wang, W. Liu, Y. Dang, Z. Du, W. Chen, C. Yang, Z. Liu, and M. Sun. Scaling large-language-model-based multi-agent collaboration. arXiv preprint arXiv:2406.07155, 2024.   
S. Qiao, N. Zhang, R. Fang, Y. Luo, W. Zhou, Y. Jiang, C. Lv, and H. Chen. AutoAct: Automatic agent learning from scratch for QA via self-planning. In L.-W. Ku, A. Martins, and V. Srikumar, editors, Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 3003–3021, Bangkok, Thailand, Aug. 2024. Association for Computational Linguistics. doi: 10.18653/v1/2024.acl-long.165. URL https://aclanthology.org/2024. acl-long.165/.   
M. Reid, N. Savinov, D. Teplyashin, D. Lepikhin, T. P. Lillicrap, J. Alayrac, R. Soricut, A. Lazaridou, O. Firat, J. Schrittwieser, I. Antonoglou, R. Anil, S. Borgeaud, A. M. Dai, K. Millican, E. Dyer, M. Glaese, T. Sottiaux, B. Lee, F. Viola, M. Reynolds, Y. Xu, J. Molloy, J. Chen, M. Isard, P. Barham, T. Hennigan, R. McIlroy, M. Johnson, J. Schalkwyk, E. Collins, E. Rutherford, E. Moreira, K. Ayoub, M. Goel, C. Meyer, G. Thornton, Z. Yang, H. Michalewski, Z. Abbas, N. Schucher, A. Anand, R. Ives, J. Keeling, K. Lenc, S. Haykal, S. Shakeri, P. Shyam, A. Chowdhery, R. Ring, S. Spencer, E. Sezener, and et al. Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context. CoRR, abs/2403.05530, 2024. doi: 10.48550/ARXIV.2403.05530. URL https: //doi.org/10.48550/arXiv.2403.05530.   
B. Ru, X. Wan, X. Dong, and M. Osborne. Interpretable neural architecture search via bayesian optimisation with weisfeiler-lehman kernels. International Conference on Learning Representations (ICLR), 2021.   
J. Saad-Falcon, A. G. Lafuente, S. Natarajan, N. Maru, H. Todorov, E. Guha, E. K. Buchanan, M. Chen, N. Guha, C. Ré, et al. Archon: An architecture search framework for inference-time techniques. arXiv preprint arXiv:2409.15254, 2024.   
Y. Shang, Y. Li, K. Zhao, L. Ma, J. Liu, F. Xu, and Y. Li. Agentsquare: Automatic llm agent search in modular design space. arXiv preprint arXiv:2410.06153, 2024.   
N. Shinn, F. Cassano, A. Gopinath, K. Narasimhan, and S. Yao. Reflexion: Language agents with verbal reinforcement learning. Advances in Neural Information Processing Systems, 36, 2024.   
A. Singh, A. Ehtesham, S. Kumar, and T. T. Khoei. Agentic retrieval-augmented generation: A survey on agentic rag. arXiv preprint arXiv:2501.09136, 2025.   
H. Su, R. Sun, J. Yoon, P. Yin, T. Yu, and S. Ö. Arık. Learn-by-interact: A data-centric framework for self-adaptive agents in realistic environments. arXiv preprint arXiv:2501.10893, 2025.   
H. Trivedi, N. Balasubramanian, T. Khot, and A. Sabharwal. MuSiQue: Multihop questions via singlehop question composition. Transactions of the Association for Computational Linguistics, 10:539–554, 2022. doi: 10.1162/tacl_a_00475. URL https://aclanthology.org/2022.tacl-1.31/.   
M. Verma, S. Bhambri, and S. Kambhampati. On the brittle foundations of react prompting for agentic large language models. arXiv preprint arXiv:2405.13966, 2024.   
X. Wan, B. Ru, P. M. Esperança, and Z. Li. On redundancy and diversity in cell-based neural architecture search. In International Conference on Learning Representations, 2022. URL https: //openreview.net/forum?id $\cdot^{=}$ rFJWoYoxrDB.   
X. Wan, R. Sun, H. Nakhost, and S. O. Arik. Teach better or show smarter? on instructions and exemplars in automatic prompt optimization. In The Thirty-eighth Annual Conference on Neural Information Processing Systems, 2024. URL https://openreview.net/forum?id $\risingdotseq$ IdtoJVWVnX.   
X. Wan, H. Zhou, R. Sun, H. Nakhost, K. Jiang, and S. Ö. Arık. From few to many: Self-improving manyshot reasoners through iterative optimization and generation. arXiv preprint arXiv:2502.00330, 2025.   
F. Wang, X. Wan, R. Sun, J. Chen, and S. Ö. Arık. Astute rag: Overcoming imperfect retrieval augmentation and knowledge conflicts for large language models. arXiv preprint arXiv:2410.07176, 2024a.   
J. Wang, J. Wang, B. Athiwaratkun, C. Zhang, and J. Zou. Mixture-of-agents enhances large language model capabilities. arXiv preprint arXiv:2406.04692, 2024b.   
Q. Wang, Z. Wang, Y. Su, H. Tong, and Y. Song. Rethinking the bounds of LLM reasoning: Are multi-agent discussions the key? In L.-W. Ku, A. Martins, and V. Srikumar, editors, Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 6106–6131, Bangkok, Thailand, Aug. 2024c. Association for Computational Linguistics. doi: 10.18653/v1/2024.acl-long.331. URL https://aclanthology.org/2024.acl-long.331/.   
X. Wang, J. Wei, D. Schuurmans, Q. V. Le, E. H. Chi, S. Narang, A. Chowdhery, and D. Zhou. Selfconsistency improves chain of thought reasoning in language models. In The Eleventh International Conference on Learning Representations, 2023. URL https://openreview.net/forum?id= 1PL1NIMMrw.   
C. White, M. Safari, R. Sukthanker, B. Ru, T. Elsken, A. Zela, D. Dey, and F. Hutter. Neural architecture search: Insights from 1000 papers. arXiv preprint arXiv:2301.08727, 2023.   
Q. Wu, G. Bansal, J. Zhang, Y. Wu, S. Zhang, E. Zhu, B. Li, L. Jiang, X. Zhang, and C. Wang. Autogen: Enabling next-gen llm applications via multi-agent conversation framework. arXiv preprint arXiv:2308.08155, 2023.   
C. S. Xia, Y. Deng, S. Dunn, and L. Zhang. Agentless: Demystifying llm-based software engineering agents. arXiv preprint arXiv:2407.01489, 2024.   
Z. Yang, P. Qi, S. Zhang, Y. Bengio, W. Cohen, R. Salakhutdinov, and C. D. Manning. HotpotQA: A dataset for diverse, explainable multi-hop question answering. In E. Riloff, D. Chiang, J. Hockenmaier, and J. Tsujii, editors, Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing, pages 2369–2380, Brussels, Belgium, Oct.-Nov. 2018. Association for Computational Linguistics. doi: 10.18653/v1/D18-1259. URL https://aclanthology.org/ D18-1259/.   
S. Yao, J. Zhao, D. Yu, N. Du, I. Shafran, K. R. Narasimhan, and Y. Cao. React: Synergizing reasoning and acting in language models. In The Eleventh International Conference on Learning Representations, 2023. URL https://openreview.net/forum?id $\risingdotseq$ WE_vluYUL-X.   
G. Zhang, Y. Yue, Z. Li, S. Yun, G. Wan, K. Wang, D. Cheng, J. X. Yu, and T. Chen. Cut the crap: An economical communication pipeline for llm-based multi-agent systems. arXiv preprint arXiv:2410.02506, 2024a.   
J. Zhang, J. Xiang, Z. Yu, F. Teng, X. Chen, J. Chen, M. Zhuge, X. Cheng, S. Hong, J. Wang, et al. Aflow: Automating agentic workflow generation. arXiv preprint arXiv:2410.10762, 2024b.   
J. Zhang, X. Xu, N. Zhang, R. Liu, B. Hooi, and S. Deng. Exploring collaboration mechanisms for LLM agents: A social psychology view. In L.-W. Ku, A. Martins, and V. Srikumar, editors, Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 14544–14607, Bangkok, Thailand, Aug. 2024c. Association for Computational Linguistics. doi: 10.18653/v1/2024.acl-long.782. URL https://aclanthology.org/2024.acl-long.782/.   
S. Zhang, J. Zhang, J. Liu, L. Song, C. Wang, R. Krishna, and Q. Wu. Offline training of language model agents with functions as learnable weights. In Forty-first International Conference on Machine Learning, 2024d. URL https://openreview.net/forum?id $\l=$ 2xbkWiEuR1.   
W. Zhang, K. Tang, H. Wu, M. Wang, Y. Shen, G. Hou, Z. Tan, P. Li, Y. Zhuang, and W. Lu. Agentpro: Learning to evolve via policy-level reflection and optimization. In L.-W. Ku, A. Martins, and V. Srikumar, editors, Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 5348–5375, Bangkok, Thailand, Aug. 2024e. Association for Computational Linguistics. doi: 10.18653/v1/2024.acl-long.292. URL https://aclanthology.org/2024.acl-long.292/.   
Y. Zhang, R. Sun, Y. Chen, T. Pfister, R. Zhang, and S. O. Arik. Chain of agents: Large language models collaborating on long-context tasks. In The Thirty-eighth Annual Conference on Neural Information Processing Systems, 2024f. URL https://openreview.net/forum?id $\cdot^{=}$ LuCLf4BJsr.   
H. Zhou, X. Wan, I. Vulić, and A. Korhonen. Survival of the most influential prompts: Efficient blackbox prompt search via clustering and pruning. In H. Bouamor, J. Pino, and K. Bali, editors, Findings of the Association for Computational Linguistics: EMNLP 2023, pages 13064–13077, Singapore, Dec. 2023. Association for Computational Linguistics. doi: 10.18653/v1/2023.findings-emnlp.870. URL https://aclanthology.org/2023.findings-emnlp.870/.   
H. Zhou, X. Wan, Y. Liu, N. Collier, I. Vulić, and A. Korhonen. Fairer preferences elicit improved human-aligned large language model judgments. In Y. Al-Onaizan, M. Bansal, and Y.-N. Chen, editors, Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pages 1241–1252, Miami, Florida, USA, Nov. 2024a. Association for Computational Linguistics. doi: 10.18653/v1/2024.emnlp-main.72. URL https://aclanthology.org/2024.emnlp-main. 72/.   
H. Zhou, X. Wan, L. Proleev, D. Mincu, J. Chen, K. A. Heller, and S. Roy. Batch calibration: Rethinking calibration for in-context learning and prompt engineering. In The Twelfth International Conference on Learning Representations, 2024b. URL https://openreview.net/forum?id $\risingdotseq$ L3FHMoKZcS.   
H. Zhou, X. Wan, I. Vulić, and A. Korhonen. AutoPEFT: Automatic configuration search for parameterefficient fine-tuning. Transactions of the Association for Computational Linguistics, 12:525–542, 2024c. doi: 10.1162/tacl_a_00662. URL https://aclanthology.org/2024.tacl-1.29/.  

W. Zhou, Y. Ou, S. Ding, L. Li, J. Wu, T. Wang, J. Chen, S. Wang, X. Xu, N. Zhang, et al. Symbolic learning enables self-evolving agents. arXiv preprint arXiv:2406.18532, 2024d. M. Zhuge, W. Wang, L. Kirsch, F. Faccio, D. Khizbullin, and J. Schmidhuber. GPTSwarm: Language agents as optimizable graphs. In Forty-first International Conference on Machine Learning, 2024. URL https://openreview.net/forum?id=uTC9AFXIhg.  

### A. Limitations and future work  

Mass is a multi-agent design meta-framework also orthogonal to prompt and topology optimizers. Mass has brought substantial improvements over a single agent design by searching in a customizable topology space. Though our proposed topology space has covered the vast majority of effective MAS designs, including serial, parallel, and mixture of connections, it is still likely that incorporating other topologies may further improve the final performance of Mass, which is complementary to the development of Mass. For instance, the debate topology proposed in Mass involves a fully-connected topology across agents. Recent work has been identifying the sparsity of agent communications (Li et al., 2024b; Zhang et al., 2024a), and pruning redundant communications may further enhance the overall efficiency of the strongest Mass-found design. Though the topology optimizer in Mass already traverses efficiently in the proposed topology space, incorporating more advanced search algorithms, such as the Bayes optimizer (Kandasamy et al., 2018; Ru et al., 2021), may further improve the sample efficiency of Mass when faces a more complex design space. Similarly, the sample efficiency of the prompt optimizer may be further enhanced by conditioning on textual feedback from error logs (Pryzant et al., 2023; Wan et al., 2024), which we will endeavor to explore in future work.  

### B. Implementation details  

#### B.1. Datasets  

In this work, we included the following dataset: 1) Hendryck’s MATH (Hendrycks et al., 2021) consisting challenging competition-level mathematics problems, and DROP (Dua et al., 2019) requires discrete and symbolic reasoning over paragraphs; 2) HotpotQA (Yang et al., 2018), MuSiQue (Trivedi et al., 2022), and 2WikiMultiHopQA (Ho et al., 2020) to evaluate on information seeking from long-context with agentic systems, which we report from standardized versions in LongBench (Bai et al., 2024); 3) MBPP (Austin et al., 2021), HumanEval (Chen et al., 2021), and LiveCodeBench (Jain et al., 2024) as well-established coding benchmarks. Regarding LiveCodeBench, we use the ‘test output prediction’ task as an agent cooperative task. In line with AFlow (Zhang et al., 2024b), we use the public test cases of MBPP and HumanEval for the executor to retrieve reliable external feedback signals.  

To save computation resources, we randomly sample a subset of the original validation and test splits to conduct all the experiments, where the specifications are reported in Table 2.  

Table 2 | The specification of evaluation tasks: dataset split, topology search space, and the Massoptimized MAS (on Gemini $1.5\mathrm{Pro}.$ )   


<html><body><table><tr><td>Task</td><td>Type</td><td>[Val|</td><td>|Test]</td><td>TopologySearchSpace</td><td>MASS</td></tr><tr><td>MATH</td><td>MathematicalReasoning</td><td>60</td><td>100</td><td>{Aggregate,Reflect,Debate}</td><td>{9, 0,0}</td></tr><tr><td>DROP</td><td>DiscreteReasoning</td><td>60</td><td>200</td><td>{Aggregate,Reflect,Debate}</td><td>{5,0,0}</td></tr><tr><td>HotpotQA</td><td>Long-context Understanding</td><td>50</td><td>100</td><td>{Summarize,Aggregate,Reflect,Debate}{</td><td>{0,5,0,1}</td></tr><tr><td>MuSiQue</td><td>Long-context Understanding</td><td>50</td><td>100</td><td>{Summarize,Aggregate,Reflect,Debate}</td><td>{0,3,0,2}</td></tr><tr><td>2WikiMQA</td><td>Long-context Understanding</td><td>50</td><td>100</td><td>{Summarize,Aggregate,Reflect,Debate}</td><td>{0,3,0,1}</td></tr><tr><td>MBPP</td><td>Coding</td><td>60</td><td>200</td><td>{Aggregate,Reflect,Debate,Executor}</td><td>{1, 4, 0, 1}</td></tr><tr><td>HumanEval</td><td>Coding</td><td>50</td><td>100</td><td>{Aggregate,Reflect,Debate,Executor}</td><td>{1,3,0,1}</td></tr><tr><td>LiveCodeBench</td><td>Coding: test output prediction</td><td>100</td><td>200</td><td>{Aggregate,Reflect,Debate,Executor}</td><td>{3, 1, 1, 1}</td></tr></table></body></html>  

Table 3 | The search dimension for each topology. The minimum topology defines the building block that Mass Stage (1) optimized.   


<html><body><table><tr><td>Topology</td><td>SearchSpace</td><td>Minimum Topology Building Block</td><td>Specification</td></tr><tr><td>Summarize</td><td>0,1,2,3,4}</td><td>{Summarizer,Predictor}</td><td>{1, 1}</td></tr><tr><td>Aggregate</td><td>{1,3,5,7,9}</td><td>{Predictor,Aggregator}</td><td>{3,1}</td></tr><tr><td>Reflect</td><td>{0,1,2,3,4}</td><td>{Predictor,Reflector}</td><td>{1, 1}</td></tr><tr><td>Debate</td><td>{0,1,2,3,4}</td><td>{Predictor,Debator}</td><td>{2,1}</td></tr><tr><td>Execute</td><td>{0, 1}</td><td>{Predictor,Executor,Reflector}</td><td>{1, 1, 1}</td></tr></table></body></html>  

#### B.2. Baselines  

In this section, we report the specifications of all our baselines. We note that for the baselines: CoT, SC, Self-Refine, and Multi-Agent Debate, we follow the prompts given in ADAS (Hu et al., 2024a).  

1) Chain-of-Thought (CoT) (Kojima et al., 2022). Direct chain-of-thought reasoning via zero-shot prompting: “Please think step by step and then solve the task."  

2) Self-Consistency (SC) (Wang et al., 2023). In self-consistency, we generate diverse chain-ofthought reasoning traces with a temperature of 0.8, followed by a rule-based majority vote that collects the most consistent answer. In Table 1, we report $\operatorname{SC@9}$ to provide a fair comparison across baselines.  

3) Self-Refine (Madaan et al., 2024): This baseline consists of one predictor that constantly takes feedback and a self-reflector that provides criticism. It involves a stop criterion whenever the self-reflector outputs “correct” in its prediction. We set the maximum number of rounds of reflections to 5, such that the worst case will involve 11 $(1+2\times5)$ calls.  

4) Multi-Agent Debate (Du et al., 2024; Liang et al., 2024). In this baseline, it involves 3 agents that conduct reasoning and debating for 3 rounds. The opinions along the rounds of debating are finally judged by an aggregator that makes the final prediction. Hence, it contains 10 $(3\times3+1)$ agents.  

5) Automated Design of Agentic Systems (ADAS) (Hu et al., 2024a). Consistent with our main experimental setups. We use Gemini 1.5 as both LLM optimizer and evaluator for reproducing all ADAS results. The generation of ADAS is conditioned on former evaluations of baselines, including CoT, SC, Self-Refine, and Multi-Agent Debate. We report ADAS with 30 rounds of search, and each round is evaluated on the validation set 3 times to stablize the prediction.  

6) AFlow (Zhang et al., 2024b). Automatic workflow design via Monte-Carto Tree Search over a set of pre-defined operators. Similar to ADAS, AFlow also relies on an LLM optimizer to generate new nodes and topologies expressed in codes. However, we find the meta-prompt of AFlow does not generalize to other LLM backbones. Consequently, we report AFlow with its original LLM optimizer by Claude 3.5 Sonnet, and reproduce experiments with Gemini $1.5\mathrm{Pro}$ as the LLM executor. Therefore, the comparison is not completely fair, and we treat the results from AFlow as a good reference. We note that the ‘-’ in Table 1 refers to out-of-time errors, where the LLM executor has been trapped in executing accidental scripts with infinite loops. We still endeavored to report most results from AFlow as shown in Table 1 & Fig. 6 with the default experimental setup from AFlow: 20 rounds, 5 runs of validation per round, and k at 3.  

![](https://cdn-mineru.openxlab.org.cn/extract/480bcfb1-0272-4d7a-b502-2d4631527442/472ddfe5f90bca2439d6a43bd14e6bb8354baf6ccdcd924149e7cddde7c3f06c.jpg)  
Figure 8 | Visualization of the topology building blocks and best Mass-discovered topologies from Gemini $1.5\mathrm{Pro}$ .  

#### B.3. Mass: Multi-Agent System Search  

In this section, we provide additional details for Mass. The topology search space for each task is defined in Table 2. In addition, for Stage (1) block-level prompt optimization, the specification of the building block is defined in Table 3. We provide the visualization of both the minimum building blocks and the optimized topology in Fig. 8. We refer the reader to App. $\S\mathrm{D}\&\S\mathrm{E}$ for the prompt templates we used to define each type of agent and the best prompts discovered.  

### C. Additional experiments  

Table 4 | Results on the evaluation set with Claude 3.5 Sonnet. We keep the same experimental setup as Table 1. Since Claude 3.5 Sonnet does not support the same context window as Gemini, we report the standard HotpotQA instead of the LongBench. As we transfer the prompt template for each agent from Gemini to Claude, it is noticeable that the basic topology on some tasks may result in severe degradation of performance, and Mass successfully recovers the performance and brings significant improvements over the initial agent.  

<html><body><table><tr><td colspan="8">Claude-3.5-Sonnet</td></tr><tr><td>Task</td><td colspan="2">Reasoning</td><td>Multi-hop</td><td colspan="3">Coding</td><td></td></tr><tr><td>Method</td><td>MATH</td><td>DROP</td><td>HotpotQA</td><td>MBPP</td><td>HumanEval</td><td>LCB</td><td>Avg.</td></tr><tr><td>CoT</td><td>57.330.94</td><td>55.520.42</td><td>23.561.52</td><td>67.501.47</td><td>88.671.70</td><td>72.672.39</td><td>60.21</td></tr><tr><td>Self-Consistency</td><td>61.671.89</td><td>57.860.45</td><td>25.690.44</td><td>69.170.62</td><td>90.000.82</td><td>72.672.39</td><td>62.84</td></tr><tr><td>Self-Refine</td><td>57.001.63</td><td>56.260.56</td><td>23.572.56</td><td>68.000.82</td><td>87.001.41</td><td>49.331.65</td><td>56.86</td></tr><tr><td>Multi-Agent Debate</td><td>45.003.74</td><td>26.620.11</td><td>31.413.30</td><td>00.000.00</td><td>84.333.30</td><td>72.821.84</td><td>43.36</td></tr><tr><td>MASS</td><td>63.000.00</td><td>68.930.38</td><td>66.980.99</td><td>68.830.62</td><td>93.000.82</td><td>73.731.43</td><td>72.43</td></tr></table></body></html>  

Table 5 | The detailed ablation results per optimization stage of Mass. Practical gains can be obtained by further conducting workflow-level prompt optimization (3PO) on the best-found topology.   
MATH (gemini-1.5-pro-002)   


<html><body><table><tr><td colspan="10">Gemini-1.5-pro-002</td></tr><tr><td>Task</td><td colspan="2">Reasoning</td><td colspan="3">Multi-hop Long-context</td><td colspan="3">Coding</td><td></td></tr><tr><td>Method</td><td>MATH</td><td>DROP</td><td>HotpotQA</td><td>MuSiQue</td><td>2WikiMQA</td><td>MBPP</td><td>HumanEval</td><td>LCB</td><td>Avg.</td></tr><tr><td>Base Agent</td><td>62.330.94</td><td>71.650.61</td><td>56.961.26</td><td>43.320.13</td><td>49.200.61</td><td>68.830.85</td><td>89.331.70</td><td>66.332.09</td><td>63.54</td></tr><tr><td>+ APO</td><td>79.331.89</td><td>77.510.38</td><td>59.720.00</td><td>43.970.00</td><td>61.490.24</td><td>67.001.08</td><td>86.331.25</td><td>68.501.22</td><td>67.44</td></tr><tr><td>+ 1PO</td><td>80.000.00</td><td>86.450.90</td><td>62.521.86</td><td>48.860.61</td><td>67.400.58</td><td>80.331.25</td><td>91.671.25</td><td>76.000.00</td><td>74.56</td></tr><tr><td>+ 2TO</td><td>83.001.63</td><td>86.751.32</td><td>65.221.34</td><td>52.610.52</td><td>72.820.86</td><td>85.001.08</td><td>92.000.82</td><td>81.330.00</td><td>77.55</td></tr><tr><td>+ 3PO</td><td>84.670.47</td><td>90.520.64</td><td>69.911.11</td><td>51.400.42</td><td>73.340.67</td><td>86.500.41</td><td>91.670.47</td><td>82.330.85</td><td>78.40</td></tr></table></body></html>  

![](https://cdn-mineru.openxlab.org.cn/extract/480bcfb1-0272-4d7a-b502-2d4631527442/e4be62710e00f1cc65705b36487b3eba93c9afde19e7d5964bfc65d094b623e9.jpg)  
Figure 9 | The Pareto-front of Mass-optimized designs compared to multi-agent baselines. Total tokens include both inference input tokens and output tokens. Additional multi-agent baselines from ADAS (Hu et al., 2024a) and two best-found ADAS designs are included.  

### D. Prompt template  

We provide all prompt templates we used for defining the Mass search space. We use $<>$ to enclose texts that have been skipped for presentation purposes. We follow the DSPy (Khattab et al., 2024) in constructing these agentic templates.  

The general template for instruction, exemplar, and input/output fields:  

1 2 <Instruction > 3 4 5 6 Follow the following format . 7 8 Input : \${ Input } 9 10 Output : \${ output } 11 12 13  

14 <example_1 >   
15   
16   
17   
18 Input : <Input >   
19   
20 Output : <output >  

#  

MATH:   
1 Predictor :   
2   
3 Let ’s think step by step .   
5 Question : \${ question }   
6 Reasoning : Let ’s think step by step in order to \${ produce the answer }. We ...   
7 Answer : \${ answer }   
8   
9   
10 Reflector :   
11   
12 Please review the answer above and criticize on where might be wrong . If you are   
absolutely sure it is correct , output ’True ’ in ’correctness ’.   
13   
14   
15 Question : \${ question }   
16 Text : \${ text }   
17 Reasoning : Let ’s think step by step in order to \${ produce the correctness }. We ...   
18 Feedback : \${ feedback }   
19 Correctness : True / False indicating if answer is correct given the question .   
20   
21   
22 Refiner :   
23   
24 Given previous attempts and feedback , carefully consider where you could go wrong in   
your latest attempt . Using insights from previous attempts , try to solve the   
task better . Show your final answer bracketed between <answer > and </answer > at   
the end.   
25   
26   
27 Question : \${ question }   
28 Previous answer : \${ previous_answer }   
29 Reflection : \${ reflection }   
30 Correctness : \${ correctness }   
31 Thinking : \${ thinking }   
32 Answer : \${ answer }   
33   
34   
35   
36 Debator :   
37   
38 These are the solutions to the question from other agents . Examine the solutions   
from other agents in your rationale , finish by giving an updated answer . Show   
your final answer bracketed between <answer > and </answer > at the end.   
39   
40   
41 Question : \${ question }   
42 Solutions : the solutions to the question from other agents   
43 Reasoning : Let ’s think step by step in order to \${ Examine the solutions from other   
agents }. We ...   
44 Answer : The updated answer for the question . Do not repeat Answer :  

### DROP:  

1 Predictor :   
2 Please think step by step and then solve the task . # Your Task :   
4 Please answer the following question based on the given context .   
5 Question : \${ question } Context : \${ context }   
8 Thinking : \${ thinking }  

9 Answer : Directly answer the question . Keep it very concise .   
10   
11   
12 Reflector :   
13   
14 Verify that the answer is based on the provided context . Give your reflection in the   
rationale .   
15   
16   
17 Question : \${ question }   
18 Context : \${ context }   
19 Text : \${ text }   
20 Reasoning : Let ’s think step by step in order to \${ produce the correctness }. We ...   
21 Correctness : True / False indicating if answer is correct given the observations and   
question .   
22   
23   
24 Refiner :   
25   
26 Please think step by step and then solve the task . # Your Task :   
27 Based on the reflection , correctness of the previous answer , and the context again ,   
give an updated answer .   
28   
29   
30 Question : \${ question }   
31 Context : \${ context }   
32 Previous answer : \${ previous_answer }   
33 Reflection : \${ reflection }   
34 Correctness : \${ correctness }   
35 Thinking : \${ thinking }   
36 Answer : Directly answer the question . Keep it very concise .   
37   
38   
39   
40 Debator :   
41   
42 These are the solutions to the question from other agents . Based on the context ,   
examine the solutions from other agents in your rationale , finish by giving an   
updated answer .   
43   
44   
45 Question : \${ question }   
46 Context : \${ context }   
47 Solutions : the solutions to the question from other agents   
48 Reasoning : Let ’s think step by step in order to \${ Examine the solutions from other   
agents }. We ...   
49 Answer : The updated answer for the question . Do not repeat Answer :  

### HotpotQA, MuSiQue, and 2WikiMQA:  

1 Predictor : 2 3 Answer the question with information based on the context . Only return the answer as your output . 5 Question : \${ question } 6 Context : \${ context } 7 Answer : Only give me the answer . Do not output any other words . 8 9   
10 Summarizer :   
11   
12 Based on the question , retrieve relevant information from context that is ONLY helpful in answering the question . Include all key information . Do not repeat context .   
13   
14 Question : \${ question }   
15 Context : \${ context }   
16 Summary : Only generate the summary . Start with Summary :   
17   
18  

19 Reflector :   
20   
21 Verify that the answer is based on the provided context .   
22   
23   
24 Question : \${ question }   
25 Context : \${ context }   
26 Text : \${ text }   
27 Reasoning : Let ’s think step by step in order to \${ produce the correctness }. We ...   
28 Correctness : True / False indicating if answer is correct given the observations and   
question .   
29   
30   
31   
32 Debator :   
33   
34 These are the solutions to the question from other agents . Based on the context ,   
examine the solutions from other agents in your rationale , finish by giving an   
updated answer .   
35   
36   
37 Question : \${ question }   
38 Context : \${ context }   
39 Solutions : the solutions to the question from other agents   
40 Reasoning : Let ’s think step by step in order to \${ Examine the solutions from other   
agents }. We ...   
41 Answer : The updated answer for the question . Do not repeat Answer :  

### MBPP:  

1 Predictor : 2   
3 Let ’s think step by step . Provide a complete and correct code implementation in python .   
5 Question : \${ question } 6 Thinking : \${ thinking }   
7 Answer : Only the code implementation . Do not include example usage or explainations .   
8 9   
10 Reflector :   
11   
12 Please determine the correctness of the solution in passing all test cases . If it fails , based on the error message and trackback , think step by step , carefully propose an updated solution in the answer output with a correct code implementation in python .   
13   
14   
15 Question : \${ question }   
16 Previous solution : \${ previous_solution }   
17 Traceback : It contains the test cases , execution results , and ground truth . If there is an error , the relevant traceback is given .   
18 Correctness : ’True /False ’ based on the correctness of executive feedback . If there is an error message , output ’False ’   
19 Thinking : \${ thinking }   
20 Answer : \${ answer }   
21   
22   
23   
24 Debator :   
25   
26 These are the solutions to the question from other agents . Examine the solutions from other agents in your rationale , finish by giving an updated answer . Let ’s think step by step . Provide a complete and correct code implementation in python   
27   
28   
29 Question : \${ question }   
30 Solutions : the solutions to the question from other agents   
31 Reasoning : Let ’s think step by step in order to \${ Examine the solutions from other agents }. We ..  

### HumanEval:  

1 Predictor :   
2   
3 Let ’s think step by step . Provide a complete and correct code implementation in python . 4   
5 Question : \${ question }   
6 Thinking : \${ thinking }   
7 Answer : \${ answer }   
8 9   
10 Reflector :   
11   
12 Please determine the correctness of the solution in passing all test cases . If it fails , based on the error message and trackback , think step by step , carefully propose an updated solution in the answer output with a correct code implementation in python .   
13   
14   
15 Question : \${ question }   
16 Previous solution : \${ previous_solution }   
17 Traceback : \${ traceback }   
18 Thinking : \${ thinking }   
19 Answer : \${ answer }   
20   
21   
22   
23 Debator :   
24   
25 These are the solutions to the question from other agents . Examine the solutions from other agents in your rationale , finish by giving an updated answer . Let ’s think step by step . Provide a complete and correct code implementation in python   
26   
27   
28 Question : \${ question }   
29 Solutions : the solutions to the question from other agents   
30 Reasoning : Let ’s think step by step in order to \${ Examine the solutions from other agents }. We ...   
31 Answer : \${ answer }  

### LiveCodeBench:  

1 Predictor :   
2   
3 You are a helpful programming assistant and an expert Python programmer . The user has written a input for the testcase . Think step by step. You will generate the code based on the problem requirepement . You will calculate the output of the testcase and write the whole assertion statement in the markdown code block with the correct output .   
5 Question : \${ question }   
6 Thinking : \${ thinking }   
7 Code : \${ code }   
8 Answer : complete the testcase with assertion .   
9   
10   
11 Reflector :   
12   
13 If there is an executive output in the traceback , parse the output into an assertion in the answer given the executive output .   
14   
15   
16 Question : \${ question }   
17 Previous solution : \${ previous_solution }   
18 Traceback : It contains the test cases , execution results , and ground truth . If there is an error , the relevant traceback is given .   
19 Correctness : ’True /False ’ based on the correctness of executive feedback . If there is an error message , output ’False ’  

20 Thinking : \${ thinking }   
21 Answer : \${ answer }   
22   
23   
24   
25 Debator :   
26   
27 These are the solutions to the question from other agents . Examine the solutions   
from other agents in your rationale , finish by giving an updated answer .   
28   
29   
30 Question : \${ question }   
31 Solutions : the solutions to the question from other agents   
32 Reasoning : Let ’s think step by step in order to \${ Examine the solutions from other   
agents }. We ...   
33 Answer : assert \${ function ( input )} $==$ { executive_output }  

### E. Best Prompts Discovered  

In this section, we show a few typical prompts that have been discovered by Mass. Similarly, we indicate $<>$ that skips some comments.  

#### MATH:  

1 Predictor :   
2   
3 Let ’s think step by step to solve the given problem . Clearly explain your reasoning process , showing all intermediate calculations and justifications . Express your final answer as a single numerical value or simplified expression enclosed within <answer ></ answer > tags . Avoid extraneous text or explanations outside of the core reasoning and final answer . 5   
6   
7 Follow the following format .   
8   
9 Question : \${ question }   
10 Reasoning : Let ’s think step by step in order to \${ produce the answer }. We ...   
11 Answer : \${ answer }   
12   
13   
14   
15 Question : What is the average of the two smallest positive integer solutions to the congruence \$\$14u \\ equiv 46 \\ pmod {100}\~? \$\$   
16 Reasoning : Let ’s think step by step in order to find the average of the two smallest positive integer solutions to the congruence $\$143$ \\ equiv 46 \\ pmod $\{100\}\$8$ . We are looking for solutions to $\$143$ \\ equiv 46 \\ pmod $\{100\}\Phi$ . We can divide the congruence by 2 to get $\$71$ \\ equiv 23 \\ pmod {50} \$. We want to find the inverse of 7 modulo 50. We use the Euclidean Algorithm : \\ begin { align \*} 50 &= 7 \\ cdot 7 $^+$ 1 \\\\ 1 $\&=$ 50 - 7 \\ cdot 7 \\ end { align \*} Thus , \$7 ( -7) \\ equiv 1 \\ pmod $\{50\}\$8$ , so $\$7$ \\ equiv -7 \\ equiv 43 \\ pmod {50} \$. Multiplying $\$71$ \\ equiv 23 \\ pmod $\{50\}\$8$ by 43 gives \$u \\ equiv 23 \\ cdot 43 \\ pmod {50} \$. $\$23$ \\ cdot $43=989=50$ \\ cdot $19+39\$3$ , so $\$23$ \\ cdot 43 \\ equiv 39 \\ pmod $\{50\}\$8$ . Therefore , $\$1$ \\ equiv 39 \\ pmod $\{50\}\$8$ . The two smallest positive integer solutions are $\$112393$ and $\$1$ $\begin{array}{r l}{=}&{{}39+50=89\Phi}\end{array}$ . The average of these two solutions is $\$123,456$ c $\{128\}\{2\}=64\Phi$ .   
17 Answer : 64   
18   
19  

In the following prompts, interestingly, we observe that including the data summary, task demonstrations, and past instructions that have been used in MIPRO (Opsahl-Ong et al., 2024) to propose new candidates actually improves the final performance. Hence, we keep these prompts that lead to strong task performance.  

DROP: 1 Predictor :  

2 3 This dataset is designed for extractive question answering , focusing on retrieving concise , factual answers from short texts . Many questions involve extracting numerical information and performing simple calculations , suggesting applications in areas like sports analytics or financial data analysis . However , the dataset ’s Western cultural bias and lack of complex reasoning questions limit its generalizability and real - world applicability . 4 5 TASK DEMO (S): 6 <example_1 > 7 Question : How many more points did the Spurs win by in Game 4 against the Mavericks ? 8 9 Context :   
10 The Mavericks finished 49 -33 , one game ahead of Phoenix for the eighth and final playoff spot , which meant that they would once again have to face their in - state rivals , the San Antonio Spurs , who were the top seed in the Western Conference with a 62 -20 record . In Game 1 in San Antonio , Dallas had an 81 -71 lead in the fourth quarter , but the Spurs rallied back and took Game 1, 85 -90. However , the Mavs forced 22 turnovers in Game 2 to rout the Spurs 113 -92 , splitting the first two games before the series went to Dallas . In Game 3, Manu Gin\ u00f3bili hit a shot that put the Spurs up 108 -106 with 1.7 seconds left , but a buzzer - beater by Vince Carter gave the Mavs the victory , putting them up 2-1 in the series . The Spurs took Game 4 in Dallas 93 -89 despite a late Dallas comeback after the Spurs at one point had a 20- point lead and later won Game 5 at home , 109 -103 , giving them a 3-2 series lead. The Mavs avoided elimination in Game 6 at home by rallying in the fourth quarter , winning 111 -113. Game 7 was on the Spurs home court , and the Spurs beat the Mavericks 119 -96 , putting an end to the Mavericks season . 11   
12 Thinking :   
13 The Spurs scored 93 points in Game 4. The Mavericks scored 89 points in Game 4. The difference is $93-89=4$ .   
14 Answer : 4 15 16 17 BASIC INSTRUCTION : 18 ‘‘‘   
19 You are a highly specialized AI tasked with extracting critical numerical information for an urgent news report . A live broadcast is relying on your accuracy and speed . Think step -by -step , focusing on the numerical information provided in the context . Then , answer the question concisely with the extracted numerical answer . Failure to provide the correct numerical information will result in the broadcast being interrupted . 20   
21 Question : { question } 22 Context : { context } 23 ‘‘‘   
24 25 TIP: Keep the instruction clear and concise . 26 27 PROPOSED INSTRUCTION : 28 29 ‘‘‘   
30 Extract the numerical answer to the following question . Show your reasoning by identifying the relevant numbers from the provided context and performing any necessary calculations . Respond with only the final numerical answer .   
31   
32 Question : { question } 33 Context : { context }   
34 ‘‘‘ HotpotQA:   
1 Predictor :   
2   
3 This multi - passage question answering dataset focuses on complex questions requiring synthesis of information from multiple Wikipedia - like sources , often involving named entities and temporal reasoning . It emphasizes integrating information , handling ambiguity , and leveraging real - world knowledge , posing a significant challenge for models relying solely on provided text . The dataset appears well - suited for evaluating advanced language models ’ reasoning abilities across  

diverse domains and varying complexity levels . 5 TASK DEMO (S): 6 Question : The actor that plays Phileas Fogg in \" Around the World in 80 Days \", co - starred with Gary Cooper in a 1939 Goldwyn Productions film based on a novel by what author ? 7 Context : Provided in prompt 8 Answer : Charles L. Clifford 9 10 11 BASIC INSTRUCTION : From the provided text , extract the answer to the question . Output \* only \* the answer . 12 13 TIP: Keep the instruction clear and concise . Emphasize reliance \*only \* on the provided text . 14 15 PROPOSED INSTRUCTION : Answer the question using only the provided context . Do not use external knowledge . 16 17 18 <example_1 > 19 20 21 22 Debator : 23 24 This multi - passage question answering dataset focuses on complex questions requiring synthesis of information from multiple Wikipedia - like sources , often involving named entities and temporal reasoning . It emphasizes integrating information , handling ambiguity , and leveraging real - world knowledge , posing a significant challenge for models relying solely on provided text . The dataset appears well - suited for evaluating advanced language models ’ reasoning abilities across diverse domains and varying complexity levels . 25 26 TASK DEMO (S): 27 Provided above . 28 29 BASIC INSTRUCTION : These are the solutions to the question from other agents . Based on the context , examine the solutions from other agents in your rationale , finish by giving an updated answer . 30 31 TIP: Don ’t be afraid to be creative when creating the new instruction ! 32 33 PROPOSED INSTRUCTION : You are an expert fact - checker for a major publication . Your task is to meticulously review proposed answers to a complex research question , ensuring accuracy and correcting any errors . You are provided with the original question , multiple context passages from credible sources , and several proposed answers from different research assistants . Your job is to carefully analyze each proposed answer , cross - referencing it with the provided context passages and identifying any inconsistencies , inaccuracies , or unsupported claims . 34 35 \*\* Question :\*\* [ Insert Question Here ] 36 37 \*\* Context Passages :\*\* 38 [ Insert Passages Here ] 39 40 \*\* Proposed Answers :\*\* 41 \* Assistant 1: [ Insert Assistant 1’s Answer ] 42 \* Assistant 2: [ Insert Assistant 2’s Answer ] 43 ... 44 \* Assistant N: [ Insert Assistant N’s Answer ] 45 46 47 \*\* Instructions :\*\* 48 49 1. \*\* Fact - Check & Analyze :\*\* Evaluate each proposed answer individually . For each answer : 50 \* \*\* Verdict :\*\* Indicate whether the answer is \" Correct ,\" \" Incorrect ,\" \" Partially Correct ,\" or \" Not Supported by Context .\" 51 \* \*\* Evidence :\*\* Provide specific quotes and passage numbers from the context to  

support your verdict . Explain how the evidence supports or refutes the proposed answer . Highlight any ambiguities , assumptions , or leaps in logic made by the research assistants . 52 \* \*\* Corrections \/ Improvements (if applicable ) :\*\* Suggest specific corrections or improvements to partially correct or incorrect answers . Explain how these changes align with the context . 53 54 2. \*\* Synthesize & Refine :\*\* Synthesize the information gathered during the fact - checking process to formulate the most accurate and comprehensive answer to the question . This may involve : 55 \* Selecting the most accurate proposed answer . 56 \* Combining elements from multiple proposed answers . 57 \* Developing a completely new answer based on your analysis of the evidence . 58 59 3. \*\* Final Answer :\*\* Clearly state your final , verified answer to the question . 60 61 4. \*\* Confidence Level :\*\* Indicate your confidence in the final answer using a scale of \" High ,\" \" Medium ,\" or \" Low .\" Briefly explain the factors influencing your confidence level . 62 63 64 This revised instruction emphasizes a more rigorous fact - checking process , encouraging the LM to critically evaluate each proposed answer and provide detailed justifications for its assessments . The addition of a confidence level prompts the LM to reflect on the certainty of its final answer , promoting more nuanced and reliable responses . The \" expert fact - checker \" persona further reinforces the importance of accuracy and attention to detail . 65 66 67 <example_1 > 68 <example_2 >  

#### MBPP:  

1 Predictor : 2   
3 You are a highly skilled Python programmer tasked with generating a correct and efficient Python function based on the given natural language problem description . Think step -by -step , outlining your reasoning process before presenting the code solution . Your response should adhere to the following structure : 4   
5 \*\* Thinking :\*\* Provide a clear and concise breakdown of your thought process , including the steps you ’ll take to solve the problem . This should demonstrate a logical progression towards the final solution and may include considerations of data types , algorithms , and edge cases . For example : 6   
7 1. Identify the input data type and expected output .   
8 2. Determine the core logic or algorithm required .   
9 3. Consider potential edge cases or special scenarios .   
10 4. Outline the steps for implementing the solution in Python .   
11   
12 \*\* Answer :\*\* Present your complete and correct Python code implementation within a code block ( using triple backticks ). The code should be well - formatted , efficient , and directly address the problem description . Ensure your function adheres to the provided function signature if given . For example :   
13   
14 ‘‘‘ python   
15 def function_name ( input_arguments ):   
16 # Code implementation here   
17 # ...   
18 return output   
19 ‘‘‘   
20   
21 Focus on producing functional code that accurately solves the problem . Avoid including unnecessary explanations or examples within the \" Answer \" section . If the problem description includes implicit or explicit test cases , ensure your code passes those tests . Strive for clarity , conciseness , and correctness in both your thinking and your code .   
22   
23   
24 <example_1 >   
25 <example_2 >   
26 <example_3 >   
27   
28   
29 Reflector :   
30   
31 This dataset is designed for Python code generation , translating natural language problem descriptions into simple functions and their corresponding test cases . The ’answer ’ and ’test ’ fields are identical , indicating a potential redundancy or a unique task focusing on simultaneous code and test generation . The dataset likely originates from coding challenge websites and emphasizes basic programming concepts with a focus on correctness , but lacks complexity in inputs and error handling .   
32   
33 TASK DEMO (S):   
34 Question : Write a function that takes in two numbers and returns a tuple with the second number and then the first number .   
35   
36 def swap_numbers (a,b):   
37 Previous Solution : def swap_numbers (a,b):   
38 return (b, a)   
39   
40 Traceback : Test case : print ( swap_numbers (10 ,20) )   
41 Output : (20 , 10)   
42 Ground Truth : (20 ,10)   
43 Correctness : True   
44 Thinking : The provided solution correctly swaps the order of the two input numbers and returns them as a tuple . The test case demonstrates this functionality , and   
45 Answer : ‘‘‘ python the output matches the ground truth . Therefore , no changes are required .   
46 def swap_numbers (a,b):   
47 return (b, a)   
48 ‘‘‘   
49 <example_2 >   
50 <example_3 >   
51   
52   
53 BASIC INSTRUCTION : Please determine the correctness of the solution in passing all test cases . If it fails , based on the error message and trackback , think step by step , carefully propose an updated solution in the answer output with a correct code implementation in python .   
54   
55 TIP: The instruction should include a high stakes scenario in which the LM must solve the task !   
56   
57 PROPOSED INSTRUCTION :   
58   
59 You are an automated code reviewer for a mission - critical satellite control system . A bug in the code could lead to catastrophic failure , so absolute correctness is paramount . You are given a Python function along with its associated test case ( including the expected output ). Analyze the provided   
60   
61 <example_1 >   
62 <example_2 >  