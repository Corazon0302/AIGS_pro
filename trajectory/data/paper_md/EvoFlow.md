# EvoFlow: Evolving Diverse Agentic Workflows On The Fly  

Guibin Zhang \* 1 Kaijie Chen \* 1 Guancheng Wan 2 Heng Chang Hong Cheng 4 Kun Wang 5 Shuyue Hu 6 Lei Bai 6  

# Abstract  

The past two years have witnessed the evolution of large language model (LLM)-based multi-agent systems from labor-intensive manual design to partial automation (e.g., prompt engineering, communication topology) and eventually to fully automated design. However, existing agentic automation pipelines often lack LLM heterogeneity and focus on single-objective performance optimization, limiting their potential to combine weaker models for more customized and cost-effective solutions. To address this challenge, we propose EvoFlow, a niching evolutionary algorithm-based framework to automatically search a population of heterogeneous and complexity-adaptive agentic workflows, rather than a single homogeneous, complex workflow. Technically, EvoFlow performs (1) tag-based retrieval to extract parent workflows from an agentic population, evolves new workflows through (2) crossover and (3) mutation, and employs (4) niching-based selection to maintain population diversity and quality. Extensive evaluations across seven benchmarks demonstrate that EvoFlow is: (I) diverse, evolving a population of workflows ranging from simple I/O tasks to complex multi-turn interactions; $(\mathbf{II})$ high-performing, outperforming previous handcrafted and automated workflows by $1.23\%\sim29.86\%$ ; (III) economical, surpassing powerful o1-preview at $12.4\%$ of its inference cost using weaker open-source models. The code will be available at https://github.com/ bingreeky/EvoFlow.  

# 1. Introduction  

Large Language Model (LLM)-based agents (Richards & et al., 2023; Nakajima, 2023; Reworkd, 2023) have exhibited remarkable capabilities across a wide spectrum of tasks, including question answering (Zhu et al., 2024a), data analysis (Hong et al., 2024; Li et al., 2024), decisionmaking (Song et al., 2023), code generation (Shinn et al., 2023), video gaming (Wang et al., 2023), and autonomous driving (Jin et al., 2023), among others. Recent advancements further highlight that integrating single agents into agentic workflows, i.e., structured sequences of LLM-based agent interactions, can surpass the cognitive and functional limitations of individual agents (Du et al., 2023; Liang et al., 2023; Wang et al., 2023b; Jiang et al., 2023; Shinn et al., 2023; Zheng et al., 2023a; Wu et al., 2023; Zhang et al., 2024a), thereby exhibiting human-esque collaborative intelligence in multi-agent systems (Zhang et al., 2023).  

![](https://cdn-mineru.openxlab.org.cn/extract/f6fc2303-638d-4bb4-901b-d70af57f7b1c/8d776f3575abdda13305594b7840504e1f29727233d6d15f757d5792b0b78bea.jpg)  
Figure 1. Paradigm comparison. Baseline methods seek a “onesize-fits-all” complex homogenoues workflow, while EvoFlow optimizes a Pareto set of diverse, heterogenous workflows.  

The progress of machine learning consistently reveals a recurring pattern: manually crafted artifacts are often replaced by learnable and optimizable ones (Tyson & Zysman, 2022; Clune, 2019). For natural language processing (NLP), this is exemplified by the replacement of hand-designed representations like Bag-of-Words (Mikolov et al., 2013) with learned embeddings like GloVe (Pennington et al., 2014) and BERT (Devlin, 2018). Similarly, agentic systems have experienced a rapid transition from manual to automated over the past two years: Early attempts, e.g., CAMEL (Li et al., 2023), AutoGen (Wu et al., 2023), and MetaGPT (Hong et al., 2023), relied heavily on manual configurations, while follow-up works significantly reduced dependence on human intervention, like DsPy (Khattab et al., 2023) automating prompt optimization, GPTSwarm (Zhuge et al., 2024) optimizing inter-agent communication, and EvoAgent (Yuan et al., 2024) self-evolving agent profiling. More recent efforts (Hu et al., 2024b; Shang et al., 2024; Zhang et al., 2024c), have demonstrated that these automated pipelines enable the development of surprisingly creative and powerful agentic workflows, marking significant progress toward fully autonomous agentic AI. Despite their success, existing automation pipelines often lack diversity in both LLM heterogeneity and complexity scheduling:  

➠ Lack of LLM heterogeneity. Mainstream multi-agent workflows are often homogeneous, relying on a single, expensive LLM like GPT-3.5/GPT-4o to instantiate all agents (Zhang et al., 2024c; Chen et al., 2023a). However, as highlighted by studies on LLM routing (Chen et al., 2023c; Hu et al., 2024a), the capabilities of different LLMs are often complementary rather than forming a strict superset relationship. In certain scenarios, smaller LLMs can perform tasks comparably or even outperform their larger counterparts at a significantly lower token cost. Against this backdrop, we advocate for agentic workflows to be heterogeneous, incorporating a diverse ensemble of LLMs with varying sizes, capabilities, and sources.  

➠ Lack of complexity diversity. Existing automated agentic workflows often prioritize single-objective optimization, focusing on performance or solution quality. This typically results in a singular, complex workflow incorporating elements like CoT ensembles and multi-turn discussions (Hu et al., 2024b; Shang et al., 2024). However, real-world user queries vary significantly in difficulty, as exemplified by the MMLU benchmark (Hendrycks et al., 2021a), which spans tasks from elementary to graduatelevel. While complex workflows are essential for the latter, simpler queries can be efficiently addressed by single-agent I/O (Feng et al., 2024). Thus, we advocate for optimizing a diverse set of workflows, tailoring simple workflows to straightforward tasks and reserving complex ones for more intricate challenges.  

The above considerations and observations raise critical questions regarding the current paradigm of agentic system design: How can we automatically optimize a set of heterogeneous, complexity-adaptive agentic workflows that provide diverse solutions for varied queries?  

To this end, we propose EvoFlow, a niching evolutionary algorithm (EA)-based framework for automatically searching a population of heterogeneous, complexity-adaptive agentic workflows, rather than a single, homogeneous, complex workflow. Technically, EvoFlow innovatively frames the agentic search as a multi-objective optimization problem, considering both cost and performance, ultimately generating a Pareto-optimal set of workflows balancing these factors. EvoFlow uses operator nodes, i.e., a set of LLM-agent invoking nodes, as the fundamental units of its search space. The workflow population is initialized by selecting and combining multiple operator nodes. Afterward, EvoFlow continuously evolves by processing incoming queries. It $\bullet$ tag-based retrieves relevant workflows as parents, performs $\pmb{\phi}$ crossover to generate offspring workflows, applies $\pmb{\theta}$ mutation with extensive mutation functions, i.e., LLM/prompt/operator mutation, to evolve the offsprings. Finally, $\pmb{\mathscr{Q}}$ niching selection is leveraged to maintain diversity and quality in the population. During inference, EvoFlow autonomously retrieves domain-relevant and complexityadapted agentic systems from the well-optimized population, to swiftly and efficiently address user queries.  

We conduct comprehensive evaluations on six widely adopted benchmarks. In heterogeneous settings, EvoFlow surpasses powerful o1-preview at $12.4\%$ of its inference cost by utilizing weaker open-source models (e.g., LLaMa-3.1-70b and $\mathsf{Q W e n}\mathrm{-}2\cdot5\mathrm{-}72\mathrm{b}\mathrm{)}$ ; In homogeneous settings, EvoFlow outperforms state-of-theart (SOTA) agentic workflows by an average of $1.23\%\sim$ $29.86\%$ in performance. More importantly, EvoFlow is highly economical, with a training cost of only one-third of SOTA baseline AFlow $\$0.45$ vs $\$1.23$ and an inference cost of merely one-fifth $(\$0.51$ vs $\$2.62$ ), while surpassing AFlow by $5.91\%\uparrow$ on the MATH Benchmark.  

Briefly put, our contributions can be summarized as:  

• Paradigm Transformation. We for the first time explicitly formulate agentic workflow automation as a costperformance multi-objective optimization problem, highlighting LLM heterogeneity and complexity diversity as key features for the development of multi-agent systems. • Practical Solution. We propose a niching evolutionary algorithm-based framework, EvoFlow, which autonomously evolves a population of heterogeneous and complexity-diverse agentic workflows across various task domains with minimal human intervention. • Emperical Evaluation. Extensive experiments on seven benchmarks show that EvoFlow is (I) diverse, evolving a workflow population ranging from simple I/O to complex multi-turn interactions; $(\mathbf{II})$ high-performing, surpassing previous handcrafted and automated workflows by $1.23\%\sim29.86\%$ ; (III) economical, surpassing powerful o1-preview with weaker open-source models.  

# 2. Related Work  

LLM-based Autonomous Agents Building on the success of single agent (Shen et al., 2024; Zhu et al., 2024b; Zhong et al., 2024), studies have shown that interaction among multiple LLM-based agents can substantially enhance individual model capabilities (Wang et al., 2024), as seen in several early frameworks, including CAMEL (Li et al., 2023), AutoGen (Wu et al., 2023), BabyAGI (Nakajima, 2023), and LLM-Debate (Du et al., 2023). However, these initial approaches heavily depended on manually crafted designs, which constrained the adaptability and flexibility of agents in addressing unforeseen challenges (He et al., 2023; Chen et al., 2023d). Consequently, the push toward automating agentic workflows has gained momentum.  

Automated Agentic Workflows Efforts to automate agentic workflows can be broadly categorized into the following types: (1) Prompt Optimization, exemplified by PromptBreeder (Fernando et al., 2023) and DsPy (Khattab et al., 2023); (2) Inter-agent Topology, which focuses on orchestrating interactions among agents, such as GPTSwarm (Zhuge et al., 2024), DyLAN (Liu et al., 2023), EvoMAC (Hu et al., 2024c), and G-Designer (Zhang et al., 2024b); (3) Agent Persona/Profile, represented by AgentVerse (Chen et al., 2023d) and EvoAgent (Yuan et al., 2024). More recently, Hu et al. (2024b) formalized the concept of Automated Design of Agentic Systems, with subsequent advancements by AgentSquare (Shang et al., 2024) and AFlow (Zhang et al., 2024c). However, these automation pipelines are predominantly homogeneous, i.e., utilizing a single-source LLM, and lack the integration of heterogeneous LLM agents of varying sizes and sources. Additionally, they typically produce a fixed workflow (Yuan et al., 2024; Zhuge et al., 2024; Zhang et al., 2024c), which cannot dynamically allocate resources when confronted with tasks/queries of different levels and complexities.  

Table 1. Comparison among different automation techniques.   


<html><body><table><tr><td>Method</td><td>Prompt</td><td>Agent OptimizeTopologyProfile</td><td>Agent</td><td>LLM eBackboneAdaptivity</td><td>Complexity</td></tr><tr><td>AgentVerse</td><td></td><td></td><td></td><td></td><td>×</td></tr><tr><td>GPTSwarm</td><td></td><td></td><td>×</td><td></td><td></td></tr><tr><td>EvoMAC</td><td></td><td></td><td>×</td><td></td><td></td></tr><tr><td>EvoAgent</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>EvoPrompt</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>ADAS</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>AFlow</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>AgentSquare</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>EvoFlow</td><td></td><td></td><td></td><td></td><td></td></tr></table></body></html>  

Evolutionary Algorithm Evolutionary algorithms (EAs) are no new to agentic AI (Cetnarowicz et al., 1996; Li & Liu, 2016; Liu et al., 2020). In the era of LLM-based agents, researchers have explored the interplay between EA and LLM agents, including prompt engineering (Xu et al., 2022; Shi et al., 2024), code generation (Romera-Paredes et al., 2024), project planning (Tao et al., 2023) and inference time scaling (Lee et al., 2025). EvoAgent (Yuan et al., 2024) and EvoPrompt (Guo et al., 2023) employ simple genetic algorithms to optimize agent profiles and prompts, whose, however, level of automation is highly constrained, focusing solely on single-agent prompt optimization and failing to evolve at the workflow level, as illustrated in Table 1.  

# 3. Preliminary  

In this section, we formally define the search space of EvoFlow and the objective of workflow optimization.  

![](https://cdn-mineru.openxlab.org.cn/extract/f6fc2303-638d-4bb4-901b-d70af57f7b1c/bff851496e13627f4daa80ee043b22e9645489f85650358bc7a04f229ab09df3.jpg)  
Figure 2. The visualization of notations in EvoFlow.  

Search Space. The search space of EvoFlow is defined hierarchically, with the basic unit being the (LLM-)invoking node. These are further assembled into (composite) operator nodes, which are then combined to form the complete workflow $\mathcal{G}$ , as visualized in Figure 2. Each invoking node $I_{i}$ is defined as follows:  

$$
I_{i}=(M_{i},P_{i},\tau_{i}),P_{i}\in\mathcal{P},\tau_{i}\in[0,1],
$$  

where $P_{i}$ represents the associated prompt, with $\mathcal{P}$ denoting the feasible prompt space, and $\tau_{i}$ is the temperature parameter. $M_{i}=(|M_{i}|,C_{i},L_{i})$ represents an LLM instance from the feasible model pool $\mathcal{M}=\{M_{1},\cdot\cdot\cdot~,M_{|\mathcal{M}|}\}$ , characterized by its model size $|M_{i}|$ , token cost $C_{i}$ , and inference delay $L_{i}$ . Thus, the feasible space for invoking nodes is given by $\mathcal{I}=\mathcal{M}\times\mathcal{P}\times\mathbb{R}_{[0,1]}$ . Notably, prior agentic automation pipelines have generally excluded $\mathcal{M}$ from their search space (Zhang et al., $2024\mathrm{c}$ ; Yuan et al., 2024; Zhuge et al., 2024), often preselecting a single LLM $M$ to instantiate all agents, which constrains the development of more diverse and capability-rich agentic systems. Building upon the invoking nodes, the operator node $O_{j}$ is represented by:  

$$
O_{j}=(\mathbb{Z}_{j}^{o},\mathbb{E}_{j}^{o}),\mathbb{Z}_{j}^{o}=\{I_{1},\ldots,I_{n}\},\mathbb{\mathcal{E}}_{j}^{o}\subseteq\mathbb{Z}_{j}^{o}\times\mathbb{Z}_{j}^{o},
$$  

where $\mathcal{T}_{j}^{o}$ is a subset of invoking nodes, and $\mathcal{E}_{j}^{o}$ signifies the connectivity relationship among invoking nodes. The overall agentic workflow $\mathcal{G}$ is defined as:  

$$
\begin{array}{r l}&{\mathcal{G}=(\mathcal{O}^{S},\mathcal{E}^{a}),\mathcal{O}^{S}=\{O_{1},\ldots,O_{m}\},\mathcal{E}^{a}\subseteq\mathcal{O}^{S}\times\mathcal{O}^{S},}\ &{\quad=(\mathcal{T}^{S},\mathcal{E}^{o}),\mathcal{T}^{S}=\bigcup_{j=1}^{m}\mathcal{Z}_{j}^{o},\mathcal{E}^{o}=\bigcup_{j=1}^{m}\mathcal{E}_{j}^{o}\cup\mathcal{E}^{a},}\end{array}
$$  

where ${\mathcal{O}}^{S}\subseteq{\mathcal{O}},{\mathcal{T}}^{S}\subseteq{\mathcal{T}}$ , $m$ denotes the number of operator nodes in $\mathcal{G}$ , $\mathcal{E}^{a}/\mathcal{E}^{o}$ denote intra/inter-operator connections.  

Problem Formulation We first present the optimization objective for traditional agentic automation methods. Given a task domain $T$ and an performance evaluator function $u(\cdot)$ , the objective function is defined as:  

$$
\mathcal{G}^{*}=\underset{\mathcal{G}\in\mathcal{H}(\mathbb{Z},\mathcal{E})}{\arg\operatorname*{max}}u(\mathcal{G},T)=\underset{\mathbb{Z}^{S}\subseteq\mathbb{Z},\mathcal{E}\subseteq\mathbb{Z}^{S}\times\mathbb{Z}^{S}}{\arg\operatorname*{max}}u\left((\mathcal{Z}^{S},\mathcal{E}),T\right),
$$  

where $\mathcal{T}$ represents the feasible space of invoking nodes, and $\mathcal{H}(\mathcal{I},\mathcal{E}^{o})$ denotes the invoking node-based search space for $\mathcal{G}$ . As shown in Equation (4), existing methods (Zhang et al., $2024\mathrm{c}$ ; Shang et al., 2024; Zhuge et al., 2024) typically perform single-objective optimization. In contrast, the optimization objective of EvoFlow is multi-objective:  

![](https://cdn-mineru.openxlab.org.cn/extract/f6fc2303-638d-4bb4-901b-d70af57f7b1c/c1acb74e32252e09488164b6db83bb30c415619143d5dafdb5376d86bceac7eb.jpg)  
Figure 3. The overall framework of EvoFlow. The fundamental unit is the invoking nodes, which collectively form the operator node. EvoFlow initializes the population by combining multiple operator nodes into a workflow (individual), followed by tag-based retrieval and crossover & mutation to generate novel offspring workflows. The population is updated via niching-based selection.  

$$
\mathcal{G}^{\star}=\underset{\mathcal{G}\in\mathcal{H}(\mathbb{Z},\mathcal{E}^{o})}{\arg\operatorname*{max}}~\left[u(\mathcal{G},T),-c(\mathcal{G},T)\right]^{\top},
$$  

where $c(\cdot)$ evaluates the system cost, and $\mathcal{G^{\star}}$ represents the Pareto optimal set balancing cost and performance, which includes a set of non-dominated agentic workflows that are well-distributed and located near the Pareto front (PF) in the objective space. Detailed explanations are in Appendix C.  

However, the optimization in Equation (5) is currently based on invoking nodes, which results in an excessively large search space and does not explicitly account for many already-existing high-performing composite structures, such as CoT (Wei et al., 2022), ToT (Yao et al., 2023a), and Multiagent Debate (Du et al., 2023). Therefore, we reformulate the search space to be operator node-based:  

$$
\begin{array}{r l}&{\mathcal{G}^{\star}=\underset{\mathcal{G}\in\mathcal{H}(\mathcal{O},\mathcal{E}^{a})}{\arg\operatorname*{max}}~\left[u(\mathcal{G},T),-c(\mathcal{G},T)\right]^{\top},}\ &{\quad\quad=\underset{\mathcal{O}^{S}\subseteq\mathcal{O},\mathcal{E}^{a}\subseteq\mathcal{O}^{S}\times\mathcal{O}^{S}}{\arg\operatorname*{max}}~\left[u((\mathcal{O}^{S},\mathcal{E}^{a}),T),-c((\mathcal{O}^{S},\mathcal{E}^{a}),T)\right]^{\top},}\end{array}
$$  

Where $\mathcal{O}$ represents the feasible space for operator nodes.  

# 4. Methodology  

As shown in Figure 3, EvoFlow initializes a set of workflows with varying complexities and domain specializations, each tagged with its domain expertise (Section 4.1). As new user queries arrive, EvoFlow performs tag-based retrieval to select the most relevant (workflow) individuals as parents and generates offspring workflows through crossover and mutation (Section 4.2). Upon receiving environmental feedback, the framework evaluates the new workflows in the multi-objective space, and conducts environmental selection to maintain population diversity and efficiency (Section 4.3).  

## 4.1. Population Initialization  

EvoFlow initially populates the feasible space of operator nodes with a basket of powerful single-/multi-agent structures, including CoT (Wei et al., 2022), Ensemble (Jiang et al., 2023), Self-Reflexion (Shinn et al., 2023), Multi-agent Debate (Du et al., 2023), etc. Detailed formalizations are in Appendix D. This transforms $\mathcal{O}$ into a finite set:  

$$
{\mathcal O}^{(0)}=\{{\cal O}_{\mathrm{CoT}},{\cal O}_{\mathrm{Reflexion}},\cdot\cdot\cdot,{\cal O}_{\mathrm{Debate}}\}.
$$  

Importantly, thought $\mathcal{O}^{(0)}$ is initialized as a finite set, this does not constrain EvoFlow’s potential to explore a broader search space. This is because (1) practitioners can easily customize operator templates as needed, and (2) EvoFlow’s crossover and mutation can generate novel operators.  

To initialize the workflow population $\mathbf{P}^{(0)}\qquad=$ $\{\mathcal{G}_{1},\mathcal{G}_{2},\cdots,\mathcal{G}_{N}\}$ (where the population size is set to $N$ ), we generate $N$ workflows as follows:  

$$
\mathcal{G}_{k}\gets\{\mathcal{O}_{k},\mathcal{E}_{k}^{a}\},\mathcal{E}_{k}^{a}\subseteq\mathcal{O}_{k}\times\mathcal{O}_{k},
$$  

$$
\mathcal{O}_{k}=\{O_{i}(\mathcal{M}_{k,i}^{S},\mathcal{P}_{k,i}^{S},\tau_{k})\}_{i=1}^{m}\subseteq\mathcal{O}^{(0)},
$$  

$$
i\sim\mathrm{Uniform}(1,|\mathcal{O}^{(0)}|),\mathcal{M}_{k,1}^{S}\subseteq\mathcal{M},\mathcal{P}_{k,i}^{S}\subseteq\mathcal{P},
$$  

where $m$ denotes the number of operators in workflow $\mathcal{G}_{k}$ , $O_{i}$ represents the operator template randomly selected from  

O(0), and it is instantiated with LLMs MkS,i (sampled from LLM pool $\mathcal{M}$ ) and prompts $\mathcal{P}_{k,i}^{S}$ . Upon creating a workflow individual, we assign it multiple utility indicator tags, which suggest the task domains where it might excel. These tags are conducive to a rapid match between user queries and relevant workflows, generated as follows:  

$$
\{\varkappa_{1}^{k},\cdot\cdot\cdot~,\varkappa_{\kappa}^{k}\}\leftarrow f_{\mathrm{tag}}(\mathcal{G}_{k}),
$$  

where $\varkappa_{i}$ represents the $i$ -th tag of $\mathcal{G}_{k}$ , with a total of $\kappa$ tags per workflow, and $f_{\mathrm{tag}}$ is a LLM-powered tag generation function (see Appendix E.1). Thus, we have initialized the workflow population $\mathbf{P}^{(0)}$ prior to iterative evolution.  

## 4.2. Retrieval, Crossover, and Mutation  

Upon initializing $\mathbf{P}^{(0)}$ , we seek to evolve it driven by incoming task queries. Rather than activating the entire population for each query, we select a subset of workflows that are most relevant and complexity-adapted to it, which prevents the population from converging solely toward high-complexity, single-domain evolution. Specifically, for the $t$ -th query $q_{t}$ , we first select the $K$ most relevant workflow individuals from the population $\mathbf{P}^{(t)}$ , based on utility indicator tags:  

$$
\begin{array}{r l}&{\{\mathcal{G}_{t1},\boldsymbol{\cdot\cdot\cdot},\mathcal{G}_{t K}\}=\mathrm{TopK}\left(\boldsymbol{S}(\{\mathcal{G}_{i}\}_{i=1}^{N}\mid q_{t}),K\right),}\ &{\qquad\quad S(\mathcal{G}_{i}\mid q_{t})=\displaystyle\sum_{j=1}^{\kappa}\frac{\mathbf{v}\left(\mathcal{H}_{i,j}\right)\cdot\mathbf{v}\left(q_{t}\right)}{\|\mathbf{v}\left(\mathcal{H}_{i,j}\right)\|\|\mathbf{v}\left(q_{t}\right)\|},}\end{array}
$$  

where $\varkappa_{i,j}$ is the $j$ -th tag of $\mathcal{G}_{i}$ , $\mathrm{TopK}(\cdot,K)$ is a selection function that outputs elements with $K$ largest values, and $\mathbf{v}(\cdot)$ maps queries/tags to fixed-length embeddings using lightweight models such as SentenceBERT (Reimers, 2019) or MiniLM (Wang et al., 2020). The similarity score $S({\mathcal{G}}_{i})$ $q_{t}$ ) is computed based on the cosine similarity between the tag/query embeddings. After identifying the $K$ most relevant workflows, these are treated as parents to generate offspring workflows with crossover function:  

$$
\mathcal{G}_{\circ}^{(t)}\leftarrow\mathrm{Crossover}(\mathcal{G}_{t1},\cdot\cdot\cdot,\mathcal{G}_{t K}),
$$  

where G(t) represents the generated workflow, and the Crossover $(\cdot)$ function is LLM-facilitated (see prompts in Appendix E.2). To further enhance the diversity of the population and foster the evolution of novel agentic architectures, we apply a suite of mutation functions to refine the sketched offspring, as described below:  

LLM Mutation $\mu^{l}(\cdot)$ replaces the LLM backbone of an invoking node within an existing workflow. This mutation can be beneficial in scenarios where an agent in the workflow is underperforming, such as when a small 7b agent fails to handle complex subtask decomposition and needs to be replaced by a larger one, or when a simpler 72b model suffices to accomplish a task as a 405b model does. The mutation process empowers EvoFlow to evolve into powerful heterogeneous workflows, formalized as follows:  

$$
\begin{array}{c}{\tilde{\mathcal{G}}=\mu^{l}(\mathcal{G},\mathcal{R})=(\mathcal{T}^{\prime},\mathcal{E}),}\ {\quad\mathcal{T}^{\prime}=\{(M_{i}^{\prime},P_{i},\tau_{i})\mid I_{i}\in\mathcal{T},M_{i}^{\prime}=\mathcal{R}(M_{i}\mid\mathcal{P}_{\mathrm{LLM}})\},}\end{array}
$$  

where $\tilde{\mathcal{G}}$ represents the mutated individual, and $\mathcal{R}^{l}(\cdot)$ is an LLM-powered process that determines whether the LLM in $I_{i}$ should be changed based on the LLM performance history pool $\mathcal{P}_{\mathrm{LLM}}$ . Details of $\mathcal{P}_{\mathrm{LLM}}$ and $\mathcal{R}^{l}(\cdot)$ are provided in Appendix F.1 and Appendix E.3.1, respectively.  

Prompt Mutation $\mu^{p}(\cdot)$ involves modifying the prompts of invoking nodes, such as incorporating few-shot examples or providing clearer task instructions, as follows:  

$$
\begin{array}{c}{{\tilde{\mathcal{G}}=\mu^{p}(\mathcal{G},\mathcal{R}^{p})=(\mathcal{T}^{\prime},\mathcal{E}),}}\ {{\mathrm{}\mathcal{T}^{\prime}=\{(M_{i},P_{i}^{\prime},\tau_{i})|I_{i}\in\mathbb{Z},P_{i}^{\prime}=\mathcal{R}^{p}(P_{i}|\mathcal{P}_{\mathrm{wf}})\},}}\end{array}
$$  

where $\mathcal{P}_{\mathrm{wf}}$ is the experience history of the workflow population, the configuration of which is placed at Appendix F.2. $\mathcal{R}^{p}(\cdot)$ is also LLM-powered (see Appendix E.3.2).  

Operator Mutation $\mu^{o}(\cdot)$ refers to the modification of the operators or their topological connections $(\mathcal{E}^{a}\cup\mathcal{E}^{o})$ . Practical scenarios include removing redundant Reflexion operators or adding a “format” operator within the workflow to enhance the formatting accuracy of code generation. The process is described as:  

$$
\begin{array}{r l r}&{}&{\tilde{\mathcal{G}}_{k}=\mu^{o}(\mathcal{G}_{k},\mathcal{R}^{o})=(\mathcal{O}_{k}^{\prime},\mathcal{E}_{k}^{a^{\prime}}),}\ &{}&{\mathcal{O}_{k}^{\prime}=\left(\mathcal{O}_{k}\backslash\mathcal{O}^{d e l}\right)\cup\mathcal{O}^{a d d},}\ &{}&{{\mathcal{E}_{k}^{a^{\prime}}}=\mathcal{R}^{o}(\mathcal{O}_{k}^{\prime}\mid\mathcal{P}_{\mathrm{wf}})\subseteq\mathcal{O}_{k}^{\prime}\times\mathcal{O}_{k}^{\prime},}\end{array}
$$  

where $\mathcal{O}^{d e l}$ and $\mathcal{O}^{a d d}$ are deleted and added operators by ${\mathcal{R}}^{o}$ , and $\mathcal{E}_{k}^{a^{\prime}}$ denotes the modified topological operator connections. The prompts for ${\mathcal{R}}^{o}$ is in Appendix E.3.3.  

The mutated offspring is denoted as G(⊚t). With new individuals introduced, the critical challenge is: how can we design an efficient selection mechanism to evolve the population toward greater diversity and higher performance?  

## 4.3. Niching-based Selection  

The aforementioned challenge is one that prior agentic automation methods have struggled with: their singleobjective optimization approaches often result in increasingly complex workflows, lengthier evaluation/test, and significantly higher API costs (Zhang et al., 2024c). For instance, running ADAS (Hu et al., 2024b) on ARC benchmark (Chollet, 2019) with $\mathtt{g p t-3.5\mathrm{-turbo-}0125}$ incurs cost up to $\$300$ USD. In contrast, EvoFlow introduces an efficient niching-based workflow selection mechanism, guiding evolution across multiple domains and complexity.  

Niching, in our context, analogous to previous niching EAs (White et al., 2023), refers to clusters of similar individuals where environmental selection is conducted. To determine the niching area for a new individual $\mathcal{G}_{\circledcirc}^{(t)}$ , we compute it based on cost and utility tags as follows:  

$$
\begin{array}{r l}&{\mathbf{P}^{N A}=\{\mathcal{G}_{q1},\cdots,\mathcal{G}_{q E}\}=\mathrm{TopK}\left(\{-\mathrm{Rank}(\mathcal{G}_{i})\}_{i=1}^{N},E\right),}\ &{\quad\mathrm{Rank}(\mathcal{G}_{i})=\mathrm{Rank}_{S}(\mathcal{G}_{i})+\mathrm{Rank}_{c}(\mathcal{G}_{i})}\ &{\qquad=\mathrm{Index}\left(\mathcal{G}_{i},\mathrm{Sort}(\{S(\mathcal{G}_{\odot}^{(t)},\mathcal{G}_{j})\}_{j=1}^{N})\right)}\ &{\qquad+\mathrm{Index}\left(\mathcal{G}_{i},\mathrm{Sort}(\{|c(\mathcal{G}_{\odot}^{(t)})-c(\mathcal{G}_{j})\}_{j=1}^{N})\right)}\end{array}
$$  

where $\mathbf{P}^{N A}$ denotes the identified niching area comprising $E$ individuals. The function $\mathrm{Rank}(\cdot)$ computes the approximate ranking of an individual $\mathcal{G}_{i}$ relative to the new individual, which is determined by the cost similarity rank $\mathrm{Rank}_{c}(\cdot)$ and the tag-based similarity rank $\mathrm{Rank}_{S}(\cdot)$ . Subsequently, the parents, offspring, and workflows in the niching area are executed for query $q_{t}$ , and their records are updated as follows:  

$$
\begin{array}{r l}&{c^{(t)}({\mathcal{G}}_{i})=1/t_{i}^{\prime}\left(c^{(t-1)}({\mathcal{G}}_{i})\cdot t_{i}^{\prime}+c({\mathcal{G}}_{i}\mid q_{t})\right),}\ &{p^{(t)}({\mathcal{G}}_{i})=1/t_{i}^{\prime}\left(p^{(t-1)}({\mathcal{G}}_{i})\cdot t_{i}^{\prime}+p({\mathcal{G}}_{i}\mid q_{t})\right),}\ &{\qquad\mathcal{G}_{i}\in\mathbf{P}^{N A}\cup\{{\mathcal{G}}_{t i}\}_{i=1}^{K}\cup\{{\mathcal{G}}_{\odot}^{(t)}\},}\end{array}
$$  

where $c(\mathcal{G}\mid q)$ measures the economical cost incurred by workflow $\mathcal{G}$ in addressing query $q,t_{i}^{\prime}$ records the number of times workflow $\mathcal{G}_{i}$ being executed up to iteration $t$ , $c^{(t)}(\mathcal{G}_{i})$ tracks the cumulative cost of $\mathcal{G}_{i}$ . $p(\cdot)$ follows a similar formulation for performance metrics. Finally, the environmental selection is performed within the niching area by calculating the fitness value as follows:  

$$
\mathcal{F}(\mathcal{G})=\sum_{\mathcal{G}\in\mathbf{P}^{N A}\cup\{\mathcal{G}_{\circledast}^{(t)}\}}\left(\exp\frac{\mathbf{I}(\mathcal{G},\mathcal{G}_{\circledast}^{(t)})}{\varphi\cdot\mathbf{I}^{\operatorname*{max}}}\right),
$$  

where $\mathbf{I}(\cdot,\cdot)$ is a Pareto dominance-preserving binary indicator. If individual $\mathcal{G}_{1}$ dominates $\mathcal{G}_{2}$ , i.e., $c({\mathcal{G}}_{1})~<$ $c(\mathcal{G}_{2})$ and $p(\mathcal{G}_{1})<p(\mathcal{G}_{2})$ (aligning with our objective in Equation (5)), then $\mathbf{I}(\mathcal{G}_{1},\mathcal{G}_{2})<\mathbf{I}(\mathcal{G}_{2},\mathcal{G}_{1})$ . The maximum absolute indicator value, $\mathbf{I}^{\mathrm{max}}$ , is defined as $\mathbf{I}^{\mathrm{max}}=$ $\operatorname*{max}_{\mathcal{G}_{1},\mathcal{G}_{2}\in\mathbf{P}^{N A}\cup\{\mathcal{G}_{\circledast}^{(t)}\}}\left|\mathbf{I}(\mathcal{G}_{1},\mathcal{G}_{2})\right|$ . The scaling factor $\varphi$ is set to 0.05, following established practices (Zitzler & Künzli, 2004). A smaller fitness value in Equation (17) corresponds to a better individual. The worst-performing workflow, $\mathcal{G}^{\mathrm{worst}}$ , which has the largest fitness value in $\mathbf{P}^{N A}\cup\{\mathcal{G}_{\circledcirc}^{(t)}\}$ , will be eliminated from the population.  

## 4.4. Discussion  

The evolution process of EvoFlow operates on a query-byquery basis, continuously evolving, mutating, and nichingselecting workflows in response to incoming queries. This iterative process gradually produces a Pareto set of agentic workflows with varying complexity and superior performance. The overall algorithmic procedure is summarized in Appendix B, with notations clarified in Appendix A.  

# 5. Experiments  

## 5.1. Experiment Setup  

Tasks and Benchmarks. We evaluate EvoFlow on six public benchmarks covering four domains: (1) math reasoning, GSM8K (Cobbe et al., 2021), MATH (Hendrycks et al., 2021b), and MultiArith (Roy & Roth, 2016); (2) code generation, HumanEval (Chen et al., 2021) and MBPP (Austin et al., 2021)); (3) embodied, ALFWorld (Shridhar et al., 2021). For the MATH benchmark, we follow (Hong et al., 2024) in selecting a harder subset (617 problems). The dataset statistics and splits are in Appendix G.1.  

Baselines. We compare EvoFlow with two series of agentic baselines: (1) manually designed workflows, including Chain-of-Thought (Wei et al., 2022), ComplexCoT (Fu et al., 2022)), Self-Consistency (SC) (Wang et al., 2023a), LLM-Debate (Du et al., 2023), LLM-Blender (Jiang et al., 2023), DyLAN (Liu et al., 2023), AgentVerse (Chen et al., 2023d) and MacNet (Qian et al., 2024); (2) autonomous workflows, including GPTSwarm (Zhuge et al., 2024), AutoAgents (Chen et al., 2023b), ADAS (Hu et al., 2024b), AgentSquare (Shang et al., 2024) and AFlow (Zhang et al., 2024c). Detailed baseline setups are in Appendix G.2.  

LLM Backbones. We leverage one closed-source model, gpt-4o-mini-0718, along with four opensource models: $\mathtt{l}\mathtt{l}\mathtt{a m a}-3\le70\mathtt{b}$ , $\mathtt{Q w e n}{-}2{-}72\mathrm{b}$ , Deepseek-V2.5, and Hermes-3-70b. LLMs are accessed via APIs, with the temperature set to 1.  

Parameter Configuration. We select the following operators to initialize the feasible space of operator nodes: CoT, LLM-Debate, Take-a-step-back, Self-consistency, SelfRefine, Ensemble, ReAct, and ExpertPrompting. Detailed instructions are in Appendix D. The function $\mathbf{v}(\cdot)$ adopts allMiniLM-L6-v2 (Wang et al., 2020). The number of parent workflows in Equation (10) is set as $K=3$ , and the number of utility indicator tags in Equation (9) is set as $\kappa=5$ . The population size $N$ is 15, and $E=5$ in Equation (15).  

## 5.2. Performance Analysis  

We present via three experimental settings: ■ homogeneous setting, where all methods, including EvoFlow, are equipped with a unified LLM backbone; ■ heterogeneous setting, where EvoFlow is assigned an LLM pool and optimizes a heterogeneous agentic workflow population; ■ cross-domain setting, where we mix up multiple crossdomain datasets for training. The analysis is as follows:  

Homogeneous Performance Table 2 demonstrates that EvoFlow outperforms existing hand-crafted or automated agentic workflows across six benchmarks. Specifically, on the MATH benchmark, it exceeds vanilla $\mathtt{g p t-4o-m i n i}$ by $11.41\%$ and surpasses the SOTA baseline AFlow by $6.42\%$ . On the embodied benchmark ALFWorld, EvoFlow achieves the optimal $68.57\%$ , outperforming the secondbest AgentSquare by $2.15\%$ .  

<html><body><table><tr><td>Method</td><td>GSM8K</td><td>MATH</td><td>MultiArith</td><td>HumanEval</td><td>MBPP</td><td>ALFWorld</td><td>Avg.</td></tr><tr><td>Vanilla</td><td>87.45</td><td>46.29</td><td>96.85</td><td>87.08</td><td>71.83</td><td>38.71</td><td>71.37</td></tr><tr><td>CoT (Wei et al.,2022)</td><td>87.10↓0.35</td><td>46.40↑0.11</td><td>96.31↓0.54</td><td>88.13↑1.05</td><td>71.83↓0.00</td><td>39.92个1.21</td><td>71.62个0.25</td></tr><tr><td>ComplexCoT (Fu et al.,2022)</td><td>86.89↓0.56</td><td>46.53↑0.24</td><td>96.70±0.15</td><td>87.49↑0.41</td><td>72.36↑0.53</td><td>41.68↑2.97</td><td>71.94r0.57</td></tr><tr><td>SC (CoTx5) (Wang et al.,2023a)</td><td>87.57↑0.12</td><td>47.91↑1.62</td><td>96.58↓0.27</td><td>88.60↑1.52</td><td>73.60↑1.77</td><td>40.55↑1.84</td><td>72.47↑1.10</td></tr><tr><td>MultiPersona (Wang et al., 2023b)</td><td>87.50↑0.05</td><td>45.430.86</td><td>97.49↑0.64</td><td>88.32个1.24</td><td>73.19↑1.36</td><td>39.10↑0.39</td><td>71.84↑0.47</td></tr><tr><td>LLM-Debate (Du et al.,2023)</td><td>89.47↑2.02</td><td>48.54↑2.25</td><td>97.33↑0.48</td><td>88.68↑1.60</td><td>70.29↓1.54</td><td>44.68↑5.97</td><td>73.17个1.80</td></tr><tr><td>LLM-Blender (Jiang et al.,2023)</td><td>88.35↑0.90</td><td>46.92个0.63</td><td>97.29↑0.44</td><td>88.80↑1.72</td><td>77.05个5.22</td><td>43.79个5.08</td><td>73.70↑2.33</td></tr><tr><td>DyLAN (Liu et al.,2023)</td><td>89.98↑2.53</td><td>48.63↑2.34</td><td>97.12个0.27</td><td>90.42↑3.34</td><td>77.30个5.47</td><td>53.32↑14.61</td><td>76.13↑4.76</td></tr><tr><td>AgentVerse (Chen et al.,2023d)</td><td>89.91↑2.46</td><td>47.35↑1.06</td><td>97.50↑0.65</td><td>89.29↑2.21</td><td>74.28↑2.45</td><td>45.03↑6.32</td><td>73.89↑2.52</td></tr><tr><td>MacNet (Qian et al.,2024)</td><td>87.95↑0.50</td><td>45.18↓1.11</td><td>96.03↓0.82</td><td>84.57↓2.51</td><td>65.286.55</td><td>43.66↑4.95</td><td>70.45 0.92</td></tr><tr><td>AutoAgents (Chen et al.,2023b)</td><td>87.69↑0.24</td><td>45.3210.97</td><td>96.420.43</td><td>87.64↑0.56</td><td>71.95↑0.12</td><td>46.15个7.44</td><td>72.53↑1.16</td></tr><tr><td>GPTSwarm (Zhuge et al.,2024)</td><td>89.14↑1.69</td><td>47.88↑1.59</td><td>96.79↓0.06</td><td>89.32个2.24</td><td>77.43个5.60</td><td>53.19↑14.48</td><td>75.63↑4.26</td></tr><tr><td>ADAS (Hu et al.,2024b)</td><td>86.12↓1.33</td><td>43.18↓3.11</td><td>96.0210.83</td><td>84.19↓2.89</td><td>68.13↓3.70</td><td>47.66↑8.95</td><td>70.880.49</td></tr><tr><td>AgentSquare (Shang et al., 2024)</td><td>87.62个0.17</td><td>48.51↑2.22</td><td>97.77↑0.92</td><td>89.08个3.00</td><td>78.46↑6.63</td><td>66.42个27.71</td><td>78.14↑6.77</td></tr><tr><td>AFlow(Zhanget al.,2024c)</td><td>91.16↑3.71</td><td>51.28↑3.31</td><td>96.22↓0.63</td><td>90.93↑3.85</td><td>81.67↑9.84</td><td>59.16个20.45</td><td>78.40个7.03</td></tr><tr><td>EvoFlow (Ours)</td><td>92.90↑4.85</td><td>57.70↑11.41</td><td>98.80↑1.95</td><td>92.85↑5.77</td><td>84.50↑10.34</td><td>68.57↑29.86</td><td>82.55↑11.18</td></tr></table></body></html>  

Table 2. Performance comparison with single agent, hand-craft multi-agent systems, and automated agentic workflows. The base LLM is consistently set as get $-4\odot$ -mini for all baselines. We bold the best results and underline the runner-ups.   
Table 3. Heterogeneous experiments on MATH and MBPP. “DyLANQwen” indicates that only $\mathtt{Q w e n}{-}2\cdot5{-}72\mathrm{b}$ was used to optimize DyLAN. For comparison, we included results from o1-preview, although EvoFlow exclusively utilized four open-source LLMs. We shade the values of the lowest overall cost, the lowest inference token, and the highest performance for both single agents and workflows.   


<html><body><table><tr><td rowspan="2" colspan="2"></td><td colspan="5">MATH</td><td colspan="5">MBPP</td></tr><tr><td>Training cost (10-3$)</td><td>Inference cost (10-3$)</td><td>Overall cost (10-3$)</td><td>Inference token</td><td>Acc. (%)</td><td>Training cost (10-3$)</td><td>Inference cost (10-3$)</td><td>Overall cost (10-3$)</td><td>Overall token</td><td>pass@1 (%)</td></tr><tr><td rowspan="6">ngle Sing</td><td>Llama-3.1-70b</td><td></td><td>24.50</td><td>24.50</td><td>90,678</td><td>31.93%</td><td>-</td><td>10.67</td><td>10.67</td><td>38,653</td><td>65.11%</td></tr><tr><td>Qwen-2.5-72b</td><td></td><td>32.30</td><td>32.30</td><td>85,436</td><td>63.80%</td><td></td><td>9.18</td><td>9.18</td><td>24,253</td><td>69.76%</td></tr><tr><td>Deepseek-V2.5</td><td></td><td>25.89</td><td>25.89</td><td>98,986</td><td>41.17%</td><td></td><td>11.93</td><td>11.93</td><td>44,589</td><td>76.74%</td></tr><tr><td>Hermes-3-70b</td><td></td><td>18.11</td><td>18.11</td><td>68,994</td><td>22.60%</td><td>-</td><td>9.49</td><td>9.49</td><td>30,328</td><td>63.28%</td></tr><tr><td>ol-preview</td><td></td><td>7840.51</td><td>7840.51</td><td>186, 701</td><td>70.20%</td><td></td><td>3209.44</td><td>3209.44</td><td>81,334</td><td>89.65%</td></tr><tr><td>AFloWLlama</td><td>653.97</td><td>1304.07</td><td>1958.05</td><td>6,054,698</td><td>36.97%</td><td>383.40</td><td>356.88</td><td>740.29</td><td>1,510,058</td><td>67.42%</td></tr><tr><td rowspan="7">cous ? en mo</td><td>AFlowQwen</td><td>1223.46</td><td>2622.63</td><td>3846.10</td><td>8,614, 237</td><td>66.38%</td><td>824.48</td><td>773.63</td><td>1598.11</td><td>2,258,279</td><td>80.84%</td></tr><tr><td>AFloWDeepseck</td><td>815.96</td><td>1945.90</td><td>2761.86</td><td>8,693,402</td><td>48.65%</td><td>456.96</td><td>418.33</td><td>875.29</td><td>1,733,829</td><td>79.14%</td></tr><tr><td>AFlowHermes</td><td>572.09</td><td>1045.20</td><td>1617.29</td><td>4,886,371</td><td>32.14%</td><td>353.04</td><td>339.71</td><td>692.75</td><td>1, 289, 901</td><td>66.13%</td></tr><tr><td>DyLANLlama</td><td>9317.07</td><td>2676.03</td><td>11993.10</td><td>11,258,530</td><td>38.19%</td><td>5817.09</td><td>966.12</td><td>6783.21</td><td>4,879,526</td><td>69.92%</td></tr><tr><td>DyLANQwen</td><td>12847.72</td><td>4015.97</td><td>16863.69</td><td>15,242, 982</td><td>64.17%</td><td>7491.78</td><td>1480.94</td><td>8972.72</td><td>3,386,087</td><td>75.63%</td></tr><tr><td>DyLANDeepseek</td><td>10388.54</td><td>2375.11</td><td>12763.64</td><td>13,282,450</td><td>46.20%</td><td>6209.41</td><td>1084.34</td><td>7293.45</td><td>4,296,199</td><td>80.13%</td></tr><tr><td>DyLANHermes</td><td>7103.88</td><td>2106.35</td><td>9210.23</td><td>8,129,786</td><td>30.14%</td><td>3965.38</td><td>714.55</td><td>4679.93</td><td>4,150,887</td><td>65.29%</td></tr><tr><td></td><td>EvoFlow</td><td>459.24</td><td>513.34</td><td>972.58</td><td>1,660,284</td><td>72.90%</td><td>479.10</td><td>286.05</td><td>565.15</td><td>8,193,669</td><td>87.62%</td></tr></table></body></html>  

Heterogeneous Performance qwen-2.5-72b exhibits the best performance among the four open-source models, but even with the sophisticated optimization from AFlow, it only shows a $2.58\%$ improvement on MATH, still trailing behind the powerful o1-preview by $3.82\%$ ↓; however, EvoFlow, through the collective assembly and evolution of the four open-source models, surpasses $_{\odot1}$ -preview by $2.7\%$ . More importantly, the overall cost of EvoFlow is merely $12.4\%$ of that of $_{\mathsf{O}1-\mathsf{p}\mathtt{r e v}\mathrm{i}\in\mathsf{w}}$ . This clearly illustrates both the necessity and potential of optimizing LLM-heterogeneous workflows.  

Cross-domain Performance We also include a crossdomain optimization setting, where training sets from different domain datasets are concatenated to assess whether an agentic automation method can optimize reasonable workflows across domains. As shown in Table 6, crossdomain optimization challenges many existing baselines: vanilla Deepseek-V2.5 achieves $41.17\%$ on MATH, and GPTSwarm improves it by $4.19\%$ when optimized solely for MATH. However, joint optimization on MATH+MBPP results in a negative gain, reducing performance to $39.18\%$ Other methods like DyLAN and AFlow also suffer from the same issue. In contrast, EvoFlow successfully benefits from cross-domain training on MBPP, improving from $87.62\%$ to $88.35\%$ , a result attributed to the optimization of the workflow population rather than a single individual.  

## 5.3. Cost Analysis & Case Study  

We demonstrate the resource-friendly nature of EvoFlow’s agentic automation system across three dimensions: training/inference API costs and token consumption. As shown in Table 3, optimizing AFlowQwen on MATH incurs a training cost of $1.22\S$ and an inference cost of $2.62\Phi$ . In comparison, EvoFlow requires only $37.5\%$ of the training cost and $19.5\%$ of the inference cost: this is because EvoFlow ’s workflows do not continuously rely on the most expensive $\mathtt{q w e n}{-}2\cdot5{-}72\mathtt{b}$ , and instead opt for more economical models such as $11\mathtt{a m a}-3.1$ or herme $-3\r-70\mathrm{b}$ when appropriate, without compromising performance.  

![](https://cdn-mineru.openxlab.org.cn/extract/f6fc2303-638d-4bb4-901b-d70af57f7b1c/3c10749039c15dfeb6a30baace530b717203bf832f6ade78736c2536b9e46f26.jpg)  
Figure 4. The cost-performance plane of workflows from EvoFlow, DyLAN, and AFlow.  

![](https://cdn-mineru.openxlab.org.cn/extract/f6fc2303-638d-4bb4-901b-d70af57f7b1c/e3f615e462fb4ffb5a8ff87e807a692888fc387796aeb9c5b7c14d1324b1480f.jpg)  
Figure 5. The ablation study of EvoFlow.  

![](https://cdn-mineru.openxlab.org.cn/extract/f6fc2303-638d-4bb4-901b-d70af57f7b1c/9396b25e21869db8fae86cd9161bf672c4e450510f0b5c42a612da8f9de6cbef.jpg)  
Figure 6. The parameter sensitivity analysis of EvoFlow. The unit of cost per query (right) and performance (left) is $10^{-3}\cdot\S$ and accuracy $(\%)$ , respectively.  

We further visualize the optimized heterogeneous population of EvoFlow in Figure 4. It can be observed that EvoFlow forms a Pareto front in this performance-cost plane. The population begins with simple and inexpensive workflows, consisting solely of basic I/O and self-refine (accuracy $\textcircled{a}38.7\%$ , $\mathrm{cost}\ @0.00018\S,$ ), and progresses to more complex workflows incorporating multi-agent debate. The most high-performing workflows include iterative generation and ensembling, but at the cost of higher per-query token consumption (accuracy $\textcircled{a}72.57\%$ , $\mathrm{cost}@0.0037\S)$ . This highlights EvoFlow’s query-aware paradigm: for simple queries, it intelligently selects economical workflows for rapid completion, while for more complex ones, it leverages sophisticated workflows to address the increased demands.  

## 5.4. Framework Analysis  

Ablation Study We perform an ablation study on four variants of EvoFlow: w/o tag, where tag-based retrieval is removed and replaced with random selection in Equation (10); ${\mathbf{\nabla}}w/{\mathbf{\partial}}$ LLM mutation, where Equation (12) is discarded; ${\mathbf{}}w/{\mathbf{}}0$ prompt mutation, where Equation (13) is removed; and ${\mathbf{\nabla}}w/\mathbf{\partial}\mathbf{\partial} $ operator mutation, where Equation (14) is discarded. The results from Figure 5 reveal that removing tag-based retrieval and LLM mutation consistently leads to performance degradation and greater variance. This is because EvoFlow becomes heavily influenced by random parent workflow selection and the LLM backbone chosen during individual initialization. Removing operator mutation, which eliminates the potential for creating new, creative operators, results in a performance drop of $3.5\%\sim7.3\%$ .  

Sensitivity Analysis We perform an ablation study on three key parameters of EvoFlow: the number of selected parents $K$ in Equation (10), the number of tags per individual $\kappa$ in Equation (9), and the population size $N$ . As shown in Figure 6, (1) both too small and large $K$ result in performance degradation, likely because a small $K$ reduces offspring diversity, while a large $K$ challenges the LLM’s ability to aggregate multiple workflows; (2) increasing the population size $N$ consistently improves performance, with a gain of $3.1\%$ from $N=5$ to $N=25$ . However, a larger population also increases complexity, with the perquery cost rising from $8e\mathrm{~-~}4$ to $2e-3$ . Balancing costeffectiveness, we set $N=15$ across all experiments.  

# 6. Conclusion  

In this paper, we shift the paradigm of autonomous multi-agent workflow search from single-objective to costeffectiveness-driven multi-objective optimization. Building on niching-based evolutionary algorithms, we propose EvoFlow, an autonomous framework that evolves a population of heterogeneous, complexity-adapted agentic workflows. Extensive experiments across six benchmarks demonstrate the superior performance of EvoFlow with significantly lower token costs.  

# Impact Statement  

Ethical impacts. We affirm that our proposed EvoFlow method poses no ethical concerns in terms of its motivation, design, experiments, and data usage. The method is built with a focus on fostering advancements in multi-agent systems, ensuring its responsible contribution to scientific research and the development of more efficient and customized solutions.  

Expected societal implications. EvoFlow presents a transformative approach to multi-agent systems by promoting the automation of heterogeneous agent workflows, optimizing for both performance and cost. By facilitating the use of diverse models for task-specific solutions, it offers new avenues for deploying more adaptive and scalable systems in various practical domains.  

# References  

Austin, J., Odena, A., Nye, M., Bosma, M., Michalewski, H., Dohan, D., Jiang, E., Cai, C., Terry, M., Le, Q., et al. Program synthesis with large language models. arXiv preprint arXiv:2108.07732, 2021.   
Cetnarowicz, K., Kisiel-Dorohinicki, M., and Nawarecki, E. The application of evolution process in multi-agent world to the prediction system. In Proceedings of the Second International Conference on Multi-Agent Systems, ICMAS, volume 96, pp. 26–32, 1996.   
Chen, G., Dong, S., Shu, Y., Zhang, G., Sesay, J., Karlsson, B. F., Fu, J., and Shi, Y. Autoagents: A framework for automatic agent generation. CoRR, abs/2309.17288, 2023a. doi: 10.48550/ARXIV.2309.17288. URL https: //doi.org/10.48550/arXiv.2309.17288.   
Chen, G., Dong, S., Shu, Y., Zhang, G., Sesay, J., Karlsson, B. F., Fu, J., and Shi, Y. Autoagents: A framework for automatic agent generation. arXiv preprint arXiv:2309.17288, 2023b.   
Chen, L., Zaharia, M., and Zou, J. Frugalgpt: How to use large language models while reducing cost and improving performance. arXiv preprint arXiv:2305.05176, 2023c.   
Chen, M., Tworek, J., Jun, H., Yuan, Q., Ponde de Oliveira Pinto, H., Kaplan, J., Edwards, H., Burda, Y., Joseph, N., Brockman, G., Ray, A., Puri, R., Krueger, G., Petrov, M., Khlaaf, H., Sastry, G., Mishkin, P., Chan, B., Gray, S., Ryder, N., Pavlov, M., Power, A., Kaiser, L., Bavarian, M., Winter, C., Tillet, P., Petroski Such, F., Cummings, D., Plappert, M., Chantzis, F., Barnes, E., Herbert-Voss, A., Hebgen Guss, W., Nichol, A., Paino, A., Tezak, N., Tang, J., Babuschkin, I., Balaji, S., Jain, S., Saunders, W., Hesse, C., Carr, A. N., Leike, J., Achiam, J., Misra, V., Morikawa, E., Radford, A., Knight, M., Brundage, M., Murati, M., Mayer, K., Welinder, P., McGrew, B., Amodei, D., McCandlish, S., Sutskever, I., and Zaremba, W. Evaluating large language models trained on code, July 01, 2021 2021.   
Chen, W., Su, Y., Zuo, J., Yang, C., Yuan, C., Qian, C., Chan, C.-M., Qin, Y., Lu, Y., Xie, R., Liu, Z., Sun, M., and Zhou, J. Agentverse: Facilitating multi-agent collaboration and exploring emergent behaviors in agents, 2023d.   
Chollet, F. On the measure of intelligence. arXiv preprint arXiv:1911.01547, 2019.   
Clune, J. Ai-gas: Ai-generating algorithms, an alternate paradigm for producing general artificial intelligence. arXiv preprint arXiv:1905.10985, 2019.   
Cobbe, K., Kosaraju, V., Bavarian, M., Chen, M., Jun, H., Kaiser, L., Plappert, M., Tworek, J., Hilton, J., Nakano, R., Hesse, C., and Schulman, J. Training verifiers to solve math word problems. arXiv prepring, abs/2110.14168, 2021.   
Devlin, J. Bert: Pre-training of deep bidirectional transformers for language understanding. arXiv preprint arXiv:1810.04805, 2018.   
Du, Y., Li, S., Torralba, A., Tenenbaum, J. B., and Mordatch, I. Improving factuality and reasoning in language models through multiagent debate. CoRR, abs/2305.14325, 2023.   
Feng, T., Shen, Y., and You, J. Graphrouter: A graph-based router for llm selections. arXiv preprint arXiv:2410.03834, 2024.   
Fernando, C., Banarse, D., Michalewski, H., Osindero, S., and Rocktaschel, T. Promptbreeder: Self-referential self-improvement via prompt evolution. arXiv preprint arXiv:2309.16797, 2023.   
Fu, Y., Peng, H., Sabharwal, A., Clark, P., and Khot, T. Complexity-based prompting for multi-step reasoning. In The Eleventh International Conference on Learning Representations, 2022.   
Guo, Q., Wang, R., Guo, J., Li, B., Song, K., Tan, X., Liu, G., Bian, J., and Yang, Y. Connecting large language models with evolutionary algorithms yields powerful prompt optimizers. arXiv preprint arXiv:2309.08532, 2023.   
He, Z., Cao, P., Chen, Y., Liu, K., Li, R., Sun, M., and Zhao, J. Lego: A multi-agent collaborative framework with roleplaying and iterative feedback for causality explanation generation. In Findings of the Association for Computational Linguistics: EMNLP 2023, pp. 9142–9163, 2023.   
Hendrycks, D., Burns, C., Basart, S., Zou, A., Mazeika, M., Song, D., and Steinhardt, J. Measuring massive multitask language understanding. Proceedings of the International Conference on Learning Representations (ICLR), 2021a.   
Hendrycks, D., Burns, C., Kadavath, S., Arora, A., Basart, S., Tang, E., Song, D., and Steinhardt, J. Measuring mathematical problem solving with the math dataset. NeurIPS, 2021b.   
Hong, S., Zheng, X., Chen, J., Cheng, Y., Wang, J., Zhang, C., Wang, Z., Yau, S. K. S., Lin, Z., Zhou, L., Ran, C., Xiao, L., and Wu, C. Metagpt: Meta programming for multi-agent collaborative framework, August 01, 2023 2023.   
Hong, S., Lin, Y., Liu, B., Liu, B., Wu, B., Zhang, C., Wei, C., Li, D., Chen, J., Zhang, J., et al. Data interpreter: An llm agent for data science. arXiv preprint arXiv:2402.18679, 2024.   
Hu, Q. J., Bieker, J., Li, X., Jiang, N., Keigwin, B., Ranganath, G., Keutzer, K., and Upadhyay, S. K. Routerbench: A benchmark for multi-llm routing system. arXiv preprint arXiv:2403.12031, 2024a.   
Hu, S., Lu, C., and Clune, J. Automated design of agentic systems. arXiv preprint arXiv:2408.08435, 2024b.   
Hu, Y., Cai, Y., Du, Y., Zhu, X., Liu, X., Yu, Z., Hou, Y., Tang, S., and Chen, S. Self-evolving multi-agent collaboration networks for software development. arXiv preprint arXiv:2410.16946, 2024c.   
Jiang, D., Ren, X., and Lin, B. Y. LLM-blender: Ensembling large language models with pairwise ranking and generative fusion. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 14165–14178, Toronto, Canada, July 2023. Association for Computational Linguistics.   
Jin, Y., Shen, X., Peng, H., Liu, X., Qin, J., Li, J., Xie, J., Gao, P., Zhou, G., and Gong, J. Surrealdriver: Designing generative driver agent simulation framework in urban contexts based on large language model, 2023.   
Khattab, O., Singhvi, A., Maheshwari, P., Zhang, Z., Santhanam, K., Vardhamanan, S., Haq, S., Sharma, A., Joshi, T. T., Moazam, H., et al. Dspy: Compiling declarative language model calls into self-improving pipelines. arXiv preprint arXiv:2310.03714, 2023.   
Lee, K.-H., Fischer, I., Wu, Y.-H., Marwood, D., Baluja, S., Schuurmans, D., and Chen, X. Evolving deeper llm thinking. arXiv preprint arXiv:2501.09891, 2025.   
Li, G., Hammoud, H., Itani, H., Khizbullin, D., and Ghanem, B. CAMEL: communicative agents for ”mind” explo  

ration of large language model society. In NeurIPS, 2023.  

Li, Z. and Liu, J. A multi-agent genetic algorithm for community detection in complex networks. Physica A: Statistical Mechanics and its Applications, 449:336–347, 2016.   
Li, Z., Zang, Q., Ma, D., Guo, J., Zheng, T., Liu, M., Niu, X., Wang, Y., Yang, J., Liu, J., et al. Autokaggle: A multi-agent framework for autonomous data science competitions. arXiv preprint arXiv:2410.20424, 2024.   
Liang, T., He, Z., Jiao, W., Wang, X., Wang, Y., Wang, R., Yang, Y., Tu, Z., and Shi, S. Encouraging divergent thinking in large language models through multi-agent debate. CoRR, abs/2305.19118, 2023.   
Liu, Z., Chen, B., Zhou, H., Koushik, G., Hebert, M., and Zhao, D. Mapper: Multi-agent path planning with evolutionary reinforcement learning in mixed dynamic environments. In 2020 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), pp. 11748–11754. IEEE, 2020.   
Liu, Z., Zhang, Y., Li, P., Liu, Y., and Yang, D. Dynamic llmagent network: An llm-agent collaboration framework with agent team optimization. CoRR, abs/2310.02170, 2023.   
Madaan, A., Tandon, N., Gupta, P., Hallinan, S., Gao, L., Wiegreffe, S., Alon, U., Dziri, N., Prabhumoye, S., Yang, Y., Gupta, S., Majumder, B. P., Hermann, K., Welleck, S., Yazdanbakhsh, A., and Clark, P. Self-refine: Iterative refinement with selffeedback. In NeurIPS, 2023. URL http://papers. nips.cc/paper_files/paper/2023/hash/ 91edff07232fb1b55a505a9e9f6c0ff3-Abstract-Confere html.   
Mikolov, T., Chen, K., Corrado, G., and Dean, J. Efficient estimation of word representations in vector space. arXiv preprint arXiv:1301.3781, 2013.   
Nakajima, Y. Babyagi. https://github.com/ yoheinakajima/babyagi, 2023.   
Pennington, J., Socher, R., and Manning, C. D. Glove: Global vectors for word representation. In Proceedings of the 2014 conference on empirical methods in natural language processing (EMNLP), pp. 1532–1543, 2014.   
Qian, C., Xie, Z., Wang, Y., Liu, W., Dang, Y., Du, Z., Chen, W., Yang, C., Liu, Z., and Sun, M. Scaling largelanguage-model-based multi-agent collaboration. arXiv preprint arXiv:2406.07155, 2024.   
Reimers, N. Sentence-bert: Sentence embeddings using siamese bert-networks. arXiv preprint arXiv:1908.10084, 2019.   
Reworkd. Agentgpt. https://github.com/ reworkd/AgentGPT, 2023.   
Richards, T. B. and et al. Auto-gpt: An autonomous gpt-4 experiment. https://github.com/ Significant-Gravitas/Auto-GPT, 2023.   
Romera-Paredes, B., Barekatain, M., Novikov, A., Balog, M., Kumar, M. P., Dupont, E., Ruiz, F. J., Ellenberg, J. S., Wang, P., Fawzi, O., et al. Mathematical discoveries from program search with large language models. Nature, 625 (7995):468–475, 2024.   
Roy, S. and Roth, D. Solving general arithmetic word problems. arXiv preprint arXiv:1608.01413, 2016.   
Saad-Falcon, J., Lafuente, A. G., Natarajan, S., Maru, N., Todorov, H., Guha, E., Buchanan, E. K., Chen, M., Guha, N., Re, C., et al. Archon: An architecture search framework for inference-time techniques. arXiv preprint arXiv:2409.15254, 2024.   
Shang, Y., Li, Y., Zhao, K., Ma, L., Liu, J., Xu, F., and Li, Y. Agentsquare: Automatic llm agent search in modular design space. arXiv preprint arXiv:2410.06153, 2024.   
Shen, Y., Song, K., Tan, X., Li, D., Lu, W., and Zhuang, Y. Hugginggpt: Solving ai tasks with chatgpt and its friends in hugging face. Advances in Neural Information Processing Systems, 36, 2024.   
Shi, Z., Wang, Y., Yin, F., Chen, X., Chang, K.-W., and Hsieh, C.-J. Red teaming language model detectors with language models. Transactions of the Association for Computational Linguistics, 12:174–189, 2024.   
Shinn, N., Labash, B., and Gopinath, A. Reflexion: an autonomous agent with dynamic memory and selfreflection. arXiv preprint, abs/2303.11366, 2023. doi: 10. 48550/arXiv.2303.11366. URL https://doi.org/ 10.48550/arXiv.2303.11366.   
Shridhar, M., Yuan, X., Cote, M.-A., Bisk, Y., Trischler, A., and Hausknecht, M. {ALFW}orld: Aligning text and embodied environments for interactive learning. In International Conference on Learning Representations, 2021. URL https://openreview.net/forum? id $=0$ IOX0YcCdTn.   
Song, C. H., Wu, J., Washington, C., Sadler, B. M., Chao, W.-L., and Su, Y. Llm-planner: Few-shot grounded planning for embodied agents with large language models. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 2998–3009, 2023.   
Tao, N., Ventresque, A., and Saber, T. Program synthesis with generative pre-trained transformers and grammarguided genetic programming grammar. In 2023 IEEE Latin American Conference on Computational Intelligence (LA-CCI), pp. 1–6. IEEE, 2023.   
Tyson, L. D. and Zysman, J. Automation, ai & work. Daedalus, 151(2):256–271, 2022.   
Wang, G., Xie, Y., Jiang, Y., Mandlekar, A., Xiao, C., Zhu, Y., Fan, L., and Anandkumar, A. Voyager: An OpenEnded Embodied Agent with Large Language Models. arXiv e-prints, art. arXiv:2305.16291, May 2023.   
Wang, L., Ma, C., Feng, X., Zhang, Z., Yang, H., Zhang, J., Chen, Z., Tang, J., Chen, X., Lin, Y., Zhao, W. X., Wei, Z., and Wen, J. A survey on large language model based autonomous agents. Front. Comput. Sci., 18, 2024.   
Wang, W., Wei, F., Dong, L., Bao, H., Yang, N., and Zhou, M. Minilm: Deep self-attention distillation for task-agnostic compression of pre-trained transformers. Advances in Neural Information Processing Systems, 33: 5776–5788, 2020.   
Wang, X., Wei, J., Schuurmans, D., Le, Q. V., Chi, E. H., Narang, S., Chowdhery, A., and Zhou, D. Selfconsistency improves chain of thought reasoning in language models. In The Eleventh International Conference on Learning Representations, 2023a.   
Wang, Z., Mao, S., Wu, W., Ge, T., Wei, F., and Ji, H. Unleashing cognitive synergy in large language models: A task-solving agent through multi-persona selfcollaboration, July 01, 2023 2023b. work in progress.   
Wei, J., Wang, X., Schuurmans, D., Bosma, M., Ichter, B., Xia, F., Chi, E., Le, Q., and Zhou, D. Chain-of-thought prompting elicits reasoning in large language models, January 01, 2022 2022.   
White, C., Safari, M., Sukthanker, R., Ru, B., Elsken, T., Zela, A., Dey, D., and Hutter, F. Neural architecture search: Insights from 1000 papers. arXiv preprint arXiv:2301.08727, 2023.   
Wu, Q., Bansal, G., Zhang, J., Wu, Y., Zhang, S., Zhu, E., Li, B., Jiang, L., Zhang, X., and Wang, C. Autogen: Enabling next-gen llm applications via multi-agent conversation framework, August 01, 2023 2023.   
Xu, B., Yang, A., Lin, J., Wang, Q., Zhou, C., Zhang, Y., and Mao, Z. Expertprompting: Instructing large language models to be distinguished experts. arXiv preprint arXiv:2305.14688, 2023.   
Xu, H., Chen, Y., Du, Y., Shao, N., Wang, Y., Li, H., and Yang, Z. Gps: Genetic prompt search for efficient fewshot learning. arXiv preprint arXiv:2210.17041, 2022.   
Zhu, Y., Qiao, S., Ou, Y., Deng, S., Zhang, N., Lyu, S., Shen, Y., Liang, L., Gu, J., and Chen, H. Knowagent: Knowledge-augmented planning for llm-based agents. arXiv preprint arXiv:2403.03101, 2024b.   
Zhuge, M., Wang, W., Kirsch, L., Faccio, F., Khizbullin, D., and Schmidhuber, J. Gptswarm: Language agents as optimizable graphs. In Forty-first International Conference on Machine Learning, 2024.   
Zitzler, E. and Kinzli, S. Indicator-based selection in multiobjective search. In International conference on parallel problem solving from nature, pp. 832–842. Springer, 2004.   
Yao, S., Yu, D., Zhao, J., Shafran, I., Griffiths, T. L., Cao, Y., and Narasimhan, K. Tree of thoughts: Deliberate problem solving with large language models, May 01, 2023 2023a.   
Yao, S., Zhao, J., Yu, D., Du, N., Shafran, I., Narasimhan, K. R., and Cao, Y. React: Synergizing reasoning and acting in language models. In The Eleventh International Conference on Learning Representations, 2023b.   
Yuan, S., Song, K., Chen, J., Tan, X., Li, D., and Yang, D. Evoagent: Towards automatic multi-agent generation via evolutionary algorithms. arXiv preprint arXiv:2406.14228, 2024.   
Zhang, G., Yue, Y., Li, Z., Yun, S., Wan, G., Wang, K., Cheng, D., Yu, J. X., and Chen, T. Cut the crap: An economical communication pipeline for llm-based multiagent systems. arXiv preprint arXiv:2410.02506, 2024a.   
Zhang, G., Yue, Y., Sun, X., Wan, G., Yu, M., Fang, J., Wang, K., and Cheng, D. G-designer: Architecting multiagent communication topologies via graph neural networks. arXiv preprint arXiv:2410.11782, 2024b.   
Zhang, J., Xu, X., and Deng, S. Exploring collaboration mechanisms for llm agents: A social psychology view. arXiv preprint arXiv:2310.02124, 2023.   
Zhang, J., Xiang, J., Yu, Z., Teng, F., Chen, X., Chen, J., Zhuge, M., Cheng, X., Hong, S., Wang, J., et al. Aflow: Automating agentic workflow generation. arXiv preprint arXiv:2410.10762, 2024c.   
Zhang, Z., Zhang, A., Li, M., and Smola, A. Automatic chain of thought prompting in large language models. arXiv preprint arXiv:2210.03493, 2022.   
Zheng, C., Liu, Z., Xie, E., Li, Z., and Li, Y. Progressivehint prompting improves reasoning in large language models, April 01, 2023 2023a. Tech Report.   
Zheng, H. S., Mishra, S., Chen, X., Cheng, H.-T., Chi, E. H., Le, Q. V., and Zhou, D. Take a step back: Evoking reasoning via abstraction in large language models. arXiv preprint arXiv:2310.06117, 2023b.   
Zhong, W., Guo, L., Gao, Q., Ye, H., and Wang, Y. Memorybank: Enhancing large language models with long-term memory. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 38, pp. 19724–19731, 2024.   
Zhu, J.-P., Cai, P., Xu, K., Li, L., Sun, Y., Zhou, S., Su, H., Tang, L., and Liu, Q. Autotqa: Towards autonomous tabular question answering through multi-agent large language models. Proceedings of the VLDB Endowment, 17 (12):3920–3933, 2024a.  

# A. Notations  

We present a comprehensive review of the commonly used notations and their definitions in Table 4.  

Table 4. Notation and Definitions   


<html><body><table><tr><td>Notation</td><td>Definition</td></tr><tr><td>Ii</td><td>An LLM-invoking node</td></tr><tr><td>Pi</td><td>The prompt content of I</td></tr><tr><td>P</td><td>The feasible prompt space</td></tr><tr><td>Mi</td><td>The base LLM invoked by I</td></tr><tr><td>Ti</td><td>Temperature of M</td></tr><tr><td>M = {M1,.·· ,Mm|}</td><td>LLM pool</td></tr><tr><td>I = M × P × R[0,1]</td><td>The feasible space for invoking nodes</td></tr><tr><td>Oj = (T,)</td><td>An operator node composed of multiple invoking nodes</td></tr><tr><td>T = {Ii,...,In}</td><td>The selected invoking nodes in O§</td></tr><tr><td> I × I</td><td>The connectivity of operator nodes in O§</td></tr><tr><td></td><td>The feasible space of operator nodes</td></tr><tr><td>g =(O$,ε) = ($,ε°)</td><td>An agentic workflow</td></tr><tr><td>O$ e O</td><td>A subset of operator nodes used in g</td></tr><tr><td>IS ∈I</td><td>A subset of invoking nodes selected in g</td></tr><tr><td>u(9,T)</td><td>An evaluator function assessing G's performance in task domain T</td></tr><tr><td>c(9,T)</td><td>An evaluator function assessing G's cost in task domain T</td></tr><tr><td>g*</td><td>The best workflow searched by baseline methods</td></tr><tr><td>g*</td><td>The Pareto-optimal set of agentic workflows balancing cost and performance</td></tr><tr><td>P(t) = {91,92,..· ,9N}</td><td>A population of N agentic workflows at the t-th iteration</td></tr><tr><td>#</td><td>The i-th tag of workflow Gk</td></tr><tr><td>v(.)</td><td>Text embedding function</td></tr><tr><td>S(9lq)</td><td>The similarity score of workflow G with respect to query q</td></tr><tr><td>gCt)</td><td>The generated offspring workflow at the t-th iteration</td></tr><tr><td>μ()</td><td>LLM mutation function</td></tr><tr><td>μP(-)</td><td>Prompt mutation function</td></tr><tr><td>μ()</td><td>Operator mutation function</td></tr><tr><td>g</td><td>A mutated workflow</td></tr><tr><td></td><td>A mutated workflow at the t-th iteration</td></tr><tr><td>PNA = {9ql,..· ,9qE}</td><td>The identified niche area comprising E individuals</td></tr><tr><td>c(t)(Gi)</td><td>The cumulative cost of G; at the t-th iteration</td></tr><tr><td>u(t)(Gi)</td><td>The cumulative performance of G at the t-th iteration</td></tr><tr><td>I(, )</td><td>Pareto dominance-preserving binary indicator</td></tr><tr><td>F(G)</td><td>The fitness value of g</td></tr></table></body></html>  

# B. Algorithm Table  

We conclude the overall algorithm procedure of EvoFlow in Algorithm 1.  

# C. Optimization Objective  

For a better understanding of the multi-objective optimization in the context of agentic workflows, here, we define some key concepts like dominance, Pareto optimality, and the Pareto set.  

A workflow $\mathcal{G}_{1}$ is said to dominate another workflow $\mathcal{G}_{2}$ if and only if:  

$$
\forall i\in\{1,2\},f_{i}(\mathcal{G}_{1})\geq f_{i}(\mathcal{G}_{2}),\mathrm{and}\exists i\mathrm{such}\mathrm{that}f_{i}(\mathcal{G}_{1})>f_{i}(\mathcal{G}_{2}),
$$  

where $f_{1}(\cdot)=u(\cdot,T)$ represents the utility or performance metric, and $f_{2}(\cdot)=-c(\cdot,T)$ denotes the negative system cost.  

Algorithm 1 Algorithm workflow of EvoFlow   
Input :A dataset $\mathcal{D}$ containing training set $\mathcal{D}_{\mathrm{train}}$ and test set $\mathcal{D}_{\mathrm{test}}$ , Operator set $\mathcal{O}$   
Output :The well-optimized, diverse workflow population $\mathbf{P}^{|\mathcal{D}_{\mathrm{train}}|}$   
$/\star$ Initialize workflow population \*/   
for individual $k\gets1$ to $N$ do $\mathcal{G}_{k}\gets\{\mathcal{O}_{k},\mathcal{E}_{k}^{a}\},\mathcal{E}_{k}^{a}\subseteq\mathcal{O}_{k}\times\mathcal{O}_{k};$ ; ▷ Eq. 8 $/\star$ Assigning utility tags \*/ $\{\varkappa_{1}^{k},\cdot\cdot\cdot,\varkappa_{\kappa}^{k}\}\leftarrow f_{\mathrm{tag}}(\mathcal{G}_{k})$ ; ▷ Eq. 9   
end   
Obtain initialized population ${\bf P}^{(0)}=\{\mathcal{G}_{1},\mathcal{G}_{2},\cdot\cdot\cdot,\mathcal{G}_{N}\}$   
for query $q_{t}$ in $\mathcal{D}_{\mathrm{train}}$ do $/\star$ Retrieve relevant workflows via tag-based similarity \*/ Locate parent workflows: $\{\mathcal{G}_{t1},\cdot\cdot\cdot,\mathcal{G}_{t K}\}=\operatorname{TopK}\big(S(\{\mathcal{G}_{i}\}_{i=1}^{N}\mid q_{t}),K\big)$ ; ▷ Eq. 10 $/\star$ Crossover and generate the offspring workflow $\star/$ Evolve the offspring workflow via the Crossover function: $\mathcal{G}_{\circ}^{(t)}\gets\mathrm{Crossover}(\mathcal{G}_{t1},\cdot\cdot\cdot,\mathcal{G}_{t K})$ ; $\triangleright$ Eq. 11 Mutate the offspring with three mutation functions: LLM mutation $\mu^{l}(\cdot)$ , Prompt Mutation $\mu^{p}(\cdot)$ and Operator Mutation $\mu^{o}(\cdot)$ ; obtain the mutated offspring $\mathcal{G}_{\circledcirc}^{(t)}$ ; ▷ Equations (12) to (14) $/\star$ Niching-based selection & elimination $\star/$ Identify the niching area $\mathbf{P}^{N A}$ : $\mathbf{P}^{N A}=\{\mathcal{G}_{q1},\cdots,\mathcal{G}_{q E}\}=\mathrm{TopK}\left(\{-\mathrm{Rank}(\mathcal{G}_{i})\}_{i=1}^{N},E\right)$ ; ▷ Eq. 15 $/\star$ Execute parents, offsprings and niching-area workflows $\star/$ for workflow $\mathcal{G}_{i}\in\mathbf{P}^{N A}\cup\{\mathcal{G}_{t i}\}_{i=1}^{K}\cup\{\mathcal{G}_{\circledcirc}^{(t)}\mathbf{do}$ Execute $\mathcal{G}_{i}$ on query $q$ $/\star$ Record and update its cost \*/ $c^{(t)}({\mathcal{G}}_{i})=1/t_{i}^{\prime}\left(c^{(t-1)}({\mathcal{G}}_{i})\cdot t_{i}^{\prime}+c({\mathcal{G}}_{i}\mid q_{t})\right)$ $/\star$ Record and update its performance \*/ $p^{(t)}(\mathcal{G}_{i})=1/t_{i}^{\prime}\left(p^{(t-1)}(\mathcal{G}_{i})\cdot t_{i}^{\prime}+p(\mathcal{G}_{i}\mid q_{t})\right)$ end for workflow $\mathcal{G}_{i}\in\mathbf{P}^{N A}\cup\{\mathcal{G}_{t i}\}_{i=1}^{K}\cup\{\mathcal{G}_{\circledcirc}^{(t)}\mathbf{do}$ $/\star$ Calculate each individual $\mathbf{\mu}_{\cdot}\mathbf{\Lambda}_{}^{\prime}\le$ fitness value \*/ Calculate the fitness value F(G) = PNA $\begin{array}{r}{\mathcal{F}(\mathcal{G})=\sum_{\mathcal{G}\in\mathbf{P}^{N A}}\left(\exp\frac{\mathbf{I}(\mathcal{G},\mathcal{G}_{\varnothing}^{(t)})}{\varphi\cdot\mathbf{I}^{\operatorname*{max}}}\right)}\end{array}$ end Locate the individual with the largest ${\mathcal{F}}({\mathcal{G}})$ as $\mathcal{G}^{\mathrm{worst}}$ $/\star$ Update the population Update the population $\mathbf{P}^{(t+1)}\gets\mathbf{P}^{(t)}\setminus\mathcal{G}^{\mathrm{worst}}\cup\mathcal{G}_{\odot}^{(t)}$ /\* Notably, if the generated workflow performs suboptimally, i.e., it does not Pareto dominate any existing workflows, it is unlikely to be accepted into the population.   
end  

This means $\mathcal{G}_{1}$ performs at least as well as $\mathcal{G}_{2}$ in all objectives and strictly better in at least one objective.  

A workflow $\mathcal{G}^{\star}\in\mathcal{H}(\mathcal{T},\mathcal{E})$ is considered Pareto optimal if there does not exist any other workflow $\mathcal{G}^{\prime}\in\mathcal{H}(\mathcal{T},\mathcal{E})$ that dominates $\mathcal{G^{\star}}$ . The collection of all Pareto optimal workflows forms the Pareto set:  

$$
\mathcal{G}_{\mathrm{PF}}^{*}=\{\mathcal{G}\in\mathcal{H}(\mathbb{Z},\mathcal{E})|\not\exists\mathcal{G}^{\prime}\in\mathcal{H}(\mathbb{Z},\mathcal{E}),\mathcal{G}^{\prime}\mathrm{dominates}\mathcal{G}\}.
$$  

The corresponding objective space of these workflows defines the Pareto front $(\mathbf{PF})$ , which represents the trade-off surface between performance and cost:  

$$
\mathrm{PF}=\left\{\mathbf{f}(\mathcal{G})=[u(\mathcal{G},T),-c(\mathcal{G},T)]^{\top}\mid\mathcal{G}\in\mathcal{G}_{\mathrm{PF}}^{*}\right\}.
$$  

In the context of agentic workflows, identifying Pareto optimal solutions is critical as it enables the selection of workflows that provide the best possible trade-off between task performance and system cost. These solutions ensure that the agents, composed of invoking nodes and operator nodes, operate efficiently while maintaining high utility for the target tasks. Furthermore, the Pareto set provides diverse design options, offering flexibility in adapting workflows to varying operational constraints and objectives.  

# D. Operator Repository  

In this section, we detail the initialization of operator nodes, which can be categorized into the following seven types:  

1. Chain-of-Thought (CoT). CoT (Wei et al., 2022) reasoning encourages the LLM to think step by step rather than directly outputting an answer. This approach enhances its capability to solve complex problems through intermediate reasoning steps, improving task handling and providing greater transparency in the decision-making process.   
2. LLM-Debate. LLM-Debate (Du et al., 2023) allows multiple LLMs to debate, leveraging diverse perspectives to identify better solutions. In practice, we initialize three debaters and permit up to two debate rounds.   
3. Take a Step Back. As proposed by Zheng et al. (2023b), this operator prompts the LLM to first consider the principles underlying the task. By focusing on foundational principles, the model enhances its reasoning and delivers more accurate solutions.   
4. Self-Consistency. Adopting the methodology from Wang et al. (2023a), this operator aggregates five CoT reasoning paths and determines the final answer through majority voting.   
5. Self-Refine. Following Madaan et al. (2023), this operator initially generates an answer using CoT reasoning, then prompts the agent to self-reflect iteratively. We set a maximum of five refinement iterations.   
6. Ensemble. Inspired by LLM-Blender (Jiang et al., 2023), this operator involves three LLM-powered agents from different sources outputting answers to the same query. The pairwise ranking is used to evaluate and aggregate their responses into a final solution.   
7. ReAct. Following (Yao et al., 2023b), this operator enables the agent to leverage versatile tools, including code interpreter, web searching, external knowledge database, etc., to handle diverse user demands.   
8. ExpertPrompt. Similar to AutoGPT (Richards & et al., 2023) and expert prompting (Xu et al., 2023), this operator employs dynamic control flows to allow the agent to decide which expert should be utilized for the task.   
We respectfully note that the selection of these operators is highly customizable, allowing users the flexibility to incorporate   
their desired operators into the operator repository of EvoFlow.  

# E. Prompt Repository  

## E.1. Tag Generation Prompt  

# Tag generation  

TAG_PROMPT $=$ """   
\*\*Workflow Information $^{\star\star}$ $^{\star\star}$ Name: $\star\star$ {NAME} \*\*Description: $\star\star$ {DESCRIPTION} \*\*Code: $\star\star$ {CODE}  

$^{\star\star}$ Your Task\*\*  

This workflow is designed to address specific problems in the MATH dataset, which contains challenging, competition-level mathematics problems.   
Please generate five relevant tags for this workflow, focusing on the academic disciplines involved and the workflow’s level of complexity.  

Here is the task this workflow has successfully solved:  

{TASK}  

$^{\star\star}$ Your goal is to design tags for this workflow so that, when a similar task arises, the workflow’s tags will have the highest cosine similarity score with the new task’s tags. $^{\star\star}$  

\*\*Examples\*\*  

1. \*\*Example 1\*\* - \*\*Code:\*\* {MATH} \*\*Tags:\*\* Right Triangle, Intermediate Combinatorics, Intermediate Computational Mathematics, Intermediate Permutations, Intermediate Geometry  

2. \*\*Example 2\*\* \*\*Code: $\star\star$ {CUSTOM} \*\*Tags:\*\* Number Theory, Integer Properties, Relatively Prime, Prime Factors, Simple Mathematical Problems  

\*\*Output Format $\star\star$  

Provide $\star\star5$ tags\*\*, separated by commas.   
- Tags should reflect the primary academic disciplines or difficulty level associated with the MATH dataset.  

$^{\star\star}$ Guidelines\*\*  

1. $^{\star\star}$ Focus Areas:\*\*  

- \*\*Academic Disciplines: $\star\star$ Identify the main fields related to the MATH dataset (e.g., Linear Algebra, Mathematics, Calculus). \*\*Problem Difficulty: $\star\star$ Assess the complexity level of the problems (e.g., Beginner, Intermediate, Advanced).  

2. \*\*Formatting:\*\* - Do not include any additional text or explanations. - Ensure the output is a single line containing exactly five tags.  

$^{\star\star}$ Important Notes\*\*  

- \*\*Avoid General Tags:\*\* Do not use overly broad tags such as Artificial Intelligence, Natural Language Processing, or Cognitive Science.   
- \*\*Relevance to MATH: $^{\star\star}$ Ensure the tags are specifically relevant to the MATH dataset’s focus on mathematical problems and their difficulty levels.  

\*\*WRONG Implementation Examples\*\*  

1. Artificial Intelligence, Natural Language Processing, Reasoning Systems, Advanced Problem Solving, Cognitive Science, Multi-Agent System, AI-enhanced Problem Solving, Problem Solving - \*Issue:\* These tags are too general and do not focus on the difficulty level or specific academic disciplines related to the MATH dataset.  

"""  

## E.2. Offspring Generation Prompt  

# Offspring generation  

PROMPT $=$  

You are an expert machine learning researcher specializing in the design of agentbased workflows. Your goal is to optimize existing architectures and create a highly efficient, effective, and economically viable multi-agent workflow that solves a specific query from the MATH dataset, which contains challenging, competition-level mathematics problems.  

Leverage your extensive knowledge of LLM prompting techniques and agent workflows from existing literature to analyze the provided architectures. Extract valuable insights and lessons, and draw inspiration from related LLM agent papers or research in other fields to design a novel, creative architecture. THINK OUTSIDE THE BOX.  

# ## Query:  

Your task is to develop an improved multi-agent workflow that surpasses all existing workflows in accurately and efficiently solving the following query. Consider the difficulty, complexity, and discipline of the query to structure an innovative multi-agent workflow best suited to solve it.   
{QUERY} ## Reference Multi-Agent Workflow:   
You have several multi-agent workflow designs to serve as references. {PARENTS}  

## Multi-Agent Communication Structure Design Instructions:  

To improve the efficiency, effectiveness, and cost-effectiveness of communication within the Multi-Agent workflow, refer to the following communication structures when designing the workflow. Successful implementations typically do not rely on complex frameworks or specialized libraries. Instead, they emphasize building with simple, composable patterns. Below are several structures you can consider. You are also encouraged to use your imagination and logical thinking to design even more suitable structures for solving the specific task:  

{STRUCTURES} ## Output Instruction: {OUTPUT_INSTRUCTION}  

Your response should be in JSON format, adhering to the structure demonstrated in the example below:   
{EXAMPLE}   
## Common Mistakes:   
Here are some common mistakes you might make:   
{WRONG_IMPLEMENTATION}   
"""  

## E.3. Mutation Prompt  

E.3.1. LLM MUTATION  

# LLM Mutation  

# """  

1. $^{\star\star}$ Large Language Model Mutation\*\* You can replace the LLM backbone that initializes the operators. Your options are limited to the following 4 choices: - meta-llama/llama-3.1-70b-instruct - qwen/qwen-2.5-72b-instruct - deepseek/deepseek-chat-v2.5 - nousresearch/hermes-3-llama-3.1-70b  

"""  

### E.3.2. PROMPT MUTATION  

# Prompt Mutation  

"""   
2. \*\*Prompt Mutation\*\* You can modify the prompts used by invoking nodes, such as incorporating few-shot examples or clarifying task instructions. Prompt mutation can enhance the clarity of the agent’s output. You can also create specific prompts to guide the operator in generating a logical response or facilitate communication between operators. - Write your own prompt and use it in the Custom method within the workflow: ‘‘‘python STRUCTION_PROMPT $=$ ’’’Provide a comprehensive, step-by-step solution to the given mathematical problem. Utilize existing mathematical knowledge to solve the problem. Your response should include: 1. A clear restatement of the problem. 2. An explanation of the mathematical concepts and theorems involved. 3. A detailed, logical progression of steps leading to the solution. 4. Clear explanations for each step, including the reasoning behind it. 5. All mathematical expressions and equations in LaTeX format. 6. Visual aids or diagrams if applicable (described in text). 7. Make sure the final answer displayed in a boxed LaTeX format." response $=$ await self.custom(input $=$ task, instruction $\mathbf{\varepsilon}=$ INSTRUCTION_PROMPT) ’’’ ‘‘‘ - You can also concatenate previously generated string results in the input to provide more comprehensive contextual information: ‘‘‘python response $=$ await self.custom(input $=$ task $^+$ f"xxx:{{xxx}}, xxx:{{xxx}}", instruction $\equiv$ INSTRUCTION_PROMPT) ‘‘‘ The output from the Custom method can be placed anywhere in the workflow: ‘‘‘python solution $=$ await self.generate(problem $_1=$ f"Here is the task: {{task}}, here is the response from other operators:{{response[’response’]}}") ‘‘‘ \*\*Note\*\*: - Avoid using single quotes in your code, as they may cause execution errors. - In the ‘custom‘ method, the input and instruction are directly concatenated ( instruction $^+$ input), and placeholders are not supported. Be sure to handle concatenation externally and add comments where necessary.   
"""  

### E.3.3. OPERATOR MUTATION  

# Operator Mutation  

"""3. \*\*Operator Mutation $\star\star$ You can add or remove operators from the existing reference workflows. Consider their performance and compatibility with the given task. Below are descriptions of the operators you can use. Initialize and call them properly, writing appropriate prompts to organize them and ensure they collaborate efficiently to solve the task. {OPERATORS} Additionally, remember to initialize operators in the ‘__init__‘ function before calling them!   
"""  

# F. History Management of EvoFlow  

## F.1. LLM Experience Pool  

To evaluate the historical performance of LLMs within agentic workflows, we construct an experience pool, denoted as $\mathcal{P}_{L L M}$ . This pool captures the interplay between LLM instances, prompts, and workflow configurations, providing a foundation for analyzing and refining their performance across diverse tasks.  

$\mathcal{P}_{L L M}$ captures both quantitative and qualitative evaluations of LLM’s behavior across workflows. For a given workflow $\mathcal{G}_{k}$ associated with task $q$ and ground-truth answer $a$ , the performance of an LLM $M_{i}$ is represented as $\mathcal{F}_{L L M}(M_{i},\mathcal{G}_{k})=$ $\left(\mathcal{R}_{L L M},\mathcal{C}_{L L M}\right)$ , where $\mathcal{R}_{L L M}\in\mathsf{\{P o s i t i v e,N e g a t i v e,N o n e\}}$ denotes a quantitative assessment of the LLM’s output correctness. Specifically, Positive indicates that $M_{i}$ produced a correct answer, Negative indicates an incorrect answer, and None signifies that $M_{i}$ was not utilized in $\mathcal{G}_{k}$ . Additionally, $\mathcal{C}_{L L M}$ provides a qualitative evaluation, offering detailed textual feedback on $M_{i}$ ’s role in the workflow, including how its behavior contributed to or detracted from solving the task. The overall experience pool is thus defined as $\mathcal{P}_{L L M}=\bigcup_{i=1}^{|{\cal M}|}\{(M_{i},\{(\mathcal{G}_{k},\mathcal{R}_{L L M},\mathcal{C}_{L L M})~|~\forall\mathcal{G}_{k}\})\}$ , aggregating performance data across all workflows and tasks. By capturing  both the correctness and the nuanced role of each LLM in addressing diverse tasks, ${\mathcal{P}}_{L L M}$ provides a comprehensive resource for understanding the strengths, weaknesses, and contextual suitability of different LLMs. This facilitates informed decision-making for LLM selection and adaptive workflow optimization.  

## F.2. Workflow Experience Pool  

The workflow experience pool, denoted as ${\mathcal{P}}_{W F}$ , systematically captures the historical performance of workflows by maintaining a collection of records in the form of triplets $(\mathcal{G}_{k},q,\mathcal{E}_{W F})$ . Here, $\mathcal{G}_{k}$ represents a specific workflow, $\mathcal{Q}_{j}$ denotes a query or task associated with the workflow, and $\mathcal{E}_{W F}$ is the corresponding evaluation of the workflow’s performance on the given query. The evaluation $\mathcal{E}_{W F}=(\mathcal{R}_{W F},\mathcal{C}_{W F})$ consists of two components: a quantitative assessment ${\mathcal{R}}_{W F}\in$ $\{{\mathrm{Positive}},{\mathrm{Negative}}\}$ , which indicates whether the workflow successfully solved the query (Positive) or failed (Negative), and a qualitative assessment $\mathcal{C}_{W F}$ , which provides detailed textual feedback on the workflow’s effectiveness, efficiency, and potential limitations in addressing the query. Formally, the experience pool is defined as $\mathcal{P}_{W F}=\{(\mathcal{G}_{k},\mathcal{Q}_{j},\mathcal{E}_{W F})~|$ $\forall\mathcal{G}_{k},\mathcal{Q}_{j}\}$ , aggregating evaluations across diverse workflows and queries. By systematically storing and analyzing these triplets, ${\mathcal{P}}_{W F}$ offers a comprehensive resource for understanding the capabilities and limitations of various workflows, supporting iterative design refinements and enabling the development of more effective and adaptable agentic systems.  

# G. Experimental Details  

## G.1. Dataset Statistics and Splits  

Following existing practices in workflow automation (Saad-Falcon et al., 2024; Hu et al., 2024b; Zhang et al., 2024c), we partition each dataset with a TRAIN:TEST ratio of 1:4, except from ALFWorld dataset which follows the settings in (Shang et al., 2024). For the MATH benchmark, it is worth noting that we follow (Hong et al., 2024), selecting 617 problems from four representative problem types (Combinatorics & Probability, Number Theory, Pre-algebra, Pre-calculus) at difficulty level 5. The dataset statistics are concluded in Table 5.  

## G.2. Baseline Setups  

We detail the settings for all baselines in this section:  

1. CoT. CoT encourages LLM agents to reason step by step rather than directly producing an answer. We adopt the implementation from (Zhang et al., 2022). 2. ComplexCoT. The implementation is based on the code from https://github.com/FranxYao/  

Table 5. Dataset Statistics.   


<html><body><table><tr><td>Domain</td><td>Dataset</td><td>#Train</td><td>#Test</td><td>Metric</td></tr><tr><td rowspan="2">CodeGeneration</td><td>HumanEval</td><td>33</td><td>131</td><td>pass@ 1</td></tr><tr><td>MBPP</td><td>86</td><td>341</td><td>pass@ 1</td></tr><tr><td rowspan="3">MathReasoning</td><td>GSM8K</td><td>264</td><td>1055</td><td>Accuracy</td></tr><tr><td>MATH</td><td>119</td><td>486</td><td>Accuracy</td></tr><tr><td>MultiArith</td><td>150</td><td>600</td><td>Accuracy</td></tr><tr><td>Embodied</td><td>ALFWorld</td><td>230</td><td>327</td><td>Successratio</td></tr></table></body></html>  

Complexity-Based-Prompting/tree/main.  

3. Self-consistency. We ensemble five CoT-generated solutions and adopt the implementation from https: //github.com/geekan/MetaGPT/blob/4954729e7564c806d7e58b3ed8b00ef991f889cc/ metagpt/ext/aflow/scripts/operator.py#L93.  

4. LLM-Debate. We utilize five instances of the same LLM, assigning them distinct roles. These agents engage in up to two debate rounds, with the final answer determined via majority voting. Implementation follows https: //github.com/ucl-dark/llm_debate.  

5. LLM-Blender. The LLM-Blender is powered by two gpt-4o-mini, one Qwen-2.5-72b, and one llama-3.1-70b.  

6. DyLAN. We directly adopt the implementation from (Liu et al., 2023).  

7. AgentVerse. The implementation is adopted from (Chen et al., 2023d).  

8. MacNet. For MacNet (Qian et al., 2024), we select the ”MacNet-MESH” variant, which is essentially a densely connected complete graph.  

9. GPTSwarm. We follow the original implementation and settings described in (Zhuge et al., 2024).  

10. AutoAgents. The setup adheres to the original settings from (Chen et al., 2023b).  

11. ADAS. Implementation details are directly adopted from (Hu et al., 2024b).  

12. AgentSquare. We employ the modular search framework from (Shang et al., 2024). The base LLM is consistently set to gpt-4o-mini, with early stopping patience fixed at 5.  

13. AFlow. In (Zhang et al., 2024c), AFlow utilizes both gpt-4o-mini and the advanced claude-3.5-sonnet. To ensure fairness in homogeneous settings, we limit AFlow to gpt-4o-mini and set MAX ITERATION $\scriptstyle=20$ .  

# H. Supplementary Results  

Table 6. Performance comparison of different methods using various LLM backbones and training datasets. ‘’MATH” and “MBPP” represent individual training datasets, while “MATH $+$ MBPP” indicates training using both datasets combined. The two values under “MATH+MBPP” represent the performance on MATH and MBPP, respectively.   


<html><body><table><tr><td>Method</td><td>LLMBackbone</td><td>MATH</td><td>MBPP</td><td>MATH+MBPP</td></tr><tr><td>DyLAN</td><td>Deepseek-V2.5 QWen-2.5-72b</td><td>46.20 64.17</td><td>80.13 75.63</td><td>43.85/78.62 60.84/71.34</td></tr><tr><td>GPTSwarm</td><td>Deepseek-V2.5 QWen-2.5-72b</td><td>45.36 65.22</td><td>77.52 72.48</td><td>39.18/74.09 64.15/70.90</td></tr><tr><td>AFlow</td><td>Deepseek-V2.5 QWen-2.5-72b</td><td>48.65 66.38</td><td>79.14 80.84</td><td>43.22/77.02 64.71/74.90</td></tr><tr><td>EvoFlow</td><td>LLM I Pool</td><td>72.90</td><td>87.62</td><td>72.69/88.35</td></tr></table></body></html>  