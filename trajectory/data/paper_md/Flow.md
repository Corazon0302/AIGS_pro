# Flow: Modularized Agentic Workflow Automation  

Boye Niu1, Yiliao Song2, Kai Lian1, Yifan Shen3, Yu Yao1, Kun Zhang3,4 Tongliang Liu1† 1University of Sydney; 2University of Adelaide; 3Mohamed bin Zayed University of Artificial Intelligence; 4Carnegie Mellon University  

# Abstract  

Multi-agent frameworks powered by large language models (LLMs) have demonstrated great success in automated planning and task execution. However, the effective adjustment of agentic workflows during execution has not been well studied. An effective workflow adjustment is crucial in real-world scenarios, as the initial plan must adjust to unforeseen challenges and changing conditions in real time to ensure the efficient execution of complex tasks. In this paper, we define workflows as an activity-on-vertex (AOV) graph, which allows continuous workflow refinement by LLM agents through dynamic subtask allocation adjustment based on historical performance and previous AOVs. To further enhance framework performance, we emphasize modularity in workflow design based on evaluating parallelism and dependency complexity. With this design, our proposed multi-agent framework achieves efficient concurrent execution of subtasks, effective goal achievement, and enhanced error tolerance. Empirical results across various practical tasks demonstrate significant improvements in the efficiency of multi-agent frameworks through dynamic workflow refinement and modularization. The code is available at: https://github.com/tmllab/2025_ICLR_FLOW.  

## 1 Introduction  

Large Language Models (LLMs) [19, 31] show remarkable ability to understand and generate human-like text. Recent advances have significantly enhanced their capability to emulate human reasoning [21], indicating a promising future for LLM-based reasoning. With the powerful ability to handle a variety of natural language processing tasks, these models underpin a wide range of applications, from conversational agents [28] and content creation tools [27] to advanced analytics and decision-making systems [17, 23]. Building upon this foundation, a key advancement is the development of multi-agent frameworks empowered by LLM [11, 10, 8, 26, 24, 5, 12] where multiple LLM-based agents collaborate to address complex tasks, leveraging their collective reasoning and planning abilities to automate and optimize task execution processes.  

Existing LLM-based multi-agent frameworks define LLM as an agent, and agents collaborate with each other via manually designed or LLM-generated prompts. Specifically, MetaGPT [8] focuses on programming tasks by leveraging Standardized Operating Procedures (SOPs) [25, 6, 3]. It predefined distinct roles such as product manager, project manager, and engineer. For each role, an LLM agent is initialized, and these agents operate within a strict and sequential workflow to execute subtasks. CAMEL [10] can complete a variety of task types. It requires users to pre-define two agents. These agents interact and execute tasks sequentially, each agent taking on specific responsibilities. AutoGen [26] is also aimed at completing diverse tasks. Unlike CAMEL, AutoGen can automatically create an agent list with different roles based on subtask requirements. These agents execute subtasks sequentially following the order in the list.  

Building upon the strengths of current multi-agent frameworks, our work aims to further improve existing general-purpose multi-agent frameworks by enabling dynamically updating workflows during task execution and encouraging modularity in workflows when planning the workflows.  

Specifically, dynamic updating workflows allow agents to adjust subtask allocations and agent roles in real-time based on ongoing performance feedback and changing conditions. This capability ensures that the system remains responsive and efficient even when faced with unexpected obstacles. For instance, if an agent encounters a roadblock in data preprocessing, the system can reassign this subtask to another agent or introduce a new subtask to resolve the issue. Such adaptability is essential for maintaining robustness and ensuring the seamless execution of complex tasks.  

Modularity In system design, involves dividing a system into separate, independently operating modules, each responsible for specific functionalities [2]. In our context, modularity refers to the decomposition of a complex task into smaller, interchangeable subtask modules. A highly modularized workflow enables subtasks to execute concurrently, without bottlenecks from other parts of the workflow and thereby directly improves the operational efficiency of multi-agent frameworks. Furthermore, modularity enhances the ease of dynamic updating. When workflows are highly modularized, the dependency complexity between subtasks is minimal. Therefore, updating one subtask does not affect others, allowing for small workflow adjustments. For example, if an agent responsible for data preprocessing encounters an unexpected obstacle, a system of high modularity can adapt by introducing only one subtask with minimal impact on the rest of the workflow.  

In this paper, we enhance existing multi-agent frameworks by achieving modularity and enabling dynamic workflow updates. Our framework allows agents to execute their subtasks in parallel while facilitating efficient workflow updates. This is accomplished by formulating the entire workflow as an Activity-on-Vertex (AOV) graph, which is a directed acyclic graph (DAG) where each subtask is represented as a node with its status and generated logs, while the directed edges capture dependencies between subtasks. To encourage a modular workflow design from the beginning, we generate multiple candidate AOV graphs for the task. These candidates are then evaluated based on their degree of parallelism and the complexity of their dependencies. The AOV graph with the highest parallelism and lowest dependency complexity is selected.  

![](https://cdn-mineru.openxlab.org.cn/extract/88a42faf-4fb8-4760-8acb-ab4ad5e88193/4dbb0a984faf54bd6d305eff2f58dff645a2da0dc2ed993d57e43ed560164c8c.jpg)  
Figure 1: Comparative evaluations among four frameworks—AutoGen, CAMEL, MetaGPT, and Flow (ours)—across two tasks, present notable differences in performance. For the left task, AutoGen, CAMEL, and MetaGPT only managed to produce basic designs lacking in completeness while Flow excelled by creating a fully developed and well-structured website. For the right task, Flow demonstrated superior capability by successfully generating a working game with a clear and intuitive interface, while the other frameworks struggled to deliver fully functional code.  

During task execution, our framework continuously checks and refines the workflow, updating it when a subtask fails (see Fig. 2: Check & Refine). The framework updates subtask allocations and agent roles based on ongoing performance data and current workflow. As our AOV-based workflow encourages high modularity, updating one module does not necessarily affect others, allowing for localized adjustments during workflow updates (see Fig. 2: Update). Similar to the initial workflow generation, multiple AOV graphs are generated and the one with highest parallelism and lowest dependency complexity is selected during dynamic updates. This iterative workflow refinement process enhances adaptability to new challenges and evolving objectives throughout task execution, ensuring dynamic workflow updates without compromising overall performance.  

Our key contributions are as follows: 1) We introduce and encourage modularity in multiagent workflows, emphasizing the design of workflows with high parallelism and low dependency complexity. This modular design enhances efficiency, robustness, and scalability by enabling concurrent subtask execution and minimizing bottlenecks caused by complex interdependence. 2) We propose a practical multi-agent framework that supports highly flexible updates to the workflow during runtime. Our method enables local updates to the entire workflow based on global information, allowing agents to efficiently adapt to unexpected challenges while maintaining system coherence and consistency. 3)Through comprehensive experiments, we demonstrate significant improvements in both the adaptability and efficiency of our multi-agent framework compared to existing approaches.  

## 2 Related Work  

LLM-based Task Decision-Making Recent developments in LLM-based task decision making have focused on improving the reasoning and planning abilities of agents [27, 20, 30, 26, 15, 18, 1]. Previous approaches like ReAct [27] iteratively generate thoughts and actions based on current observations until task completion. This framework integrates action-taking with reasoning, allowing agents to perform complex tasks in dynamic environments. Reflexion [18] further improves this by incorporating self-reflection, where the agent evaluates and adjusts its reasoning during execution. ADAPT [15] introduces recursive task decomposition, enabling LLM-based agents to break tasks into smaller subtasks, which leads to improved task execution flexibility. However, these approaches often overlook dynamic task reallocation, particularly in multi-agent settings, which is where our work extends the current research.  

LLM-based Multi-Agent Frameworks Multi-agent frameworks have long been employed for task execution in distributed environments, with recent advances leveraging LLM to enhance coordination and decision-making [8, 10, 26, 9]. However, existing frameworks often rely on static workflows with limited adaptability to changes in the task environment. DyLAN [12] and MACNET [16] utilize static graphs to represent workflows in multi-agent frameworks; GPTSwarm [32] enhances agent interactions but maintains a fixed agent topology; DataInterpreter [7] updates workflows primarily in response to execution failures in subtasks, adjusting subsequent tasks while leaving completed tasks unchanged; AFlow [29] introduces a dynamic workflow generation framework based on Monte Carlo Tree Search, enabling adaptive adjustments through iterative code modification. This highlights the need for dynamic workflow updates.  

## 3 Method  

Our proposed Flow enhances multi-agent frameworks powered by LLM by introducing modularity and dynamic workflow updating. As depicted in Fig. 2, given the task requirement, Flow first formulates the initial workflow for execution plan generation and agent allocation. During execution, the workflow is continuously refined and dynamically updated until the task is completed. To maximize system simplicity and flexibility, we design a dictionary-based structure for implementation. In the following, we detail how to achieve these features.  

Formulating a Workflow as an AOV Graph Activity on Vertex (AOV) graph is a type of directed acyclic graph where vertices represent subtasks and edges denote precedence relations [4]. AOV graphs are widely used in project scheduling and management [13, 22], helping planners visualize dependencies and sequence subtasks efficiently.  

Inspired by that, we define the multi-agent workflow as an AOV graph where vertices represent subtasks, while edges denote dependencies between subtasks. Let $G=(V,E,A)$ denote the AOV graph, with $V$ the set of all subtasks (vertices), $E\subseteq V\times V$ the set of directed edges indicating subtask dependencies. For example, $e_{i j}=(v_{i},v_{j})\in E$ indicates that the subtask $v_{i}$ must be completed before the subtask $v_{j}$ starts. $A$ represents a set of agents for all subtasks. Each agent $a_{j}\in A$ is associated with a role that is responsible for executing a subset of subtasks ${\mathcal{T}}_{j}\subseteq V$ .  

Note that AutoGen [26] also automatically generates subtasks and agents. However, the subtasks are designed to be executed sequentially. For Flow, we allow for the generation of complementary subtasks that can run in parallel. This distinction enhances our framework’s ability to handle multiple subtasks simultaneously, which reduces overall process time and increases efficiency.  

Modularity in a Workflow Modularity in system design [2] involves dividing a system into separate, independently operating modules, each responsible for specific functionalities, allowing focus on individual components without affecting the entire system. It is essential for scalability and flexibility in workflows. By reducing dependency complexity, the system can more easily adapt to changes, such as the introduction of new tasks or the reassignment of existing ones, without requiring extensive restructuring. Theorem 3.1 demonstrates additional dependencies in a workflow reduce the expected success rate of subtasks. Following this conclusion, Flow advocates for the creation of subtasks that can be executed independently.  

Theorem 3.1. Consider two topologically sorted workflows $A$ and $B$ each consisting of $N$ subtasks according to their execution order. Suppose  

1. (Random fail probability) Each subtask $v\in\mathcal T$ fails with probability $p_{f}$ , where $0<p_{f}<1$ . 2. (Additional dependency in Workflow $\pmb{B}$ ) There exist at least one subtask $v^{\ast}\in\mathcal{T}$ and $a$ subtask $b\in{\mathcal{T}}$ such that the set of immediate predecessors (dependencies) of $v^{*}$ in Workflow $B$ is $D_{B}(v^{*})=D_{A}(v^{*})\cup\{b\}$ , where $D_{A}(v^{*})$ is the set of immediate predecessors of $v^{*}$ in Workflow A. For all other subtasks $v\neq v^{*}D_{A}(v)\subseteq D_{B}(v)$ .  

The expected number of completed subtasks in Workflow $A$ is strictly greater than in Workflow $B$ : $E[S_{A}]>E[S_{B}]$ .  

To encourage modularity in the generated AOV graph, we define two quantitative measures that evaluate parallelism and dependency complexity respectively. Parallelism measures the extent to which subtasks can be executed concurrently. Let $S_{t}$ represent the set of subtasks executed in the $t$ step. Let $T$ be the total number of steps (the maximum depth of the DAG). Given an AOV graph $G=(V,E,A)$ , the degree of parallelism overall is defined as the average subtask ratio over steps:  

$$
P_{\mathrm{avg}}=\frac{1}{T}\sum_{t=1}^{T}S_{t}.
$$  

Although $P_{\mathrm{avg}}$ provides a measure of parallelism, it is insufficient to fully capture the modularity that arises when subtasks can be executed independently. Consider two workflows, both containing the same subtasks $\{A,B,C,D\}$ . For Workflow 1, the task dependencies are defined as: $A\rightarrow C,B\rightarrow C,A\rightarrow D,B\rightarrow D,C\rightarrow D$ . In contrast, Workflow 2 has dependencies: $A\rightarrow C,B\rightarrow C,C\rightarrow D$ . Although both workflows exhibit the same level of parallelism, Workflow 2 is structurally simpler in terms of task dependencies, as it contains fewer edges.  

To account for this complexity, we measure the dependency structure by analyzing the degree distribution within the subtask graph. For each subtask $v_{i}$ , we define $\deg(v_{i})$ as the number of direct connections it has on the graph $G$ . The dependency complexity is quantified by the standard deviation of the number of direct connections:  

$$
C_{\mathrm{dependency}}=\sigma_{\mathrm{deg}(v_{i})}=\sqrt{\frac{1}{|V|}\sum_{v_{i}\in V}(\deg(v_{i})-\bar{d})^{2}}.
$$  

This measure reflects the variability in the number of dependencies each subtask has, providing insight into the overall complexity of the workflow structure.  

Task dependencies alone are insufficient to fully capture the modularity that allows subtasks to be executed independently. Consider Workflow 3: $A\rightarrow B\rightarrow C\rightarrow D$ , which may have a similar dependency complexity to Workflow 2. However, Workflow 2 provides greater modularity and separation of subtasks, highlighting the importance of evaluating both dependency complexity and modularity to fully assess and promote effective workflow designs. Both measures are essential to ensure that subtasks can be executed in parallel while maintaining a modular approach.  

# A Sample Prompt for Initialization Pinit  

You are an intelligent workflow planner. Given the following task requirements, generate a set of necessary sub-tasks along with their dependencies and assign appropriate agents to each task. Ensure that tasks that can be executed in parallel are identified to enhance efficiency. The workflow should be represented as a dictionary where each key is a task and its value contains the task’s status, data, number of parents not completed, child tasks, and assigned agent.  

Task Requirements: {TASK_REQUIREMENTS}  

Output Format: { "Task_A": { "status": "not started", "data": null, " num_parents_not_completed": 0, "child": ["Task_B", "Task_C"], "agent": "Agent_1 " }, "Task_B": { "status": "not started", "data": null, " num_parents_not_completed": 1, "child": ["Task_D"], "agent": "Agent_2" }, ... }  

Generate an Initial AOV Graph Given a task requirement prompt $\mathcal{P}$ , we prompt an LLM $f$ to generate a set of candidate AOV graphs $\{G_{1},G_{2},\dots,G_{K}\}$ based on $\mathcal{P}$ and a designed prompt for initialization $\mathcal{P}_{\mathrm{init}}$ , i.e. $\{G_{1},G_{2},...,G_{K}\}=f(\mathcal{P}_{\mathrm{init}},\mathcal{P})$ . Each candidate AOV graph $G_{k}=$ $(V_{k},E_{k},A_{k})$ is evaluated using the measures of parallelism and dependency complexity. We prioritize the workflow with the highest parallelism score. If multiple graphs share the highest score, we select the one with the lowest dependency complexity.  

Note that we prioritize parallelism and modularity early in the process and focus on refining the workflow through data-driven adjustments during running. The reasons are: 1) LLMgenerated workflows possess reasoning capabilities, but may not prioritize efficiency. If parallelism and independence are not explicitly encouraged during the initial workflow generation, the applied workflow is very likely to be overly complex, which results in inefficient subtask implementation; 2) verifying correctness is inherently challenging as no additional data is available as supervised information at an early stage. As compensation, we refine the workflow by parallelism and modularity.  

![](https://cdn-mineru.openxlab.org.cn/extract/88a42faf-4fb8-4760-8acb-ab4ad5e88193/cd3bd0be544b4bdffd59872bf246864dd8bfe0f6ae911515ef5cb47587ae79a5.jpg)  
Figure 2: The process starts with task initialization, encouraging the modularity and execute parallel of subtasks. Outputs are evaluated. If errors are detected, the workflow is dynamically updated by modifying the task graph. This iterative process continues until successful task completion.  

Execution Plan Generation and Agent Allocation After we obtain the best AOV graph, a topological sort is performed on the dependency graph of the subtasks to produce a linear order of the subtasks $o:V\rightarrow\{1,2,\dots,|V|\}$ such that for any edge $(v_{i},v_{j})\in E,o(v_{i})<o(v_{j})$ . The result is a sequence of subtask steps, where each step consists of subtasks that can be executed in parallel. This execution plan minimizes the number of steps needed to perform while ensuring that all subtasks are completed in the shortest possible time, adhering to their dependencies.  

Each agent $a_{j}\in A$ is associated with a set of subtasks $\mathcal{T}_{j}\subseteq V$ , indicating the subtasks that the agent is responsible for handling. However, if two subtasks $v_{p}$ and $v_{q}$ require the same agent $a_{j}$ at the same step $s_{i}$ , we create a clone of the agent, denoted $a_{j}^{\prime}$ , to run both subtasks simultaneously  

without increasing the wait time.  

# Prompt for Update Pupdate  

You are an intelligent workflow updater. Based on the current workflow and the all subtasks’ progress data, update the workflow for acheving the objective by adding, removing, or modifying subtasks as necessary. Ensure that the updated workflow maintains modularity and maximizes parallel execution.  

Output Format: { "Task_A": { "status": "not started", "data": null, ...  

Workflow Refinement and Dynamic Updating We leverage LLM as a global inspector to continuously monitor task progress and dynamically modify the AOV graph based on global information when necessary. Specifically, given the task requirements prompt $\mathcal{P}$ and the update prompt $\mathcal{P}_{\mathrm{update}}$ , the current AOV graph $G^{t}$ , and the generated data $D^{t}$ containing the status of erate $K$ candidate graphs: $\{G_{1}^{t+1},G_{2}^{t+1},\ldots,G_{K}^{t+1}\}=\{\bar{\phi}_{\mathrm{{update}}},\mathcal{P},D^{t}\}$ . We follow the same selection strategy as in initialization, which prioritizes the workflow with the highest parallelism score and further selects the one with the lowest dependency complexity if multiple graphs share the highest parallelism score.  

With the modularity constraint introduced in previous sessions, our dynamic updates can largely fulfill flexibility, allowing modifications to subtask allocations including deletion, addition, editing, rerunning, and reassignment of agents without necessarily affecting other agents or their assigned subtasks. This unique advantage is particularly beneficial when subtask requirements become more challenging, as subtask dependencies can be highly complex.  

Note that with sufficient data and computational resources, we could further enhance our framework by fine-tuning LLM with reinforcement learning for workflow generation. For example, the LLM would be trained to maximize a reward function designed around key performance indicators such as task completion speed, resource utilization, and minimization of workflow disruptions.  

Implementation Our framework employs a dictionary-based structure, $\tilde{G}$ , to efficiently manage and dynamically update workflows within a multi-agent framework. Each subtask $v$ in the workflow is represented as a key in $\tilde{G}$ , the value being another dictionary that encapsulates various attributes of the subtask. The structure is specifically defined as:  

$\begin{array}{r}{\tilde{G}[v]=\left\{\begin{array}{l l}\end{array}\right.}\end{array}$ "subtask requirement", "status", "data", "num_parents_not_completed", "child", "agent" $\}$ .  

In each $\tilde{G}[v]$ , the values of each key are as follows:  

• "subtask requirement": the text of the task requirement;   
• "status": the current task implementation status e.g. "not started", "in progress", "completed";   
• "data": data relevant to this task;   
• "num_parents_not_completed": the count of uncompleted parent tasks to manage dependencies;   
• "child": a list of child tasks that depend on the current task’s completion;   
• "agent": the agent assigned to the task.  

This dictionary-based structure can be converted directly to JSON, and the organized information is easily readable and summarizable by LLM, granting our system inherent simplicity and flexibility. Each subtask execution readiness is determined by the attribute "num_parents_not_completed" Subtasks with a count of zero are eligible to run concurrently, leveraging our system’s capability to handle parallel subtask execution effectively. Upon completion of each subtask, we perform a systematic review to determine if the workflow requires refinement, ensuring that all dependencies are accurately accounted for and that the workflow remains aligned with project goals. In addition to monitoring the subtask completion by the "status" and "num_parents_not_completed" counts reported by agents. Flow also double-checks the completion of each subtask by asking if all the requirements of this subtask are fulfilled. This will largely prevent errors from inaccurate reporting by agents or unforeseen system anomalies. This rigorous verification process enhances the reliability and integrity of our workflow management system.  

## 4 EXPERIMENTS  

Baselines In all experiments, we compare Flowto the existing multi-agent frameworks i.e. (1) AutoGen [26], (2) Camel [10], and (3) MetaGPT [8]. In our experiments, we use agents empowered by GPT-4o-mini and GPT-3.5-Turbo [14].  

Experiment Design We designed three diverse and engaging tasks to evaluate multi-agent collaboration frameworks: 1) website design, 2) LaTeX Beamer writing, and 3) gobang game development. The rationale for selecting coding-based experiments is two-fold. First, most multi-agent frameworks, such as MetaGPT [8], are optimized for coding and writing tasks. Using non-coding tasks could introduce bias. Second, coding tasks effectively showcase the ability of a framework to assign agents and manage task allocation.  

Gobang Game Development: This task requires creating a gobang game with a user interface and a simple AI opponent. Players can choose between black or white stones, with the UI clearly indicating turns and announcing the winner or draw when the game ends. This task demonstrates the framework’s ability to handle modular design and task parallelism, as it involves coordinating game logic, AI implementation, and user interface development simultaneously.  

LaTeX Beamer Writing: This task focuses on generating LaTeX slides that cover reinforcement learning algorithms, including motivations, problem statements, intuitive solutions, and detailed mathematical equations. A specific page requirement is to test the framework’s ability to follow instructions precisely. The task highlights the framework’s parallel processing capabilities of simultaneous generation of content, formatting, and presentation structure. The structured format of LaTeX also tests how effectively the framework manages modularity and concurrent tasks.  

Website Design: This task involves building a professional website for the International Conference on Learning Representations, hypothetically scheduled for San Francisco from April 27 to May 1, 2025. The website must feature key elements such as a detailed conference schedule and venue information with an interactive map. This task assesses each framework’s ability to manage parallel workflows and modular components, including user interface design, functionality, and adherence to design guidelines, showcasing how well the framework handles task decomposition and execution.  

### 4.1 Evaluations over Three Designed Tasks  

Evaluation Metrics To conduct both quantitative and qualitative evaluations, we employ two metrics: Success Rate and Human Rating. The success rate is a quantitative measure that ranges from 0 to 1. It assesses whether the multi-agent framework successfully generates executable outputs that fully meet the task requirements. A higher score indicates a greater level of success in accurately fulfilling the task objectives. Different tasks may have different evaluation metrics. The description for each evaluation metric is defined in Appendix B.1, B.2 and B.3. Human ratings are used to evaluate the quality of the generated results in alignment with the task description. We gathered 50 participants with programming and machine learning backgrounds to rank the outcomes produced by different methods. A detailed description of how we take scores is shown in Appendix A.  

Summary We summarize the performance of different methods on three tasks from Table 1, 2 and 3, comparing the overall score with respect to the success rate and human rating. For Flow, the overall score and human rating over three tasks are (100, 4) on game development, (100, 3.33) on LaTeX writing, and (80, 3.28) on website design. Thus, the average performance of Flow is a $93\%$ success rate and 3.54 out of 4 in human satisfaction. Similarly, we have the average performance of AutoGen as (66.7, 2.63), MetaGPT as (71, 1.60), and CAMEL as (48.67, 2.12). Overall, our method Flow has completed tasks with the most satisfaction and the highest success rate. Information about Flow’s workflow on those tasks is in Appendix D.  

### 4.2 Result for Gobang Game Development  

The experimental setup is thoroughly detailed in Appendix B.2 and the visualization result is shown in Fig.1. As shown in Table 1, Flow achieves a $100\%$ success rate across all aspects, as well as the highest human satisfaction. More explanations for each method are given below.  

AutoGen: Of the five trials, one of the tests failed to generate a valid result. Of the four successful attempts, one contained a code error that hindered normal execution, while another exhibited a bug in the game interface. The remaining two trials were completed successfully, although the chess pieces were displayed as the text ’black’ and ’white’ instead of graphical representations.  

MetaGPT : After five trials, all MetaGPT attempts were successful and intractable. However, in four trials, a Tic-Tac-Toe game was generated instead of Gobang; out of these, the left one was functional, allowing both the user and AI to make moves and correctly terminate.  

CAMEL: In all five trials, CAMEL was only successful twice. In the other trials, the generated Python code was not executable. In the two successful trials, CAMEL successfully implemented the correct termination conditions but did not have an AI component and no termination message.  

Flow: After running Flow five times, our framework consistently generated successful outputs without errors. The game functioned as expected, allowing both the player and the naive AI to take turns seamlessly. The game also ended correctly when either the board was fully occupied or one side achieved victory. In the game interface, actual black and white chess pieces were displayed rather than text labels, enhancing the user experience.  

### 4.3 Result for LaTeX Beamer Writing  

Experimental results are presented in Table 2 with the following explanations:  

Table 1: Comparison of different multi-agent frameworks on Gobang Game Development   


<html><body><table><tr><td rowspan="2">Model</td><td colspan="4">Success Rate (%)</td><td rowspan="2">Human Rating (1-4)</td></tr><tr><td>Compilable</td><td>Intractable</td><td>GameRule</td><td>OverallScore</td></tr><tr><td>AutoGen[26]</td><td>80</td><td>60</td><td>40</td><td>60</td><td>2.26</td></tr><tr><td>MetaGPT [8]</td><td>100</td><td>100</td><td>20</td><td>73</td><td>1.24</td></tr><tr><td>CAMEL [10]</td><td>40</td><td>40</td><td>0</td><td>27</td><td>2.50</td></tr><tr><td>Flow (Ours)</td><td>100</td><td>100</td><td>100</td><td>100</td><td>4.00</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td></tr></table></body></html>  

Table 2: Comparison of different multi-agent frameworks on LaTeX Beamer Writing   


<html><body><table><tr><td rowspan="2">Model</td><td colspan="4">Success Rate (%)</td><td rowspan="2">Human Rating (1-4)</td></tr><tr><td>Compilable</td><td>Completeness</td><td>Page Limit</td><td>OverallScore</td></tr><tr><td>AutoGen [26]</td><td>80</td><td>80</td><td>40</td><td>67</td><td>3.00</td></tr><tr><td>MetaGPT[8]</td><td>80</td><td>80</td><td>20</td><td>60</td><td>1.83</td></tr><tr><td>CAMEL [10]</td><td>100</td><td>100</td><td>0</td><td>66</td><td>1.83</td></tr><tr><td>Flow (Ours)</td><td>100</td><td>100</td><td>100</td><td>100</td><td>3.33</td></tr></table></body></html>  

<html><body><table><tr><td rowspan="2">Model</td><td colspan="4">Success Rate (%)</td><td rowspan="2">Human Rating (1-4)</td></tr><tr><td>Compilable</td><td>BasicInformation</td><td>Sections</td><td>OverallScore</td></tr><tr><td>AutoGen [26]</td><td>80</td><td>80</td><td>60</td><td>73</td><td>2.62</td></tr><tr><td>MetaGPT[8]</td><td>100</td><td>100</td><td>40</td><td>80</td><td>1.72</td></tr><tr><td>CAMEL [10]</td><td>80</td><td>80</td><td>0</td><td>53</td><td>2.02</td></tr><tr><td>Flow (Ours)</td><td>80</td><td>80</td><td>80</td><td>80</td><td>3.28</td></tr></table></body></html>

Table 3: Comparison of different multi-agent frameworks on Website Design  

AutoGen: After five trials, AutoGen successfully generated the output each time. However, one output failed to compile in LaTeX due to syntax errors, and in two instances, the outputs did not meet the required length. The remaining outputs met both the length and content requirements.  

MetaGPT : In five trials, four of them successfully generated a valid LaTeX version, with the only error being related to writing Python code within the ’.tex’ file. In these four successful trials, all documents met the required content specifications, but only one meet the requirement of either 30 or 20 pages.  

CAMEL: CAMEL successfully generated five valid ’.tex’ files, all of which could be rendered into Beamer format. Each presentation contained the required information, including sections such as motivation. However, none met the required page count of either 30 or 20 pages.  

Flow: After five tests, our framework successfully generated output each time, and all outputs were able to be compiled in LaTeX. However, one output contained some repetitive content. In the remaining valid outputs, the Beamer presentations met the specified length requirements and adequately covered all required content.  

### 4.4 Result For Website Design  

Similarly to the previous two, the detailed experiment setup is in Appendix B.3. Here we illustrate the results in Table 3 as follows:  

AutoGen: In five trials, four of the AutoGen results successfully rendered into an HTML website. However, in one attempt, each section of the website contained only one or two sentences and lacked interactive features and essential elements like maps or tables.  

![](https://cdn-mineru.openxlab.org.cn/extract/88a42faf-4fb8-4760-8acb-ab4ad5e88193/1a9e40e1faa22937fe1e6a90ce86ac1b1e80a198cc8f972ba629ad860198bb7e.jpg)  
Figure 3: Workflow and dynamic update in two experiments.  

MetaGPT : MetaGPT managed to create complete HTML and CSS, meeting basic functionality requirements and showcasing its code generation capabilities. However, the outputs were overly simplistic, missing content and key functional sections like the required venue and map.  

CAMEL: CAMEL’s outputs were executable in four out of five runs, though they did not include all the necessary elements, achieving all basic functions only. CAMEL restricts communication to only two agents, regardless of task complexity, hindering its ability to fully complete complex website development tasks. One of the results generated complete HTML code but omitted the CSS file.  

Flow: Flow achieved an $80\%$ success rate across five trials. One trial failed to generate an HTML website. Among the four remaining trials, each section of the website featured detailed introductions and necessary interactive functionalities. For example, the venue section included travel information and local transportation options. The registration section was fully functional, with a complete table, input boxes, and a submit button.  

## 5 Workflow Update  

Update based On Generated Data Fig. 3(a) demonstrates the update process of Flow in the conference website creation example. Upon completion of the first subtask, the system identifies potential changes and redundancies, triggering a restructuring process to improve efficiency. Once the subtask "Define the website structure" is completed, the generated data, which includes HTML structures and elements, is sufficient to proceed with the CSS creation. As a result, the workflow is updated to incorporate the development of CSS based on the completed "Define the website structure" subtask.  

Fig. 3(b) illustrates a result of our dynamic updating process, where the system, upon receiving information about completed subtasks, decides to add a bridging subtask to handle gaps and ensure that the workflow continues smoothly.  

Error handling To evaluate the effectiveness of our update mechanism, we intentionally introduced random masking to certain subtasks’ output, replacing them with "none" before passing them to the next agent. We conducted five trials and recorded the success scores. Since other frameworks employ a sequential workflow, we limit the comparison to our own approach in this context.  

Table 4: Success Rate $\left(\%\right)$ of Error handling with dynamically updating.   


<html><body><table><tr><td>Task</td><td>Flow w/o Update</td><td>Flow</td></tr><tr><td>WebsiteDesign</td><td>46</td><td>87</td></tr><tr><td>Gobang Game Development</td><td>0</td><td>93</td></tr><tr><td>LaTeX Beamer Writing</td><td>67</td><td>93</td></tr></table></body></html>  

We observed a significant difference in the success rate between using dynamic update and not, particularly in the Interactive Game section as shown in Table 4. The main issue arises when the previous agent fails to provide the necessary information, yet the second agent continues with its subtask, leading to a major disconnect in the code. This often results in Python being unable to compile due to missing or mismatched components. Similarly, in website design, the lack of required elements caused by this failure impacts the overall functionality and structure. During the execution of subtasks, errors may arise due to the limitations of the LLM-based agent or underperformance in certain tasks. Therefore, the ability to dynamically update the agent workflow to address such issues is essential.  

## 6 Conclusion  

We present Flow, a novel LLM-based multi-agent framework that can dynamically adapt to unforeseen challenges for general task executions. By dynamically updating the agentic workflow using AOV graphs, our framework has largely fulfilled the modularity requirements to complete complex tasks. We demonstrate our method through case studies on a series of experiments, ranging from website design, game development, and LaTeX Beamer writing, as well as testing its capability to solve general benchmark tasks. Through objective evaluation metrics and human feedback, we found that Flow improves execution efficiency, offers better error tolerance, and delivers overall stronger performance.  

# References  

[1] Michael Ahn, Anthony Brohan, Noah Brown, Yevgen Chebotar, Omar Cortes, Byron David, Chelsea Finn, Chuyuan Fu, Keerthana Gopalakrishnan, Karol Hausman, Alex Herzog, Daniel Ho, Jasmine Hsu, Julian Ibarz, Brian Ichter, Alex Irpan, Eric Jang, Rosario Jauregui Ruano, Kyle Jeffrey, Sally Jesmonth, Nikhil J Joshi, Ryan Julian, Dmitry Kalashnikov, Yuheng Kuang, Kuang-Huei Lee, Sergey Levine, Yao Lu, Linda Luu, Carolina Parada, Peter Pastor, Jornell  

Quiambao, Kanishka Rao, Jarek Rettinghouse, Diego Reyes, Pierre Sermanet, Nicolas Sievers, Clayton Tan, Alexander Toshev, Vincent Vanhoucke, Fei Xia, Ted Xiao, Peng Xu, Sichun Xu, Mengyuan Yan, and Andy Zeng. Do as i can, not as i say: Grounding language in robotic affordances, 2022. URL https://arxiv.org/abs/2204.01691.   
[2] Carliss Y. Baldwin and Kim B. Clark. Design Rules: The Power of Modularity Volume 1. MIT Press, Cambridge, MA, USA, 1999. ISBN 0262024667.   
[3] R.M. Belbin. Team Roles at Work. Butterworth-Heinemann, 2010. ISBN 9781856178006. URL https://books.google.com.au/books?id $=$ hF2yJzYfUBAC.   
[4] A. Bondy and U.S.R. Murty. Graph Theory. Graduate Texts in Mathematics. Springer London, 2011. ISBN 9781846289699. URL https://books.google.com.au/books?id= HuDFMwZOwcsC.   
[5] Weize Chen, Yusheng Su, Jingwei Zuo, Cheng Yang, Chenfei Yuan, Chi-Min Chan, Heyang Yu, Yaxi Lu, Yi-Hsin Hung, Chen Qian, Yujia Qin, Xin Cong, Ruobing Xie, Zhiyuan Liu, Maosong Sun, and Jie Zhou. Agentverse: Facilitating multi-agent collaboration and exploring emergent behaviors. In The Twelfth International Conference on Learning Representations, 2024. URL https://openreview.net/forum?id $=$ EHg5GDnyq1.   
[6] T. DeMarco and T.R. Lister. Peopleware: Productive Projects and Teams. Addison-Wesley, 2013. ISBN 9780321934116. URL https://books.google.com.au/books?id= DVlsAQAAQBAJ.   
[7] Sirui Hong, Yizhang Lin, Bang Liu, Bangbang Liu, Binhao Wu, Ceyao Zhang, Chenxing Wei, Danyang Li, Jiaqi Chen, Jiayi Zhang, Jinlin Wang, Li Zhang, Lingyao Zhang, Min Yang, Mingchen Zhuge, Taicheng Guo, Tuo Zhou, Wei Tao, Xiangru Tang, Xiangtao Lu, Xiawu Zheng, Xinbing Liang, Yaying Fei, Yuheng Cheng, Zhibin Gou, Zongze Xu, and Chenglin Wu. Data interpreter: An llm agent for data science, 2024. URL https://arxiv.org/ abs/2402.18679.   
[8] Sirui Hong, Mingchen Zhuge, Jonathan Chen, Xiawu Zheng, Yuheng Cheng, Jinlin Wang, Ceyao Zhang, Zili Wang, Steven Ka Shing Yau, Zijuan Lin, Liyang Zhou, Chenyu Ran, Lingfeng Xiao, Chenglin Wu, and Jürgen Schmidhuber. MetaGPT: Meta programming for a multi-agent collaborative framework. In The Twelfth International Conference on Learning Representations, 2024. URL https://openreview.net/forum?id $=$ VtmBAGCN7o.   
[9] Shengran Hu, Cong Lu, and Jeff Clune. Automated design of agentic systems, 2024. URL https://arxiv.org/abs/2408.08435.   
[10] Guohao Li, Hasan Abed Al Kader Hammoud, Hani Itani, Dmitrii Khizbullin, and Bernard Ghanem. Camel: Communicative agents for "mind" exploration of large language model society. In Thirty-seventh Conference on Neural Information Processing Systems, 2023.   
[11] Zhiwei Liu, Weiran Yao, Jianguo Zhang, Le Xue, Shelby Heinecke, Rithesh Murthy, Yihao Feng, Zeyuan Chen, Juan Carlos Niebles, Devansh Arpit, Ran Xu, Phil Mui, Huan Wang, Caiming Xiong, and Silvio Savarese. Bolaa: Benchmarking and orchestrating llm-augmented autonomous agents, 2023. URL https://arxiv.org/abs/2308.05960.   
[12] Zijun Liu, Yanzhe Zhang, Peng Li, Yang Liu, and Diyi Yang. A dynamic llm-powered agent network for task-oriented agent collaboration, 2024. URL https://arxiv.org/abs/ 2310.02170.   
[13] J.J. Moder, C.R. Phillips, and E.W. Davis. Project Management with CPM, PERT, and Precedence Diagramming. Van Nostrand Reinhold, 1983. ISBN 9780442254155. URL https: //books.google.com.au/books?id=WmhRAAAAMAAJ.   
[14] OpenAI. Gpt-4o mini: Advancing cost-efficient intelligence. https://openai.com/ index/gpt-4o-mini-advancing-cost-efficient-intelligence/, 2024. Accessed: 2024-09-29.   
[15] Archiki Prasad, Alexander Koller, Mareike Hartmann, Peter Clark, Ashish Sabharwal, Mohit Bansal, and Tushar Khot. Adapt: As-needed decomposition and planning with language models. arXiv, 2023.   
[16] Chen Qian, Zihao Xie, Yifei Wang, Wei Liu, Yufan Dang, Zhuoyun Du, Weize Chen, Cheng Yang, Zhiyuan Liu, and Maosong Sun. Scaling large-language-model-based multi-agent collaboration, 2024. URL https://arxiv.org/abs/2406.07155.   
[17] Aditya Ramesh, Mikhail Pavlov, Gabriel Goh, Scott Gray, Chelsea Voss, Alec Radford, Mark Chen, and Ilya Sutskever. Zero-shot text-to-image generation, 2021. URL https:// arxiv.org/abs/2102.12092.   
[18] Noah Shinn, Federico Cassano, Ashwin Gopinath, Karthik R Narasimhan, and Shunyu Yao. Reflexion: language agents with verbal reinforcement learning. In Thirty-seventh Conference on Neural Information Processing Systems, 2023. URL https://openreview.net/ forum?id $=$ vAElhFcKW6.   
[19] Significant Gravitas. AutoGPT. https://github.com/ Significant-Gravitas/AutoGPT. MIT License.   
[20] Chan Hee Song, Brian M. Sadler, Jiaman Wu, Wei-Lun Chao, Clayton Washington, and Yu Su. Llm-planner: Few-shot grounded planning for embodied agents with large language models. In 2023 IEEE/CVF International Conference on Computer Vision (ICCV), pages 2986–2997, 2023. doi: 10.1109/ICCV51070.2023.00280.   
[21] Hongda Sun, Weikai Xu, Wei Liu, Jian Luan, Bin Wang, Shuo Shang, Ji-Rong Wen, and Rui Yan. DetermLR: Augmenting LLM-based logical reasoning from indeterminacy to determinacy. In Lun-Wei Ku, Andre Martins, and Vivek Srikumar, editors, Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 9828–9862, Bangkok, Thailand, August 2024. Association for Computational Linguistics. doi: 10.18653/v1/2024.acl-long.531. URL https://aclanthology.org/2024. acl-long.531.   
[22] H.A. Taha. Operations Research an Introduction. Pearson, 2017. ISBN 9780134444017. URL  

https://books.google.com.au/books?id=HbpKjwEACAAJ.  

[23] Guanzhi Wang, Yuqi Xie, Yunfan Jiang, Ajay Mandlekar, Chaowei Xiao, Yuke Zhu, Linxi Fan, and Anima Anandkumar. Voyager: An open-ended embodied agent with large language models, 2023. URL https://arxiv.org/abs/2305.16291.   
[24] Yaoxiang Wang, Zhiyong Wu, Junfeng Yao, and Jinsong Su. Tdag: A multi-agent framework based on dynamic task decomposition and agent generation, 2024. URL https: //arxiv.org/abs/2402.10178.   
[25] Michael Wooldridge and Nicholas R. Jennings. Pitfalls of agent-oriented development. In Proceedings of the Second International Conference on Autonomous Agents, AGENTS ’98, page 385–391, New York, NY, USA, 1998. Association for Computing Machinery. ISBN 0897919831. doi: 10.1145/280765.280867. URL https://doi.org/10.1145/280765.280867.   
[26] Qingyun Wu, Gagan Bansal, Jieyu Zhang, Yiran Wu, Beibin Li, Erkang Zhu, Li Jiang, Xiaoyun Zhang, Shaokun Zhang, Jiale Liu, et al. Autogen: Enabling next-gen llm applications via multi-agent conversation. In ICLR 2024 Workshop on Large Language Model (LLM) Agents, 2024.   
[27] Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik R Narasimhan, and Yuan Cao. React: Synergizing reasoning and acting in language models. In The Eleventh International Conference on Learning Representations, 2023. URL https://openreview.net/ forum?id $\c=$ WE_vluYUL-X.   
[28] Yining Ye, Xin Cong, Shizuo Tian, Yujia Qin, Chong Liu, Yankai Lin, Zhiyuan Liu, and Maosong Sun. Rational decision-making agent with internalized utility judgment, 2024. URL https://openreview.net/forum?id=l1pNNQSzZv.   
[29] Jiayi Zhang, Jinyu Xiang, Zhaoyang Yu, Fengwei Teng, Xiong-Hui Chen, Jiaqi Chen, Mingchen Zhuge, Xin Cheng, Sirui Hong, Jinlin Wang, Bingnan Zheng, Bang Liu, Yuyu Luo, and Chenglin Wu. AFlow: Automating agentic workflow generation. In The Thirteenth International Conference on Learning Representations, 2025. URL https://openreview. net/forum?id $\underline{{\underline{{\mathbf{\Pi}}}}}$ z5uVAKwmjf.   
[30] Andy Zhou, Kai Yan, Michal Shlapentokh-Rothman, Haohan Wang, and Yu-Xiong Wang. Language agent tree search unifies reasoning, acting, and planning in language models. In Ruslan Salakhutdinov, Zico Kolter, Katherine Heller, Adrian Weller, Nuria Oliver, Jonathan Scarlett, and Felix Berkenkamp, editors, Proceedings of the 41st International Conference on Machine Learning, volume 235 of Proceedings of Machine Learning Research, pages 62138– 62160. PMLR, 21–27 Jul 2024. URL https://proceedings.mlr.press/v235/ zhou24r.html.   
[31] Wangchunshu Zhou, Yuchen Eleanor Jiang, Long Li, Jialong Wu, Tiannan Wang, Shi Qiu, Jintian Zhang, Jing Chen, Ruipu Wu, Shuai Wang, Shiding Zhu, Jiyu Chen, Wentao Zhang, Xiangru Tang, Ningyu Zhang, Huajun Chen, Peng Cui, and Mrinmaya Sachan. Agents: An open-source framework for autonomous language agents. 2023. URL https://arxiv. org/abs/2309.07870.  

[32] Mingchen Zhuge, Wenyi Wang, Louis Kirsch, Francesco Faccio, Dmitrii Khizbullin, and Jürgen Schmidhuber. Language agents as optimizable graphs, 2024. URL https://arxiv. org/abs/2402.16823.  

# Appendix  

# Contents  

1 Introduction 2  

2 Related Work  

3 Method 4  

# 4 EXPERIMENTS 9  

4.1 Evaluations over Three Designed Tasks 10   
4.2 Result for Gobang Game Development . 10   
4.3 Result for LaTeX Beamer Writing 10   
4.4 Result For Website Design . 11  

# 5 Workflow Update 12  

# 6 Conclusion 13  

A Human Evaluation Process 19   
B Experiment setups 19   
Experiment setup: LaTeX Beamer Writing 19   
Experiment setup: Gobang Game Development . . . 21   
B.3 Experiment setup: Website Design . . . 22   
B.4 How Different LLM Affect Updates . . . 22   
B.5 How Different LLM Affect Performance 23   
B.6 Time Cost of Different Baseline . 24  

# C Custom Metrics for Parallelism and Dependency 25  

C.1 Parallelism Metrics . 25   
C.2 Dependency Metrics . . . 25   
C.3 Proposed Metrics for Task Workflow Evaluation . 25  

# D Examples of Flow’s Workflow 25  

D.1 Example Workflow . 27   
D.2 Pseudocode for updating AOV 29   
D.3 Prompt for Workflow Update 31   
D.4 Workflow Update Strategies . 31  

# E Framework of the Multi-Agent framework 32  

E.1 Overview 32   
E.2 Key Components . . 32   
E.3 Workflow Execution Process 34  

# F Limitation and Future Work 35  

G Proof of Theorem 3.1 35  

### A Human Evaluation Process  

Sometimes, LLM can correctly fulfill each requirement of a task, but the quality of completion may vary. In such cases, human evaluation is necessary to assess the quality of the output. For each task, the final output of each multi-agent framework was evaluated by 50 participants, who ranked the outputs from best to worst. Points were awarded based on the rankings, with the 1st place receiving 4 points, the 2nd place receiving 3 points, etc. The final result was determined by calculating the average score. The detailed distribution is shown in Fig. 5.  

### B Experiment setups  

#### B.1 Experiment setup: LaTeX Beamer Writing  

# User input  

I am a lecturer teaching a machine learning course to research students, I am preparing lecture slides on various reinforcement learning algorithms. Note that:  

1). Given that the lecture duration is 2 hours, the slides should span approximately 30 pages.   
2). For each reinforcement learning algorithm covered, the slides will include the following key components: the motivation behind the algorithm, the problem it aims to solve, an intuitive solution, and the detailed mathematical equations that underpin the method.   
3). It is essential that the lecture is comprehensive and self-contained, providing students with a clear understanding of each algorithm from both a conceptual and technical perspective.  

The task involves generating a LaTeX Beamer presentation, which is a popular LaTeX class used to create professional-quality slides with various templates and effects. In this experiment, the objective is to produce presentations with different configurations, assessing the framework’s ability to follow instructions. The experiment includes the following configurations:  

Config 1: A 30-slide presentation, including motivation, problem statement, intuitive solution, and detailed mathematical equations.   
Config 2: A 20-slide presentation, including motivation, problem statement, intuitive solution, and detailed mathematical equations.   
Config 3: A 30-slide presentation, including motivation, problem statement, intuitive solution, and pseudocode.   
Config 4: A 20-slide presentation, including only motivation and intuitive solution.   
Config 5: A 30-slide presentation, including motivation, problem statement, intuitive solution, and detailed mathematical equations.  

The goal is to examine the framework’s ability to follow specific instructions while generating over 20 and 30 slides in different scenarios.  

![](https://cdn-mineru.openxlab.org.cn/extract/88a42faf-4fb8-4760-8acb-ab4ad5e88193/c4ef1d714a06de6b09ca4a5a0f78f4880abf08bce400109d1f2c8f916e5b4546.jpg)  
Figure 4: Ranking distribution for website design across different frameworks. The results indicate that our method (Flow) outperforms others by achieving the highest percentage of first-place rankings.  

![](https://cdn-mineru.openxlab.org.cn/extract/88a42faf-4fb8-4760-8acb-ab4ad5e88193/233d76987e1b57bf2b55cd5047cac97a0b0aa54da1639675aa9c3fb440b0d12d.jpg)  
Figure 5: Ranking distribution for gobang $\mathrm{gam}_{\mathrm{\overline{{{20}}}}}$ development across different frameworks. The results indicate that our method (Flow) outperforms others by achieving the highest percentage of first-place rankings.  

This task is well-suited for evaluation because it requires not only text generation but also an understanding of formatting and presentation logic. It serves as a comprehensive test of multitasking and reasoning capabilities. The structured nature of LaTeX allows for a rigorous assessment of the agent’s ability to manage complex, multicomponent tasks.  

Evaluation Metrics: The following metrics are used to assess the performance of the generated LaTeX Beamer writing:  

(1) Compilable: Verifies whether the generated LaTeX code can compiles into a valid Beamer presentation or not. A successful compilation is rewarded with a score of 1, otherwise 0.   
(2) Completeness: Ensures that the final Beamer presentation includes all required components like: motivation, problem, intuitive solution, and equations. Missing any of these results in a score of 0.   
(3) Page Limit: Assesses whether the presentation adheres to the specified page limits as outlined in the prompt.  

The final result is calculated as the average of these three scores and is shown as percentage.  

#### B.2 Experiment setup: Gobang Game Development  

# User input  

I am developing a Gobang game that includes a naive AI and a user interface. The game should end when either a player wins or the board is completely filled. The user interface must clearly indicate whose turn it is and display a message when the game concludes, specifying the winner. Additionally, the user should have the option to play as either black or white stones.  

Gobang, also called "Five in a Row", is a strategy board game where two players take turns placing black and white pieces on a grid. The objective is to be the first to align five consecutive pieces in a horizontal, vertical, or diagonal line. This experiment assesses our framework’s ability to efficiently develop the game by utilizing parallelism to divide the development process into smaller, manageable tasks, such as game logic, AI move generation, and user interface (UI) design. We apply the same approach, taking the average score from five trials.  

Evaluation Metrics: The following metrics are used to assess the performance of the generated Gobang game:  

(1) Compilable: The code compiles without errors. Any error that causes termination will result in a score of 0.   
(2) Interactable: Properly supports both user and AI movements. If both functions are achieved, score 1 else 0.   
(3) Game Rule: Ends correctly when five pieces are aligned, correct terminated will result in 1 final score.  

Each of these metrics is scored as 0 or 1, and the final result is calculated as the average of these scores and turned into a percentage. These metrics allow for a comprehensive assessment of the efficiency, accuracy, and adaptability of each framework in developing a functional Gobang game with AI capabilities.  

#### B.3 Experiment setup: Website Design  

# User input  

I am designing a website for the International Conference on Learning Representations (ICLR2025), which will take place from April 27, 2025, to May   
1, 2025, in San Francisco, California, United States. The conference is organized by the International Association for Learning Representations. Note that:   
1). For each section, I would like to see example HTML content. Additionally, a sample CSS stylesheet should be provided to style the website. The content must be professional, clear, and appropriate for an international academic conference.   
2). The website should include all the provided details, including a comprehensive conference schedule and a section dedicated to the conference venue, featuring a map.  

We tasked the frameworks with developing a comprehensive website for the ICLR conference to evaluate their ability to handle complex tasks that require both flexible task coordination and effective problem solving. This task tested the ability of the frameworks to manage multiple interdependent steps, such as designing user interfaces, ensuring functionality, and adhering to specific design guidelines.  

Evaluation Metrics: The following metrics are used to assess the performance of the generated website:  

(1) Compilable: Checks if the HTML renders into a functioning website, If yes then score 1, can’t render will result of score 0   
(2) Basic Information: Verifies the presence of essential details like conference name, date, location, and organizer. Missing any of the information will caused the score to be 0   
(3) Sections: Ensures inclusion of all required sections, with a focus on the schedule and venue as prompt asked. Missing the required part in the prompt will result in a score of 0 in score.  

By presenting a real-world scenario involving intricate requirements, we were able to observe how well the frameworks could break down a large project into manageable components and coordinate efforts across different tasks.  

#### B.4 How Different LLM Affect Updates  

To verify how our framework performs with different capabilities of LLM, we test both GPT-4omini and GPT-3.5-Turbo on three tasks we designed. In this experiment, each task was run five times on different models, and the average of the results was calculated as the final outcome. We recorded three metrics: average init task, average changed task, and average changed ratio.  

Init task refers to the number of subtasks that need to be executed within the workflow after selecting the optimal workflow but before the execution begins.   
Average changed task indicates the number of subtasks in the original workflow that were updated after the execution of the workflow.   
Average changed ratio is calculated by dividing the average changed task by the init task, providing a more intuitive reflection of the proportion of subtasks that were updated.  

Table 5: Update information on GPT-3.5-Turbo and GPT-4o-mini   


<html><body><table><tr><td>LLM-Agent</td><td>Task</td><td>InitialTasks (avg.)</td><td>Changed Tasks (avg.)</td><td>Changed Ratio (avg.)</td></tr><tr><td>GPT-3.5-Turbo</td><td>Gobang Game Development</td><td>7.8</td><td>3.4</td><td>44%</td></tr><tr><td></td><td>WebsiteDesign</td><td>7.2</td><td>4.8</td><td>66%</td></tr><tr><td></td><td>LaTeX Beamer Writing</td><td>6.2</td><td>4.4</td><td>71%</td></tr><tr><td>GPT-4o-mini</td><td>Gobang GameDevelopment</td><td>8</td><td>2.8</td><td>35%</td></tr><tr><td></td><td>WebsiteDesign</td><td>7.2</td><td>3.4</td><td>47%</td></tr><tr><td></td><td>LaTeX Beamer Writing</td><td>9.2</td><td>4.8</td><td>53%</td></tr></table></body></html>  

#### B.5 How Different LLM Affect Performance  

In this experiment, we used the GPT-3.5-Turbo model to conduct experiments on three tasks in different frameworks. Each task was executed five times. We evaluated the results using the same scoring matrix described above.  

Table 6: Comparison of LLM-based multi-agent frameworks on Gobang Game Development with GPT-3.5-Turbo   


<html><body><table><tr><td rowspan="2">Model</td><td colspan="4">Success Rate (%)</td></tr><tr><td>Compilable</td><td>Intractable</td><td>Game Rule</td><td>Overall Score</td></tr><tr><td>AutoGen [26]</td><td>80</td><td>20</td><td>20</td><td>40</td></tr><tr><td>MetaGPT [8]</td><td>80</td><td>20</td><td>40</td><td>53</td></tr><tr><td>CAMEL [10]</td><td>80</td><td>80</td><td>40</td><td>67</td></tr><tr><td>Flow (Ours)</td><td>100</td><td>100</td><td>60</td><td>87</td></tr></table></body></html>  

Table 7: Comparison of LLM-based multi-agent frameworks on Website Design with GPT-3.5- Turbo   


<html><body><table><tr><td rowspan="2">Model</td><td colspan="4">Success Rate (%)</td></tr><tr><td>Compilable</td><td>Basic Information</td><td>Sections</td><td>Overall Score</td></tr><tr><td>AutoGen [26]</td><td>20</td><td>0</td><td>0</td><td>7</td></tr><tr><td>MetaGPT[8]</td><td>80</td><td>60</td><td>60</td><td>67</td></tr><tr><td>CAMEL [10]</td><td>40</td><td>40</td><td>20</td><td>33</td></tr><tr><td>Flow (Ours)</td><td>100</td><td>100</td><td>40</td><td>80</td></tr></table></body></html>  

Table 8: Comparison of LLM-based multi-agent frameworks on LaTeX Beamer Writing with GPT3.5-Turbo   


<html><body><table><tr><td rowspan="2">Model</td><td colspan="4">Success Rate (%)</td></tr><tr><td>Compilable</td><td>Completeness</td><td>Page Limit</td><td>Overall Score</td></tr><tr><td>AutoGen [26]</td><td>40</td><td>0</td><td>0</td><td>13</td></tr><tr><td>MetaGPT [8]</td><td>20</td><td>20</td><td>0</td><td>13</td></tr><tr><td>CAMEL [10]</td><td>80</td><td>80</td><td>0</td><td>53</td></tr><tr><td>Flow (Ours)</td><td>100</td><td>100</td><td>0</td><td>67</td></tr></table></body></html>  

Based on this table, we can observe that when using models with relatively low performance, our framework demonstrates significant advantages in task quality. Overall, even when using less powerful LLM like GPT-3.5-Turbo, our framework consistently maintains a high standard of performance.  

#### B.6 Time Cost of Different Baseline  

To quantitatively measure the cost of our framework, we use execution time as a standard. Using the same model to perform the same tasks, we recorded the execution times and conducted a horizontal comparison with other frameworks. Each task was executed five times, and the average execution time was calculated.  

<html><body><table><tr><td>Task</td><td>Flow (w/o update)</td><td>Flow (w/ update)</td><td>MetaGPT</td><td>CAMEL</td><td>AutoGen</td></tr><tr><td>GPT-3.5-Turbo</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>GobangGame</td><td>26.12± 11.35</td><td>33.57 ± 12.46</td><td>34.00 ± 15.12</td><td>121.52 ± 20.87</td><td>31.00 ± 14.67</td></tr><tr><td>WebsiteWebsite</td><td>23.46 ± 10.84</td><td>34.23 ± 13.12</td><td>85.14 ± 18.52</td><td>41.96 ± 12.89</td><td>44.00 ± 15.34</td></tr><tr><td>Latex Beamer</td><td>18.34 ± 9.73</td><td>24.12 ± 10.89</td><td>29.92 ± 14.87</td><td>166.00 ± 22.64</td><td>31.00 ± 16.78</td></tr><tr><td>GPT-4o-mini</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Gobang Game</td><td>60.45 ± 14.78</td><td>72.34 ± 13.45</td><td>99.45 ± 16.92</td><td>110.94 ± 19.67</td><td>148.72 ± 25.34</td></tr><tr><td>WebsiteWebsite</td><td>51.98 ± 20.19</td><td>52.14 ± 14.89</td><td>127.49 ± 17.52</td><td>74.53 ± 18.34</td><td>86.78 ± 21.23</td></tr><tr><td>Latex Beamer</td><td>53.19 ± 17.65</td><td>83.34 ± 15.89</td><td>66.72 ± 19.45</td><td>106.34 ± 20.78</td><td>95.21 ± 22.56</td></tr></table></body></html>  

Table 9: Comparison of task performance across different framework, including standard deviations. The standard deviations reflect realistic variability with increased variance across tasks and framework.  

The results demonstrate that incorporating the Flow mechanism significantly enhances efficiency compared to other methods, as seen in reduced execution times in both models. However, the introduction of updates incurs additional computational overhead, resulting in a noticeable increase in execution time, highlighting the trade-off between adaptability and efficiency. Nonetheless, Flow maintains faster execution times compared to several other frameworks.  

### C Custom Metrics for Parallelism and Dependency  

#### C.1 Parallelism Metrics  

Speedup $\begin{array}{r}{(S=\frac{T_{1}}{T_{p}})}\end{array}$ , this metric measures the ratio of execution time on a single processor $(T_{1})$ to that on multiple processors $(T_{p})$ . While effective in frameworks where these times can be measured, it requires actual execution on both single and multiple processors. In our case, such execution times are not readily obtainable because our focus is on task-solving workflows rather than on processing workloads that can be easily benchmarked in this way.  

Amdahl’s Law $\begin{array}{r}{(S(p)=\frac{1}{f_{s}+\frac{1-f_{s}}{p}})}\end{array}$ and Gustafson’s Law $(S(p)=p-f_{s}\cdot(p-1))$ , both laws require knowledge of $f_{s}$ , the proportion of the task that is inherently serial, and $p$ , the number of processors. Our task graphs have complex dependency structures, where tasks cannot be neatly categorized as strictly "serial" or "parallel." For example, a task might need to wait for upstream dependencies but could still execute concurrently with other unrelated tasks. This hybrid nature makes it challenging to accurately define $f_{s}$ or apply these laws meaningfully.  

#### C.2 Dependency Metrics  

Cyclomatic Complexity $(C C=E-N+p)$ , cyclomatic complexity measures the number of linearly independent paths through a program, providing an overall complexity measure. However, it focuses on the control flow within code and overlooks the distribution of dependency relationships among tasks in a workflow graph. It does not capture the "dependency concentration" or "dispersion," which are crucial to understanding the impact of dependencies on workflow robustness and the ease with which LLM can comprehend and update the workflow.  

#### C.3 Proposed Metrics for Task Workflow Evaluation  

Given these limitations, we use two simple metrics in our LLM-based multi-agent framework:  

1). Parallelism Metric: This metric does not rely on execution time measurements or require assumptions about tasks being strictly serial or parallel. It directly reflects the workflow’s potential for concurrent task execution, making it more applicable to our scenario.  

2). Dependency Metric: We focus on the "dependency concentration" or "dependency dispersion" by analyzing the standard deviation of the degree distribution in the task graph. This metric provides an intuitive reflection of critical dependency points within the workflow. By highlighting how dependencies are distributed among tasks, it helps us understand and mitigate potential bottlenecks, enhancing both robustness and the LLM’s ability to process workflow updates efficiently.  

### D Examples of Flow’s Workflow  

In this section, we present examples of actual workflows generated by Flow.  

Fig.6 showing Flow’s workflow in generating LaTeX Beamer, Flow concurrently generates the four required components for each algorithm: motivation, problem, intuitive solution, and mathematical equations.  

![](https://cdn-mineru.openxlab.org.cn/extract/88a42faf-4fb8-4760-8acb-ab4ad5e88193/6f0abab024836d6f2562932b9166044fdde93ed4b7abbc7aea011a57301028ba.jpg)  
Figure 6: Workflow of LaTeX Beamer Writing in Flow  

For the task of developing a gobang game, Flow recognizes that the UI and main game logic can be separated and executed in parallel to enhance overall speed and efficiency, as shown in Fig.7. Additionally, there remains a clear sequential process; for instance, the game rules must be defined first before the corresponding code can be deployed.  

![](https://cdn-mineru.openxlab.org.cn/extract/88a42faf-4fb8-4760-8acb-ab4ad5e88193/14f0e1ea3f0c7a7df9b09eb1479b9de8768102df33ada3d7b9a00c9bb0878029.jpg)  
Figure 7: Workflow of Gobang Game Development  

For the task of website design, as shown in Fig.8, Flow treats different parts of the HTML as individual subtasks, which helps to increase overall speed. Additionally, dividing the process into separate components allows for parallel execution and improved modularity, ensuring that if an issue arises in one part of the HTML, it will not impact the performance of other sections. This approach improves both efficiency and fault tolerance.  

![](https://cdn-mineru.openxlab.org.cn/extract/88a42faf-4fb8-4760-8acb-ab4ad5e88193/c9e0cf7f6756c4546fbf8e1ffe1254dd71cd6f1489e2e90cc5e9c120d9c5e0aa.jpg)  
Figure 8: Workflow of Website Design  

#### D.1 Example Workflow  

![](https://cdn-mineru.openxlab.org.cn/extract/88a42faf-4fb8-4760-8acb-ab4ad5e88193/d26ef71b72d540d465fc2664201757750c351eb15b48df6aff7ece96fe77ba21.jpg)  
Figure 9: A workflow of Website Design in VSCode  

![](https://cdn-mineru.openxlab.org.cn/extract/88a42faf-4fb8-4760-8acb-ab4ad5e88193/5970fd09492740a19baaa45d79e13af9abba70f2806c8c4260ac59371cd43796.jpg)  
Figure 10: Different multi-agent frameworks’ LaTeX Beamer  

#### D.2 Pseudocode for updating AOV  

Algorithm 1 Helper Function for Updating Graph Function Updat $\overline{{\mathrm{eGraph}(\tilde{G},\mathcal{P},\mathcal{P}_{i n i t})}}$ : $//$ Generate updated candidate workflows using LLM // Initialize selection variables 2 $P_{\mathrm{{max}}}\leftarrow-\infty\mathrm{{}}C_{\mathrm{{min}}}\leftarrow+\infty\mathrm{{}}\tilde{G}_{\mathrm{{optimal}}}\leftarrow\mathrm{{None}}$ // Evaluate each candidate workflow 3 for each candidate workflow $\tilde{G}_{k}$ in $\{\tilde{G}_{1},\tilde{G}_{2},\dots,\tilde{G}_{K}\}$ do Compute Parallelism $\begin{array}{r l r}{P_{k}}&{{}\leftarrow}&{P_{\mathrm{avg}}(\tilde{G}_{k})}\end{array}$ Compute Dependency Complexity $C_{k}\gets$ $C_{\mathrm{dependency}}(\tilde{G}_{k})$ 5 if $P_{k}>P_{m a x}{\pmb o r}(P_{k}==P_{m a x}$ and $C_{k}<C_{m i n}$ ) then 6 $P_{\mathrm{max}}\leftarrow P_{k}\leftarrow C_{\mathrm{min}}\leftarrow C_{k}\tilde{G}_{\mathrm{optimal}}\leftarrow\tilde{G}_{k}$ 7 end 8 end 9 return Goptimal  

# Algorithm 2 Flow  

Data: Task Requirements $\mathcal{P}$ , Initialization Prompt $\mathcal{P}_{\mathrm{init}}$ , Update Prompt Pupdate Result: Optimized Multi-Agent Workflow  

// Step 1: Implement a Workflow using a dictionary structure 10 Initialize workflow formulation by defining the task dictionary G where each key $v\in V$ maps to a dictionary containing: $\begin{array}{r}{\tilde{G}[v]=\left\{\begin{array}{l l}\end{array}\right.}\end{array}$ {status, data, num_parents_not_completed, child, agent}  

// Step 2: Generate an Initial Workflow 11  

// Step 3: Workflow Refinement and Dynamic Updating 12 while there exists at least one sub-task in $\tilde{G}$ that is not completed do 13 if an update to the workflow is required then // Generate and Select the Best Updated Workflow 14 $\tilde{G}\gets\mathrm{UpdateGraph}(\tilde{G},\mathcal{P}_{\mathrm{update}},\mathcal{P})$ Update workflow dictionary $\tilde{G}$ to $\tilde{G}_{\mathrm{best}}$ // Regenerate Execution Plan and Reallocate Agents 15 Perform Topological Sort on $\tilde{G}$ to obtain updated execution order $\sigma$ Assign agents $A_{j}$ to their respective sub-tasks ${\mathcal{T}}_{j}\subseteq V$ 16 end // Execute Available Sub-tasks in Parallel 17 foreach sub-task $v_{i}\in V$ do 18 if status of $v_{i}$ is not started and $\tilde{G}[v_{i}]$ .num_parents_not_completed $==0$ then 19 if agent $a_{j}$ is available then 20 Assign agent $a_{j}$ to sub-task $v_{i}$ 21 else 22 Clone agent $a_{j}^{\prime}$ Assign cloned agent $a_{j}^{\prime}$ to sub-task $v_{i}$ 23 end // Execute subtask $v_{i}$ in parallel 24 Execute $v_{i}$ using agent $a_{j}$ or cloned agent $a_{j}^{\prime}$ concurrently // Update Subtask Status and Data 25 Update status of sub-task $v_{i}$ to in progress // After execution, update related data 26 Update output of subtask $v_{i}$ to $\tilde{G}[v_{i}]$ .data $\tilde{G}[v_{i}]$ .status $\leftarrow$ “completed” // Update Child Tasks’ Parent Completion Count 27 foreach child task $c\in\tilde{G}[v_{i}]$ .child do 28 $\tilde{G}[c]$ .num_parents_not_completed $\leftarrow\tilde{G}[c]$ .num_parents_not_completed − 1 29 end 30 end 31 end 32 end  

#### D.3 Prompt for Workflow Update  

1. Update the Workflow - Evaluate Completed Tasks:  

- Focus: Examine only tasks with ‘"status": "completed"   
- Check Data: - Ensure that ‘"data"‘ for each task is sufficient, detailed, and directly contributes to the ‘final_goal‘.  

- Assess Workflow Structure:  

- Examine All Tasks: Review all tasks, including those labeled ‘"completed "‘, ‘"pending"‘, and ‘"in-progress"‘.   
- Check Adequacy: - Confirm the workflow is complete and logically structured to achieve the ‘final_goal‘. - Ensure there are no missing critical tasks or dependencies. Verify that ‘"next"‘ and ‘"prev"‘ connections between tasks are logical and facilitate seamless progression.  

- Identify Inefficiencies: - Detect and address unnecessary dependencies, bottlenecks, or redundant steps that hinder the workflow’s efficiency.  

- Allowed Changes:  

- Modify: Clarify and detail the objectives of tasks with insufficient or vague directives to ensure they meet the ‘final_goal‘.   
- Add: Introduce new tasks with clear, detailed descriptions to fill gaps in data or structure.   
- Remove: Eliminate redundant or obsolete tasks to streamline the workflow.  

- Maintain Logical Flow: - Reorganize task connections (‘"next"‘ and ‘"prev"‘) to enhance parallel execution and improve overall workflow efficiency.  

2. Output Format - If No Changes Are Made:  

- Return an empty JSON object to indicate that no modifications were necessary: $\mathrm{\partialjson}\{\}^{\mathrm{~\epsilon~}}$ . - If Changes Are Made: - Return a JSON object containing the updated workflow without including the ‘"data"‘ fields to optimize token usage. This JSON should only include the structural changes (task parameters and connections).  

#### D.4 Workflow Update Strategies  

We implemented two different workflow update strategies:  

#### • Update Concurrently  

In this approach, when a subtask is completed, it immediately triggers the workflow update function, even if other subtasks are still running. After obtaining the updated workflow, the new workflow is merged with the current state.  

– Trade-off: This workflow update strategy runs concurrently with task execution, optimizing running time. However, it can result in unnecessary API calls, as some subtasks still in progress may become redundant or misaligned with the updated workflow.  

#### • Update After Task Completion  

In this strategy, when a subtask is completed, no new tasks are allocated immediately. Instead, the system waits for all running subtasks to finish before triggering the workflow update. After the update is completed, new subtasks are allocated based on the updated workflow. This approach reduces unnecessary API calls by batching updates.  

– Trade-off: This workflow update strategy reduces unnecessary API calls but increases overall running time, as new subtasks are delayed until the workflow update is complete.  

In our paper, all the experiments are obtained by using the second strategy to avoid the waste of API usage.  

### E Framework of the Multi-Agent framework  

#### E.1 Overview  

The multi-agent framework is designed to execute complex tasks by decomposing them into subtasks, which are managed and executed by individual agents. The framework leverages LLM to generate and update workflows dynamically, ensuring robustness, efficiency, and adaptability.  

#### E.2 Key Components  

#### 1. Agents  

• Role Assignment  

– Automatic Role Generation: Roles are automatically generated by LLM during workflow generation and updates.   
– Flexibility: By default, roles are not fixed, allowing the system to adapt to the specific requirements of each task.   
– Role Constraints: In scenarios with resource constraints, roles can be explicitly defined to limit the number of agents or types of expertise in prompt.  

#### • Subtask Assignment  

– Matching Expertise: Subtasks are assigned to agents whose roles best match the task requirements, ensuring tasks are executed by agents with appropriate skills. – One Agent per Subtask: Only one agent is assigned per subtask to maintain clarity and responsibility.  

#### 2. Workflow Management  

#### • Workflow Generation  

– Initial Workflow: The LLM generates an initial workflow that outlines all subtasks and their dependencies required to achieve the final goal. – Task Dependencies: Dependencies are defined to ensure logical progression and to facilitate parallel execution where possible.  

#### • Workflow Update Mechanisms  

– Two strategies are employed for updating the workflow:  

(a) Update Concurrently  

$\star$ Trigger: When a subtask is completed, the workflow update function is triggered immediately, even if other subtasks are still running.   
$\star$ Process: The updated workflow is obtained and merged with the current state.   
$\star$ Trade-off: Optimizes running time but may result in unnecessary API calls, as some subtasks still in progress might become redundant after the update.  

#### (b) Update After Subtask Completion  

∗ Trigger: No new subtasks are allocated immediately after a subtask is completed. The system waits for all running subtasks to finish before updating. $\star$ Process: Once all subtasks are completed, the workflow is updated, and new subtasks are allocated based on the updated workflow. ∗ Trade-off: Reduces unnecessary API calls but increases overall running time, as new subtasks are delayed until the workflow update is complete. $\star$ Chosen Strategy: In practice, the system uses the second strategy to reduce API usage.  

#### 3. Dynamic Restructuring  

#### • Mechanism for Dynamic Workflow Restructuring  

– Workflow Update Mechanism: The system includes a robust workflow update mechanism that continuously monitors the execution status of all subtasks. If a subtask fails or is deemed unsolvable, the system triggers an update process.  

– Re-evaluation of Workflow: The system systematically reviews the current workflow, taking into account the unsolvable subtask. It assesses the impact of the failed subtask on all subtasks and the overall goal.  

– Adjusting Dependencies: The workflow is adjusted by removing or modifying the unsolvable subtask and updating dependencies accordingly. This may involve:  

$\star$ Reassigning Subtasks: Redirecting subtasks to alternative agents or creating new subtasks that can achieve similar outcomes.   
$\star$ Adding New Subtasks: Introducing new subtasks that offer alternative solutions or pathways to reach the final goal.   
$\star$ Bypassing Unnecessary Steps: If possible, restructuring the workflow to bypass the unsolvable subtask without compromising the end objectives.  

#### 4. Task Execution  

#### • Parallelism  

– Maximizing Parallel Execution: The workflow is designed to allow subtasks without dependencies to be executed in parallel, optimizing resource utilization and reducing total execution time.  

– Dependency Management: Dependencies are minimized where possible to enhance parallelism.  

#### • Dependency Minimization  

– Dependency Metric: The system analyzes the standard deviation of the degree distribution in the task graph to identify and minimize critical dependency points. – Reducing Bottlenecks: By minimizing unnecessary dependencies, the system reduces potential bottlenecks and enhances robustness.  

#### E.3 Workflow Execution Process  

#### 1. Initial Workflow Generation  

• The LLM generates a workflow based on the final goal, decomposing it into subtasks with defined dependencies.  

#### 2. Agent Role Assignment  

• Agents are assigned roles automatically by the LLM.   
• Subtasks are assigned to agents based on role matching.  

#### 3. Subtask Execution  

• Agents execute their assigned subtasks.   
• Subtasks are executed in parallel where dependencies allow.  

#### 4. Monitoring and Updates  

• The system monitors subtask completion statuses.   
• Depending on the update strategy, the workflow is updated either concurrently or after all current subtasks are completed.  

#### 5. Dynamic Restructuring  

• Detection: If a subtask is determined to be insufficient or unsolvable for achieving the requirement, the system detects this during execution.   
• Re-evaluation of Workflow: The system reviews the current workflow, assessing the impact of the failed subtask on all subtasks and the overall goal.   
• Workflow Adjustment: The LLM restructures the workflow dynamically to adjust other subtasks or redefine dependencies.   
• Continuity: This ensures that progress toward the final goal continues without significant delays.  

#### 6. Completion  

• The process continues until all subtasks are completed and the final goal is achieved.  

### F Limitation and Future Work  

Although we have generated multiple candidate workflows and selected the one with the highest modularity, it is still not the most efficient. With sufficient computing and data resources, a model trained specifically for workflow management could significantly enhance the framework’s performance. For instance, the LLM could be designed to maximize a reward function centered on key performance indicators such as task completion speed, resource utilization, and minimizing disruptions in the workflow. Such training could lead to the development of more effective workflows. The workflow updater requires global information to function effectively, which can become problematic as the context length increases. This limitation could be addressed by employing a rig or a hierarchical approach to more precisely identify errors or areas lacking efficiency, thereby facilitating more targeted updates and improvements within the workflow.  

### G Proof of Theorem 3.1  

Proof. We will compare the expected number of successfully completed subtasks in both workflows.  

#### Definitions:  

• Let $P_{A}(v)$ and $P_{B}(v)$ denote the probability that subtasks $v$ is successfully completed in Workflow A and Workflow B, respectively.   
• For each subtasks $v$ , let $D_{A}(v)$ and $D_{B}(v)$ be the sets of immediate predecessors of $v$ in Workflow A and Workflow B, respectively.  

Success Probability of a subtasks: In Workflow A, the success probability of subtasks $v$ is given by:  

$$
P_{A}(v)=(1-p_{f})\times\prod_{i\in D_{A}(v)}P_{A}(i).
$$  

Similarly, in Workflow B:  

$$
P_{B}(v)=(1-p_{f})\times\prod_{i\in D_{B}(v)}P_{B}(i).
$$  

Base Case: Since the subtasks $v$ with no dependencies (i.e., $D_{A}(v)=D_{B}(v)=\emptyset)$ have the same success probability in both workflows:  

$$
P_{A}(v)=P_{B}(v)=1-p_{f}.
$$  

Inductive Step: We proceed by induction on the subtasks’ dependency levels.  

Comparison for Subtasks $v^{*}$ : Subtasks $v^{*}$ has an additional dependency $d$ in Workflow B. Therefore:  

$$
D_{B}(v^{*})=D_{A}(v^{*})\cup\{d\}.
$$  

Using equations (1) and (2), we have:  

$$
\begin{array}{r l}&{P_{A}(v^{*})=(1-p_{f})\times\displaystyle\prod_{i\in D_{A}(v^{*})}P_{A}(i),}\ &{P_{B}(v^{*})=(1-p_{f})\times\displaystyle\prod_{i\in D_{B}(v^{*})}P_{B}(i)=(1-p_{f})\times P_{B}(d)\times\displaystyle\prod_{i\in D_{A}(v^{*})}P_{B}(i).}\end{array}
$$  

Since $D_{A}(v^{*})=D_{B}(v^{*})\setminus\{d\}$ , and $P_{A}(i)=P_{B}(i)$ for all $i\neq v^{*}$ (because their dependencies are the same), it follows that:  

$$
P_{B}(v^{*})=P_{A}(v^{*})\times P_{B}(d).
$$  

Because $0<P_{B}(d)=P_{A}(d)<1$ (since $p_{f}>0$ ), we have:  

$$
P_{B}(v^{*})=P_{A}(v^{*})\times P_{A}(d)<P_{A}(v^{*}).
$$  

Success Probabilities for Other Subtasks: For all subtasks $\boldsymbol{v}\neq\boldsymbol{v}^{*}$ , $D_{A}(v)=D_{B}(v)$ , so:  

$$
P_{A}(v)=P_{B}(v).
$$  

Expected Number of Successfully Completed Subtasks: The expected number of successfully completed subtasks in each workflow is:  

$$
\begin{array}{l}{{\displaystyle{\cal E}[S_{A}]=\sum_{v\in{\cal T}}P_{A}(v)},}\ {{\displaystyle{\cal E}[S_{B}]=\sum_{v\in{\cal T}}P_{B}(v)}.}\end{array}
$$  

Substituting the above findings:  

$$
\begin{array}{l}{{\displaystyle{E[S_{B}]=\sum_{v\neq v^{*}}P_{B}(v)+P_{B}(v^{*})}}}\ {{\displaystyle~=\sum_{v\neq v^{*}}P_{A}(v)+P_{B}(v^{*})}}\ {{\displaystyle~=\left(\sum_{v\in T}P_{A}(v)-P_{A}(v^{*})\right)+P_{B}(v^{*})}}\ {{\displaystyle~=E[S_{A}]-\left(P_{A}(v^{*})-P_{B}(v^{*})\right).}}\end{array}
$$  

Since $P_{B}(v^{*})<P_{A}(v^{*})$ , the difference $\Delta P=P_{A}(v^{*})-P_{B}(v^{*})>0$ . Thus,  

$$
E[S_{B}]=E[S_{A}]-\Delta P<E[S_{A}].
$$  

Therefore, the expected number of successfully completed subtasks in Workflow A is strictly greater than in Workflow B:  

$$
E[S_{A}]>E[S_{B}].
$$  