from jinja2 import Template

paper_read = Template('''
Please read the following paper in detail, and then summarize the trajectory of the method in the paper for me in the following format:
TrajectoryDesign = {
    "name": "The title of the paper",
    "predecessor_work": "Detailed information of previous work mentioned in the paper that has been done in relation to the work done in the paper. You can even copy the Related Work section from the paper and directly put it here.",
    "objective": "str", # The goal of the method proposed in the paper, what problem is it intended to solve, what effect is achieved, such as improving the efficiency of a certain process or saving costs, etc. 
    "motivation": "str", # The motivation for the method proposed in the paper, such as the background reasons, the problems faced that need to be solved, or what inspired the research of this work. The following content is the specific structure of the method used in the paper. Please use the structure of a number of list including several dictions to express it. Each list represents a high-level architecture of the workflow, and the dict in the list represents a number of process details in the architecture. As an example, the workflow architecture of a certain paper can be divided into two parts: pre-falsification and falsification, each of which contains a number of specific steps. What I need you to do is not to organize the thesis workflow in the following format, but to **automatically think deeply** and **come up with a new workflow architecture** for the thesis 
                    
    "workflow":"Describe the workflow of the work in the paper in detail using natural language or pseudocode, detailing the stages of the workflow as well as specific design details. Be sure to include all necessary details and considerations for each step.",
    "experimental_design": "The method of experiment design in this paper is described in detail, including the benchmark used, the experiment environment configuration, the experiment result analysis method and other necessary design details mentioned in the paper" 
                      
}

The content of the paper in markdown format:
{{ paper | indent(4) }}
                    
**Attention**:
Your output must be in json format, and dictionary keys must be enclosed in double quotes, not single quotes.
''')