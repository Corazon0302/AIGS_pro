from jinja2 import Template

idea_gen = Template('''
    You are a highly capable research assistant with expertise in designing scientific research trajectories. Your task is to rigorously generate a trajectory for a piece of scientific work based on the data provided in JSON format. The trajectory should be meticulously designed to achieve the stated research objective while innovating beyond the existing work referenced in the provided data. Your output must exhibit scientific rigor, follow research best practices, and contain sufficient detail for implementation.

    Please generate a trajectory for the following scientific work:

    {{ data | indent(4) }}
    
    Your output format should be as follows:
    {
        "rename": "the name of the scientific work",
        "predecessor_work":[
            "The predecessor work 1 I provided.",
            "The predecessor work 2 I provided."
        ],
        "objective": "The objective I provided.",
        "motivation": "The motivation for the scientific work.",
        "workflow": "The detailed workflow you have designed for the scientific work. You should describe the workflow in natural language or pseudocode, outlining the steps involved in the research process. Be sure to include all necessary details and considerations for each step."
        "experimental_design": "The experimental design you have developed for the scientific work. This should include the methodology, procedures, and techniques you plan to use to conduct the research. Provide a detailed description of the experimental setup, data collection methods, and analysis techniques."
    }
                    
''')