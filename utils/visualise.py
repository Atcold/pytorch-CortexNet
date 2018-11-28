from graphviz import Digraph
from torch.autograd import Variable
import sys, subprocess
import uuid


def make_dot(root):
    node_attr = dict(style='filled',
                     shape='box',
                     align='left',
                     fontsize='12',
                     ranksep='0.1',
                     height='0.2')
    dot = Digraph(node_attr=node_attr, graph_attr=dict(size="12,12"))
    seen = set()

    def add_nodes(var):
        if var not in seen:
            if isinstance(var, Variable):
                value = '(' + ', '.join(['%d'% v for v in var.size()]) + ')'
                dot.node(str(id(var)), str(value), fillcolor='lightblue')
            else:
                dot.node(str(id(var)), str(type(var).__name__))
            seen.add(var)
            if hasattr(var, 'next_functions'):
                for u in var.next_functions:
                    dot.edge(str(id(u[0])), str(id(var)))
                    add_nodes(u[0])
    add_nodes(root.grad_fn)
    return dot


def show_graph(root):
    dot_file_name = '/tmp/' + str(uuid.uuid4())
    make_dot(root).render(dot_file_name)
    pdf_file_name = dot_file_name + '.pdf'
    if sys.platform == 'darwin':
        subprocess.call(('open', pdf_file_name))
    elif sys.platform == 'linux':
        subprocess.call(('xdg-open', pdf_file_name))


__author__ = "Sergey Zagoruyko and Alfredo Canziani"
__credits__ = ["Sergey Zagoruyko", "Alfredo Canziani"]
__maintainer__ = "Alfredo Canziani"
__email__ = "alfredo.canziani@gmail.com"
__status__ = "Production"  # "Prototype", "Development", or "Production"
__date__ = "Feb 17"
