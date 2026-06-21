from graphviz import Digraph

dot = Digraph('DRLComLibSystem', format='png')
dot.attr(rankdir='TB', bgcolor='white')
dot.attr('node', shape='box', style='rounded,filled', fontname='Helvetica', fontsize='11')

dot.node('learner', 'Central Learner', fillcolor='#c6dbef')

dot.attr('node', shape='ellipse', style='filled', fillcolor='#f0f0f0', fontsize='11')
dot.node('transport', 'ZeroMQ\nPUSH/PULL, PUB/SUB')

dot.attr('edge', fontsize='9')

def white_label(text):
    return f'''<
      <TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="2">
        <TR><TD BGCOLOR="white"><FONT POINT-SIZE="9">{text}</FONT></TD></TR>
      </TABLE>
    >'''

# Learner <-> transport with white box labels
# Add spaces before "rollouts" so the label sits a bit to the left
dot.edge('learner', 'transport',
         label=white_label('  rollouts'),  # two leading spaces
         arrowsize='0.8')
dot.edge('transport', 'learner',
         label=white_label('weights'),
         style='dashed', arrowsize='0.8')

dot.attr('node', shape='box', style='rounded,filled', fontname='Helvetica', fontsize='10')

def add_actor(dot, name):
    with dot.subgraph(name=f'cluster_{name}') as c:
        c.attr(style='rounded', color='#cccccc', label=f'Actor {name}', fontsize='10')
        c.node(f'{name}_pol', 'Policy', fillcolor='#fee6ce')
        c.node(f'{name}_env', 'Environment', fillcolor='#e5f5e0')
        c.node(f'{name}_buf', 'Local buffer', fillcolor='#f7f7f7')

        c.edge(f'{name}_pol', f'{name}_env', arrowsize='0.7')
        c.edge(f'{name}_env', f'{name}_buf', arrowsize='0.7')

    # actor buffer -> transport (rollouts up)
    dot.edge(f'{name}_buf', 'transport',
             label=white_label('rollouts'),
             arrowsize='0.8')
    # transport -> policy (weights down)
    dot.edge('transport', f'{name}_pol',
             label=white_label('weights'),
             style='dashed', arrowsize='0.8')

add_actor(dot, 'A0')
add_actor(dot, 'A1')

dot.attr(rank='same')
dot.edge('transport', 'A0_env', style='invis')
dot.edge('transport', 'A1_env', style='invis')

output_path = dot.render('drl_comlib_system_vertical', cleanup=True)
print('Wrote:', output_path)