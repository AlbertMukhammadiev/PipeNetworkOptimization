import random
import json
import time
import networkx as nx
import matplotlib.pyplot as plt
from layouts import CompleteLayout, DelaunayLayout, SteinerLayout, MultilevelStarLayout, SquareGrid, HexagonGrid
from optimize import GA, SA
from builder import SimpleBuilder, BuilderByConfig
from cost_models import cost_model1
import helper


def initial_layout2_draw():
    from layouts import create_layout2
    la = create_layout2()
    pos = nx.get_node_attributes(la, 'pos')
    kwargs = {
        'node_color': [],
        'node_size': 30,
        'node_shape': 's',
        'alpha': 1,
        'font_size': 1,
        'rotate': False,
        'width': 0.2,
        'labels': {},
        'edge_labels': {}}
    for node, props in la.nodes(data=True):
        label = f'{node}\npos: {props["pos"]}\n{props["demand"]}'
        kwargs['labels'][node] = ''
        if props['demand'] > 0:
            color = 'lightgreen'
        elif props['demand'] < 0:
            color = 'salmon'
        else:
            color = 'palegoldenrod'
        kwargs['node_color'].append(color)

    for u, v, props in la.edges(data=True):
        info = []
        for key, value in props.items():
            info.append(f'{key}: {value}')
        label = '\n'.join(info)
        kwargs['edge_labels'][u, v] = ''

    plt.axis('equal')
    plt.axis('off')
    nx.draw_networkx_nodes(la, pos, **kwargs)
    nx.draw_networkx_labels(la, pos, **kwargs)
    nx.draw_networkx_edges(la, pos, **kwargs)
    nx.draw_networkx_edge_labels(la, pos, **kwargs)
    plt.savefig('_user_initial_layout_.pdf', format='pdf')
    plt.close()


def cost_model_to_table():
    from pandas import DataFrame
    from pandas.plotting import table
    df = DataFrame()
    df['Diameter'] = [item['diam'] for item in cost_model1]
    df['Cost'] = [item['cost'] for item in cost_model1]
    ax = plt.subplot(111, frame_on=False)
    tab = table(ax, df, loc='center')  # where df is your data frame
    tab.set_fontsize(9)
    tab.scale(0.5, 0.9)
    plt.axis('off')
    plt.savefig('_cost_model_.pdf')
    plt.close()

def layout2_nodes_to_table():
    import pandas as pd
    from pandas.plotting import table
    terminals = [
        ((34.6, 4.6), 15),
        ((1.6, 7), 30),
        ((27.5, 9.5), 10),
        ((14.6, 11.7), 10),
        ((34.6, 16), 20),
        ((2.8, 18), 5),
        ((8.9, 17.6), 10),
        ((26.2, 18.5), 15),
        ((20.1, 22.4), 20),
        ((30.7, 24.6), 25),
        ((40.5, 23.8), 10),
        ((9.9, 28.9), 5),
        ((3.9, 29.9), 10),
        ((14.2, 32.2), -1),
        ((33.3, 31.2), 20),
        ((21.1, 36.1), 30),
        ((7.9, 37), 10),
        ((16.3, 43.2), 12)
    ]

    df = pd.DataFrame()
    df['Position'] = [point for point, demand in terminals]
    df['Demand'] = [demand for point, demand in terminals]

    ax = plt.subplot(111, frame_on=False)
    tab = table(ax, df, loc='center')  # where df is your data frame
    tab.set_fontsize(9)
    tab.scale(0.5, 0.9)
    plt.axis('off')
    plt.savefig('_user_terminals_.pdf')
    plt.close()

def terminals_to_table(terminals_path):
    import pandas as pd
    from pandas.plotting import table

    terminals = get_terminals(terminals_path)
    df = pd.DataFrame()
    df['Position'] = [point for point, demand in terminals]
    df['Demand'] = [demand for point, demand in terminals]

    ax = plt.subplot(111, frame_on=False)
    tab = table(ax, df, loc='center')  # where df is your data frame
    tab.set_fontsize(9)
    tab.scale(0.5, 0.9)
    plt.axis('off')
    plt.savefig('_' + terminals_path.replace('.json', '.pdf'))
    plt.close()


def draw_from_log_and_layout(layout, path):
    builder = BuilderByConfig(layout, cost_model1)
    with open(path + '.json', 'r') as fin:
        data = json.load(fin)
    best_config = data['best'][0]
    best_cost = data['best'][1]
    builder.redesign(best_config)
    design = builder.current_design
    draw(design, best_cost, path + '_image_.pdf')



def draw_from_log(path, terminals_path):
    terminals = get_terminals(terminals_path)
    if '_complete_' in path:
        layout = CompleteLayout(terminals)
    elif '_delaunay_' in path:
        layout = DelaunayLayout(terminals)
    elif '_square_' in path:
        layout = SquareGrid(terminals, 8, 8)
    elif '_hexagon_' in path:
        layout = HexagonGrid(terminals, 8, 8)

    builder = BuilderByConfig(layout, cost_model1)
    with open(path + '.json', 'r') as fin:
        data = json.load(fin)
    best_config = data['best'][0]
    best_cost = data['best'][1]
    builder.redesign(best_config)
    design = builder.current_design
    draw(design, best_cost, path + '_image.pdf')


def draw_convergence(path):
    with open(path + '.json', 'r') as fin:
        data = json.load(fin)
    if '_ga_' in path:
        fig, ax = plt.subplots()
        ax.set_xlabel('Номер популяции')
        ax.set_ylabel('Общая стоимость')
        info = f'Количество индивидумов {data["n_individuals"]}\n' \
            f'Вероятность мутации {data["mutation_probability"]}\n' \
            f'Вероятность кроссинговера {data["crossover_probability"]}\n' \
            f'Затраченное время {helper.round2(data["time"])}'

        xs = list(range(len(data['min'])))
        ys_min = data['min']

        plt.scatter(xs, ys_min)
        plt.figtext(.5, .7, info)
        plt.savefig(path + '_convergence_.pdf')
        plt.close()
    elif '_sa_' in path:
        fig, ax = plt.subplots()
        ax.set_xlabel('Температура')
        ax.set_ylabel('Общая стоимость')
        info = f'Начальная температура T {data["T"]}\n' \
            f'Конечная температура t {data["t"]}\n' \
            f'Коэфф. изменения температуры {data["alpha"]}\n' \
            f'Длина цепи Макрова {data["L"]}\n' \
            f'Затраченное время {helper.round2(data["time"])}'

        xs, ys = [], []
        for t, cost in data['costs']:
            if t not in xs:
                xs.append(t)
                ys.append(cost)
            else:
                if cost < ys[len(ys) - 1]:
                    ys[len(ys) - 1] = cost
        plt.xlim(max(xs), min(xs))
        plt.scatter(xs, ys)
        plt.figtext(.5, .7, info)
        plt.savefig(path + '_convergence_.pdf')
        plt.close()


def draw(design, cost, path):
    pos = nx.get_node_attributes(design, 'pos')
    kwargs = {
        'node_color': [],
        'node_size': 30,
        'node_shape': 's',
        'alpha': 1,
        'font_size': 1,
        'rotate': False,
        'width': 0.2,
        'labels': {},
        'edge_labels': {}}
    for node, props in design.nodes(data=True):
        label = f'{node}\npos: {props["pos"]}\n{props["demand"]}'
        kwargs['labels'][node] = ''
        if props['demand'] > 0:
            color = 'lightgreen'
        elif props['demand'] < 0:
            color = 'salmon'
        else:
            color = 'palegoldenrod'
        kwargs['node_color'].append(color)

    for u, v, props in design.edges(data=True):
        info = []
        for key, value in props.items():
            info.append(f'{key}: {value}')
        label = '\n'.join(info)
        kwargs['edge_labels'][u, v] = ''

    plt.axis('equal')
    plt.axis('off')
    plt.title(f'cost: {helper.round2(cost)}')
    nx.draw_networkx_nodes(design, pos, **kwargs)
    nx.draw_networkx_labels(design, pos, **kwargs)
    nx.draw_networkx_edges(design, pos, **kwargs)
    nx.draw_networkx_edge_labels(design, pos, **kwargs)
    plt.savefig(path, format='pdf')
    plt.close()


def test_terminals():
    return [
        ((7.5, 2), 20),
        ((19.8, 3.2), 20),
        ((34.6, 4.6), 15),
        ((1.6, 7), 30),
        ((27.5, 9.5), 10),
        ((14.6, 11.7), 10),
        ((34.6, 16), 20),
        ((2.8, 18), 5),
        ((8.9, 17.6), 10),
        ((26.2, 18.5), 15),
        ((20.1, 22.4), 20),
        ((30.7, 24.6), 25),
        ((40.5, 23.8), 10),
        ((9.9, 28.9), 5),
        ((3.9, 29.9), 10),
        ((14.2, 32.2), -1111),
        ((33.3, 31.2), 20),
        ((21.1, 36.1), 30),
        ((7.9, 37), 10),
        ((16.3, 43.2), 12),
    ]


def create_terminals(n=30):
    points = [helper.random_point(1, 500) for _ in range(n)]
    terminals = [{'pos': p, 'demand': random.choice([5, 10, 15, 20])} for p in points]
    pos = random.randint(0, n - 1)
    terminals[pos]['demand'] = -1
    with open(f'terminals_{n}.json', 'w') as f:
        json.dump(terminals, f, indent=4)


def get_terminals(terminals_path):
    with open(terminals_path, 'r') as fin:
        data = json.load(fin)
    return [(tuple(record['pos']), record['demand']) for record in data]


def ga_delaunay(fname):
    terminals = get_terminals(fname)
    layout = DelaunayLayout(terminals)
    ga = GA(layout, cost_model1)
    ga.optimize()


def sa_delaunay(fname):
    terminals = get_terminals(fname)
    layout = DelaunayLayout(terminals)
    sa = SA(layout, cost_model1)
    sa.optimize()


def ga_complete(fname):
    terminals = get_terminals(fname)
    layout = CompleteLayout(terminals)
    ga = GA(layout, cost_model1)
    ga.optimize()


def sa_complete(fname):
    terminals = get_terminals(fname)
    layout = CompleteLayout(terminals)
    sa = SA(layout, cost_model1)
    sa.optimize()


def ga_square(fname):
    terminals = get_terminals(fname)
    layout = SquareGrid(terminals, 8, 8)
    ga = GA(layout, cost_model1)
    ga.optimize()


def sa_square(fname):
    terminals = get_terminals(fname)
    layout = SquareGrid(terminals, 8, 8)
    sa = SA(layout, cost_model1)
    sa.optimize()


def ga_hexagon(fname):
    terminals = get_terminals(fname)
    layout = HexagonGrid(terminals, 8, 8)
    ga = GA(layout, cost_model1)
    ga.optimize()


def sa_hexagon(fname):
    terminals = get_terminals(fname)
    layout = HexagonGrid(terminals, 8, 8)
    sa = SA(layout, cost_model1)
    sa.optimize()


def simple_steiner(terminals_path):
    start = time.time()
    terminals = get_terminals(terminals_path)
    layout = SteinerLayout(terminals)
    builder = SimpleBuilder(layout, cost_model1)
    end = time.time()
    design = builder.current_design
    data = {
        'best': ([], builder.development_cost()),
        'time': end - start,
    }
    with open('_simple_steiner_.json', 'w') as f:
        json.dump(data, f, indent=4)
    draw(design, builder.development_cost(), '_simple_steiner__image.pdf')


def simple_multilevel_star(terminals_path):
    start = time.time()
    terminals = get_terminals(terminals_path)
    layout = MultilevelStarLayout(terminals)
    builder = SimpleBuilder(layout, cost_model1)
    end = time.time()
    design = builder.current_design
    data = {
        'best': ([], builder.development_cost()),
        'time': end - start,
    }
    with open('_simple_multilevelstar_.json', 'w') as f:
        json.dump(data, f, indent=4)
    draw(design, builder.development_cost(), '_simple_multilevelstar__image.pdf')




if __name__ == '__main__':
    def experiment_simple_steiner(terminals_path):
        simple_steiner(terminals_path)

    def experiment_simple_mstar(terminals_path):
        simple_multilevel_star(terminals_path)

    def experiment_sa_complete(terminals_path):
        sa_complete(terminals_path)
        draw_from_log('_sa_complete_', terminals_path)
        draw_convergence('_sa_complete_')

    def experiment_ga_complete(terminals_path):
        ga_complete(terminals_path)
        draw_from_log('_ga_complete_', terminals_path)
        draw_convergence('_ga_complete_')

    def experiment_ga_delaunay(terminals_path):
        ga_delaunay(terminals_path)
        draw_from_log('_ga_delaunay_', terminals_path)
        draw_convergence('_ga_delaunay_')

    def experiment_sa_delaunay(terminals_path):
        sa_delaunay(terminals_path)
        draw_from_log('_sa_delaunay_', terminals_path)
        draw_convergence('_sa_delaunay_')

    def experiment_ga_square(terminals_path):
        ga_square(terminals_path)
        draw_from_log('_ga_square_', terminals_path)
        draw_convergence('_ga_square_')

    def experiment_sa_square(terminals_path):
        # sa_square(terminals_path)
        draw_from_log('_sa_square_', terminals_path)
        draw_convergence('_sa_square_')

    def experiment_ga_hexagon(terminals_path):
        # ga_hexagon(terminals_path)
        draw_from_log('_ga_hexagon_', terminals_path)
        draw_convergence('_ga_hexagon_')

    def experiment_sa_hexagon(terminals_path):
        sa_hexagon(terminals_path)
        draw_from_log('_sa_hexagon_', terminals_path)
        draw_convergence('_sa_hexagon_')

    def experiment_ga_user():
        from layouts import create_layout2
        layout = create_layout2()
        ga = GA(layout, cost_model1)
        ga.optimize()
        draw_from_log_and_layout(layout, '_ga_user_')
        draw_convergence('_ga_user_')

    def experiment_sa_user():
        from layouts import create_layout2
        layout = create_layout2()
        sa = SA(layout, cost_model1)
        sa.optimize()
        draw_from_log_and_layout(layout, '_sa_user_')
        draw_convergence('_sa_user_')
    #
    #
    # cost_model_to_table()
    # layout2_nodes_to_table()
    # path = 'terminals_30.json'
    # experiment_simple_steiner(path)
    # experiment_simple_mstar(path)
    # experiment_sa_complete(path)
    # experiment_ga_complete(path)
    # experiment_ga_delaunay(path)
    # experiment_sa_delaunay(path)
    # experiment_ga_square(path)
    # experiment_sa_square(path)
    # experiment_ga_hexagon(path)
    # experiment_sa_hexagon(path)




    terminals_to_table('terminals_30.json')