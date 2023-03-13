import sys
sys.path.append('..')
import pandas as pd
import networkx as nx
import csv
import matplotlib.pyplot as plt

def calculate_betweenness_centrality(filepath):
    # Define a dictionary to store the country-to-continent mapping
    country_to_continent = {}

    # Open the all.csv file
    with open('all.csv', newline='', encoding='utf8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            country_code = row['alpha-2']
            continent = row['region']
            if country_code not in country_to_continent:
                country_to_continent[country_code] = [continent]
            else:
                country_to_continent[country_code].append(continent)

    # Convert the dictionary to a pandas DataFrame
    country_to_continent_df = pd.DataFrame(country_to_continent.items(), columns=['ISO_CODE', 'Continent'])
    country_to_continent_df.set_index('ISO_CODE', inplace=True)

    # Merge the ISO code to continent mapping with the edge list data
    df = pd.read_csv(filepath)
    df = df.merge(country_to_continent_df, left_on='ISO_CODE_1', right_index=True)
    df = df.merge(country_to_continent_df, left_on='ISO_CODE_2', right_index=True, suffixes=['_1', '_2'])

    # Create a new graph and add the edges from the dataframe
    G = nx.from_pandas_edgelist(df, source='ISO_CODE_1', target='ISO_CODE_2', edge_attr='FBDist')

    # Add continent information to the nodes in the graph
    for iso_code in G.nodes:
        continents = country_to_continent[iso_code]
        # Convert the list of continents to a tuple if necessary
        if isinstance(continents, list):
            continents = tuple(continents)
        G.nodes[iso_code]['Continent'] = continents

    # Group the edges in the graph by continent and calculate the betweenness centrality for each continent
    centrality_by_continent = {}
    for continent in set(nx.get_node_attributes(G, 'Continent').values()):
        nodes = [n for n, v in G.nodes(data=True) if v['Continent'] == continent]
        subgraphs = [G.subgraph(c).copy() for c in nx.connected_components(G.subgraph(nodes))]
        subgraph_centrality = {}
        for subgraph in subgraphs:
            subgraph_centrality.update(nx.betweenness_centrality(subgraph, weight='FBDist', normalized=True, endpoints=True))
        centrality_by_continent[continent] = subgraph_centrality

    # Print the betweenness centrality for each continent
    for continent, centrality in centrality_by_continent.items():
        print(continent, sum(centrality.values()))

    return G, centrality_by_continent

G, centrality_by_continent = calculate_betweenness_centrality("../../data/fb_data/FBCosDist.csv")

def draw_graph(G, node_size=20, node_color='blue', edge_color='gray', alpha=0.5):
    # Define the layout of the graph
    pos = nx.spring_layout(G)

    # Draw the graph
    nx.draw(G, pos, node_size=node_size, node_color=node_color, edge_color=edge_color, alpha=alpha)

    # Show the plot
    plt.show()
    
draw_graph(G)