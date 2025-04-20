import geopandas
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go


def map_bar_cam_us(polygon, points, mode, dir_save):
    if mode == 'map':
        from matplotlib.colors import ListedColormap
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        polygon.plot(
            ax=ax, column='if_top', legend=False, cmap=ListedColormap(['#DCE4F2', '#6F518C']),
            edgecolor='#404040', linewidth=.1
        )
        points.sjoin(polygon, how='inner').plot(ax=ax, markersize=.5, color='#F2055C')
        ax.set_axis_off()
        ax.set_xlim(polygon.total_bounds[[0, 2]])
        ax.set_ylim(polygon.total_bounds[[1, 3]])
        plt.savefig(f'{dir_save}/map_us_cam.png', dpi=300, bbox_inches='tight', pad_inches=0)
        plt.show()
    if mode == 'bar':
        x_values = [
            polygon[polygon['if_top'] == False]['cam_per_pop'].mean() * 100000,
            polygon[polygon['if_top'] == True]['cam_per_pop'].mean() * 100000,
        ]
        fig = go.Figure([go.Bar(
            y=['Lower half counties', 'Upper half counties'], x=x_values,
            marker=dict(color=['#DCE4F2', '#6F518C']),
            text=[f'{i:.2f}' for i in x_values], textposition='inside', orientation='h'
        )])
        fig.update_layout(
            yaxis=dict(
                showline=True, linewidth=2, linecolor='black', showgrid=False,
                ticks='outside', tickformat=',', zeroline=False,
            ),
            xaxis=dict(
                title='CCTV count per 100,000 people',
                showline=True, linewidth=2, linecolor='black', showgrid=False,
                ticks='outside', tickformat=',', zeroline=False, title_font=dict(size=20)
            ),
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
            width=900, height=150, font=dict(size=20), margin=dict(l=0, r=0, t=0, b=0)
        )
        fig.show(renderer="browser")
        fig.write_image(
            f'{dir_save}/bar_us_cam.png',
            width=900, height=150, scale=3.125
        )

    return


def map_city_roads_polygon_crime(city_b, roads=None, polygons=None, crimes=None, network=None):
    import geopandas as gpd
    fig, ax = plt.subplots(figsize=(10, 10))

    # if city_b is not None:
    city_boundary = gpd.read_file(city_b)
    city_boundary = city_boundary[city_boundary['NAME'] == 'Warner Robins']
    city_boundary = city_boundary.to_crs(epsg=4326)
    city_boundary.plot(ax=ax, facecolor='lightgray', edgecolor='none')

    if roads is not None:
        if network is None:
            roads = roads.to_crs(epsg=4326)
            roads_short = roads[roads.intersects(city_boundary.buffer(0.002).unary_union)]
            roads_short.plot(ax=ax, color='#3C535B', linewidth=1)

    if polygons is not None:
        polygons = polygons.to_crs(epsg=4326)
        polygons.plot(ax=ax, facecolor='none', edgecolor='white', linewidth=0.5)

    if crimes is not None:
        crimes = crimes.to_crs(epsg=4326)
        crimes.sample(1000).plot(ax=ax, color='#F2055C',  markersize=5, alpha=0.6)

    if network is not None:
        import networkx as nx
        assert roads is not None
        roads = roads.to_crs(epsg=4326)
        roads_short = roads[roads.intersects(city_boundary.buffer(0.002).unary_union)]
        # roads_short.plot(ax=ax, color='#808080', linewidth=1)

        roads_2 = roads_short.copy()
        roads_2["x"] = roads_2.geometry.centroid.x
        roads_2["y"] = roads_2.geometry.centroid.y
        node_positions = {row['id']: (row['x'], row['y']) for _, row in roads_2.iterrows()}
        valid_nodes = set(node_positions.keys())
        filtered_edges = [(u, v) for u, v in network.copy().edges() if u in valid_nodes and v in valid_nodes]

        graph = nx.DiGraph()
        graph.add_edges_from(filtered_edges)
        nx.draw(
            graph, pos=node_positions, ax=ax,
            node_size=10, node_color="#808080", edge_color="#808080", width=1, font_size=1
        )

    # show
    ax.set_axis_off()
    xmin, ymin, xmax, ymax = city_boundary.total_bounds
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    plt.tight_layout(pad=.5)
    plt.show()

    return

