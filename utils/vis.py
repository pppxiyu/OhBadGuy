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


def scatter_crime_pred_metrics(
        metric_list_1, metric_list_2,
        group_labels=('Dynamic','Static'), y_label='Metric value', annotate=False,
        stats_test='mannwhitneyu', alternative=None, save_path=None, if_percentile=False
):
    import numpy as np
    from scipy import stats
    import matplotlib.patheffects as pe

    fig, ax = plt.subplots(figsize=(2.2, 4.5))

    plt.rcParams['font.family'] = 'Arial'

    data1, data2 = map(np.asarray, (metric_list_1, metric_list_2))
    ax.scatter([0] * data1.size, data1, color='#79B4D9', edgecolors='#666666', linewidths=1, s=50)
    ax.scatter([1] * data2.size, data2, color='#D99B66', edgecolors='#666666', linewidths=1, s=50)

    if annotate:
        txt_style = dict(
            ha='center', va='center', fontsize=7, zorder=3, color='white',
            path_effects=[pe.Stroke(linewidth=1.2, foreground='black'), pe.Normal()]
        )
        for i, y in enumerate(data1):
            ax.text(0, y, str(i), **txt_style)
        for i, y in enumerate(data2):
            ax.text(1, y, str(i), **txt_style)

    for xpos, dat in zip([0, 1], [data1, data2]):
        m, sd = dat.mean(), dat.std(ddof=1)
        ax.hlines(m, xpos + 0.2, xpos + 0.4, lw=1.5, color='#595959')
        ax.vlines(xpos + 0.3, m - sd, m + sd, lw=1.5, color='#595959')

    if stats_test == 'ttest':
        _, p_val = stats.ttest_ind(data1, data2, equal_var=False)
    elif stats_test == 'mannwhitneyu':
        """
            Use the Mann-Whitney U test when you have two independent groups with small 
            sample sizes and your data are non-normal or ordinal.
        """
        assert alternative in ('two-sided', 'less', 'greater')
        _, p_val = stats.mannwhitneyu(data1, data2, alternative=alternative)
    else:
        raise ValueError(f'Unknown stats_test: {stats_test}')

    y_max = max(data1.max(), data2.max())
    bracket_bottom = y_max + 0.02 * y_max
    bracket_top = bracket_bottom + 0.05 * y_max
    ax.plot(
        [0, 0, 1, 1], [bracket_bottom, bracket_top, bracket_top, bracket_bottom], lw=0.7, color='#454545'
    )
    ax.text(
        0.5, bracket_top + 0.03 * y_max,f"$\\it{{p}}$ = {p_val:.3f}",
        ha='center', va='bottom', color='#454545', fontsize=12,
    )

    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    gap_frac = 0.03
    ax.spines['bottom'].set_bounds(-0.3 + gap_frac, 1.3)
    ax.spines['left'].set_color('#454545')
    ax.spines['bottom'].set_color('#454545')

    ax.set_xticks([0, 1])
    ax.set_xticklabels(group_labels, fontsize=12, color='#454545')
    ax.set_ylabel(y_label, fontsize=14, color='#454545')

    ax.tick_params(axis='both', colors='#454545', labelsize=14)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    ax.set_xlim(-0.4, 1.4)
    y_min = min(data1.min(), data2.min())
    ax.set_ylim(y_min - 0.3 * y_min, bracket_top + 0.05 * y_max)
    ax.spines['bottom'].set_position(('data', y_min - 0.35 * y_min))
    plt.tight_layout()

    if save_path is None:
        plt.show()
    else:
        fig.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0)


def density_crime_map(
    values, adj_matrix, city_b, roads, polygons, crimes=None,
    spatial_resolution=100, n_layers=20, save_path=None, power_transform=1, 
):
    import geopandas as gpd
    import numpy as np
    from scipy.stats import gaussian_kde
    from matplotlib.colors import LinearSegmentedColormap

    # Check if values is a list of two arrays for difference mapping
    is_difference_map = isinstance(values, list)
    if is_difference_map:
        assert len(values) == 2, "For difference mapping, values must be a list of exactly 2 arrays"
        values1, values2 = values[0], values[1]
    
    fig, ax = plt.subplots(figsize=(10, 10))
    
    """
    Base map
    """
    city_boundary = gpd.read_file(city_b)
    city_boundary = city_boundary[city_boundary['NAME'] == 'Warner Robins']
    city_boundary = city_boundary.to_crs(epsg=4326)
    city_boundary.plot(ax=ax, facecolor='lightgray', edgecolor='none')
    
    if roads is not None:
        roads = roads.to_crs(epsg=4326)
        roads_short = roads[roads.intersects(city_boundary.buffer(0.002).unary_union)]
        # Use lighter gray for roads in difference map
        road_color = '#999999' if is_difference_map else '#3C535B'
        roads_short.plot(ax=ax, color=road_color, linewidth=1)
    
    if polygons is not None:
        polygons = polygons.to_crs(epsg=4326)
        polygons.plot(ax=ax, facecolor='none', edgecolor='white', linewidth=0.5)
    
    if crimes is not None:
        crimes = crimes.to_crs(epsg=4326)
        crimes.sample(1000).plot(ax=ax, color='#F2055C', markersize=5, alpha=0.6)
    
    """
    KDE calculation
    """
    def calculate_kde(values_input):
        """Helper function to calculate KDE for a given values array"""
        # Get centroids of roads as the spatial locations
        geometries = roads.to_crs(epsg=4326)
        x_coords = geometries.centroid.x.values
        y_coords = geometries.centroid.y.values
        
        # Calculate spatially-lagged values using the adjacency matrix
        spatial_values = np.zeros(len(values_input))
        for i in range(len(values_input)):
            neighbors = np.where(adj_matrix[i] > 0)[0]
            if len(neighbors) > 0:
                spatial_values[i] = values_input[i] + np.sum(values_input[neighbors])
            else:
                spatial_values[i] = values_input[i]
        
        # Convert value to number of points
        values_normalized = spatial_values - spatial_values.min()
        values_normalized = values_normalized / values_normalized.max() * 100
        
        x_count = []
        y_count = []
        for i, (x, y, v) in enumerate(zip(x_coords, y_coords, values_normalized)):
            n_points = max(1, int(v))
            x_count.extend([x] * n_points)
            y_count.extend([y] * n_points)
        
        # Perform Gaussian KDE on the weighted point cloud
        positions = np.vstack([x_count, y_count])
        kernel = gaussian_kde(positions, bw_method='scott')
        
        # Create a regular grid covering the city boundary
        xmin, ymin, xmax, ymax = city_boundary.total_bounds
        xx, yy = np.mgrid[xmin:xmax:spatial_resolution*1j, ymin:ymax:spatial_resolution*1j]
        positions_grid = np.vstack([xx.ravel(), yy.ravel()])
        density = np.reshape(kernel(positions_grid).T, xx.shape)
        
        # Mask density values outside city boundary
        from shapely.geometry import Point
        city_geom_buffered = city_boundary.buffer(0.0011).unary_union
        xx_flat = xx.ravel()
        yy_flat = yy.ravel()
        grid_points = gpd.GeoDataFrame(
            geometry=[Point(x, y) for x, y in zip(xx_flat, yy_flat)],
            crs='EPSG:4326'
        )
        city_boundary_buffered = gpd.GeoDataFrame(
            geometry=[city_geom_buffered],
            crs='EPSG:4326'
        )
        points_in_city = gpd.sjoin(grid_points, city_boundary_buffered, predicate='within', how='left')
        mask_flat = points_in_city.index_right.isna().values
        mask = mask_flat.reshape(xx.shape)
        density_masked = np.ma.masked_array(density, mask=mask)

        return xx, yy, density_masked
    
    # Calculate KDE(s)
    if is_difference_map:
        xx, yy, density1 = calculate_kde(values1)
        _, _, density2 = calculate_kde(values2)
        density_diff = density2 - density1
        density = density_diff
    else:
        xx, yy, density = calculate_kde(values)
    
    """
    Visualization
    """
    if is_difference_map:
        # Use diverging colormap for difference map
        cmap = 'RdBu_r'  # Red for increase, Blue for decrease
        
        # Center colormap at zero
        vmax = np.abs(density_diff).max()
        vmin = -vmax
        
        contour = ax.contourf(xx, yy, density_diff, levels=n_layers, 
                             cmap=cmap, alpha=0.6, zorder=1, 
                             vmin=vmin, vmax=vmax)
        
        cbar = plt.colorbar(contour, ax=ax, fraction=0.046, pad=0.02)
        cbar.set_label('Crime density change', 
                      rotation=270, labelpad=25, fontsize=22)
        
        # Optional: Add zero contour line to show boundary between increase/decrease
        ax.contour(xx, yy, density_diff, levels=[0], colors='black', 
                  linewidths=1.5, linestyles='--', alpha=0.5)

    else:
        # Apply power transformation if specified
        density_transformed = np.power(density, power_transform)

        # Original colormap for single density map
        colors = ['white', '#FFE6E6', '#FF9999', '#FF4444', '#CC0000', '#800000']
        n_bins = 100
        cmap = LinearSegmentedColormap.from_list('crime_density', colors, N=n_bins)

        contour = ax.contourf(
            xx, yy, density_transformed, levels=n_layers, cmap=cmap, alpha=0.6, zorder=1,
        )
        cbar = plt.colorbar(contour, ax=ax, fraction=0.046, pad=0.02)
        cbar.set_label('Crime density', rotation=270, labelpad=20, fontsize=22)
    
    # Show
    ax.set_axis_off()
    xmin, ymin, xmax, ymax = city_boundary.total_bounds
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    
    if save_path is None:
        plt.tight_layout(pad=.5)
        plt.show()
    else:
        fig.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0)
        plt.close(fig)


def density_crime_map_sequence(
    values, city_b, roads, polygons, crimes=None,
    save_path=None, 
):
    import geopandas as gpd
    import numpy as np
    from scipy.stats import gaussian_kde
    from matplotlib.colors import LinearSegmentedColormap

    fig, ax = plt.subplots(figsize=(10, 10))
    
    """
    Base map
    """
    city_boundary = gpd.read_file(city_b)
    city_boundary = city_boundary[city_boundary['NAME'] == 'Warner Robins']
    city_boundary = city_boundary.to_crs(epsg=4326)
    city_boundary.plot(ax=ax, facecolor='lightgray', edgecolor='none')
    
    if roads is not None:
        roads = roads.to_crs(epsg=4326)
        roads_short = roads[roads.intersects(city_boundary.buffer(0.002).unary_union)]
        roads_short.plot(ax=ax, color='#3C535B', linewidth=0.5)
    
    if polygons is not None:
        polygons = polygons.to_crs(epsg=4326)
        polygons.plot(ax=ax, facecolor='none', edgecolor='white', linewidth=1)
    
    if crimes is not None:
        crimes = crimes.to_crs(epsg=4326)
        crimes.sample(1000).plot(ax=ax, color='#F2055C', markersize=5, alpha=0.6)
    
    """
    Visualization
    """
    # Parameter to control number of bins
    n_bins = 6  # Adjust this value to control the number of color groups
    
    ranks = np.argsort(np.argsort(values))
    percentiles = ranks / len(ranks) * 100
    
    # Define percentile bins
    bins = np.linspace(0, 100, n_bins + 1)
    
    # Generate colors dynamically based on n_bins
    base_colors = ['#FFFFFF', '#FFE5E5', '#FFB3B3', '#FF6B6B', '#E63946', '#8B0000']
    if n_bins <= len(base_colors):
        colors = base_colors[:n_bins]
    else:
        # Interpolate additional colors if n_bins > base colors
        cmap = LinearSegmentedColormap.from_list('rank_cmap', base_colors)
        colors = [cmap(i / (n_bins - 1)) for i in range(n_bins)]
        colors = [plt.matplotlib.colors.rgb2hex(c) for c in colors]
    
    # Assign each polygon to a color group
    color_groups = np.digitize(percentiles, bins) - 1
    color_groups = np.clip(color_groups, 0, n_bins - 1)  # Ensure within range
    polygons['color_group'] = [colors[i] for i in color_groups]
    
    polygons.plot(ax=ax, color=polygons['color_group'], 
                  edgecolor='gray', linewidth=0.3, legend=False, alpha=0.75)
    
    # Re-plot roads and polygon edges on top
    if roads is not None:
        roads_short.plot(ax=ax, color='#808080', linewidth=.5, zorder=3)
    
    if polygons is not None:
        polygons.plot(ax=ax, facecolor='none', edgecolor='#D3D3D3', linewidth=.5, zorder=2)

    # Show
    ax.set_axis_off()
    xmin, ymin, xmax, ymax = city_boundary.total_bounds
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    
    if save_path is None:
        plt.tight_layout(pad=.5)
        plt.show()
    else:
        fig.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0)
        plt.close(fig)     




