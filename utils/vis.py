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


def line_marginal_gain_centrality(scores, save_path=None):
    """
    Plot marginal gain and cumulative gain of centrality scores
    """
    import numpy as np

    # Calculate marginal gains (the scores themselves are marginal gains)
    marginal_gains = scores
    
    # Calculate cumulative gains
    cumulative_gains = np.cumsum(scores)
    
    # Number of sensors
    n_sensors = np.arange(1, len(scores) + 1)
    
    # Create figure and primary axis - wider for more space
    fig, ax1 = plt.subplots(figsize=(7, 2.5))
    
    # Plot marginal gain on left y-axis (slightly stronger red)
    color_marginal = '#E53935'  # Slightly stronger red
    ax1.set_xlabel('Number of Sensors', fontsize=14, color='black')
    ax1.set_ylabel('Marginal Gain\nof Centrality', fontsize=14, color='black')
    line1 = ax1.plot(n_sensors, marginal_gains, color=color_marginal, 
                     marker='o', markersize=5, linewidth=2.5)
    ax1.tick_params(axis='y', labelcolor='black', labelsize=12)
    ax1.tick_params(axis='x', labelsize=12)
    
    # Set y-axis limits with margin above max value
    max_marginal = max(marginal_gains)
    ax1.set_ylim(bottom=0, top=max_marginal * 1.1)
    ax1.locator_params(axis='y', nbins=6)
    
    # Set x-axis to show all integer values
    ax1.set_xticks(n_sensors)
    
    # Create secondary y-axis for cumulative gain
    ax2 = ax1.twinx()
    color_cumulative = '#7E57C2'  # Slightly stronger purple
    ax2.set_ylabel('Cumulative Gain\nof Centrality', fontsize=14, color='black')
    line2 = ax2.plot(n_sensors, cumulative_gains, color=color_cumulative, 
                     marker='s', markersize=5, linewidth=2.5)
    ax2.tick_params(axis='y', labelcolor='black', labelsize=12)
    
    # Set y-axis limits with margin above max value
    max_cumulative = max(cumulative_gains)
    ax2.set_ylim(bottom=0, top=max_cumulative * 1.1)
    ax2.locator_params(axis='y', nbins=6)
    
    # Add box boundary
    for spine in ax1.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.2)
    
    if save_path is None:
        plt.tight_layout(pad=.5)
        plt.show()
    else:
        fig.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0)
        plt.close(fig)


def map_placement(
    placement_placed, values, city_b, roads, polygons, draw_polygon=False,
    save_path=None, draw_legend=True,
    placement_candidate=None,
):
    import geopandas as gpd
    import numpy as np
    from scipy.stats import gaussian_kde
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.lines import Line2D
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
        roads_short.plot(ax=ax, color='#3C535B', linewidth=1.5)
    
    if draw_polygon and polygons is not None:
        polygons = polygons.to_crs(epsg=4326)
        polygons.plot(ax=ax, facecolor='none', edgecolor='white', linewidth=1)
    if draw_polygon:
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
                    edgecolor='none', linewidth=0, legend=False, alpha=0.75)
        
        # Re-plot roads on top
        if roads is not None:
            roads_short.plot(ax=ax, color='#808080', linewidth=.5, zorder=3)
    
    """
    Add candidate placement dots (behind placed dots)
    """
    if placement_candidate is not None:
        # Filter roads to only those in the candidate placement list
        candidate_roads = roads[roads['id'].isin(placement_candidate)]
        
        # Get the centroids of the selected roads
        candidate_points = candidate_roads.geometry.centroid
        
        # Plot the candidate placement locations with blue color
        candidate_points.plot(ax=ax, color='#4A90E2', markersize=200, 
                             edgecolor='white', linewidth=2, 
                             zorder=4, alpha=0.9)
    
    """
    Add placed placement dots (on top)
    """
    # Filter roads to only those in the placement list
    placement_roads = roads[roads['id'].isin(placement_placed)]
    
    # Get the centroids of the selected roads
    placement_points = placement_roads.geometry.centroid
    
    # Plot the placement locations as larger dots with red color
    placement_points.plot(ax=ax, color='#E63946', markersize=200, 
                         edgecolor='white', linewidth=2, 
                         zorder=5, alpha=0.9)
    
    if draw_legend:
        """
        Add legend
        """
        legend_elements = []
        if placement_candidate is not None:
            legend_elements.append(Line2D([0], [0], marker='o', color='w', 
                                        markerfacecolor='#4A90E2', markersize=10,
                                        # markeredgecolor='white', markeredgewidth=1.5,
                                        label='Candidate location'))
        legend_elements.append(Line2D([0], [0], marker='o', color='w', 
                                    markerfacecolor='#E63946', markersize=10,
                                    # markeredgecolor='white', markeredgewidth=1.5,
                                    label='Selected location'))
        
        ax.legend(handles=legend_elements, loc='lower left', frameon=False, fontsize=18)
    
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


def map_placement_directional(
    placement_placed, direction_info, values, city_b, roads, polygons, draw_polygon=False,
    save_path=None, draw_legend=True,
    pre_placement=None,
):
    import geopandas as gpd
    import numpy as np
    from scipy.stats import gaussian_kde
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.lines import Line2D
    from matplotlib.patches import FancyArrow
    from shapely.geometry import Point, LineString
    import matplotlib.pyplot as plt
    
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
        roads_short.plot(ax=ax, color='#3C535B', linewidth=1.5)
    
    if draw_polygon and polygons is not None:
        polygons = polygons.to_crs(epsg=4326)
        polygons.plot(ax=ax, facecolor='none', edgecolor='white', linewidth=1)
    
    if draw_polygon:
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
                    edgecolor='none', linewidth=0, legend=False, alpha=0.75)
        
        # Re-plot roads on top
        if roads is not None:
            roads_short.plot(ax=ax, color='#808080', linewidth=.5, zorder=3)
    
    """
    Add candidate placement dots (behind placed dots)
    """
    if pre_placement is not None:
        # Filter roads to only those in the candidate placement list
        candidate_roads = roads[roads['id'].isin(pre_placement)]
        
        # Get the centroids of the selected roads
        candidate_points = candidate_roads.geometry.centroid
        
        # Plot the candidate placement locations with purple color
        candidate_points.plot(ax=ax, color='#9B59B6', markersize=200, 
                             edgecolor='white', linewidth=2, 
                             zorder=4, alpha=0.9)
    
    """
    Add placed placement triangles with directional arrows (on top)
    """
    def calculate_direction_angle(road_geom, connected_roads_geoms):
        """
        Calculate the direction angle based on the placement road and connected roads.
        Returns angle in degrees (0 = right, 90 = up, etc.)
        """
        # Get the centroid of the placement road
        center = road_geom.centroid
        
        # Calculate the average direction to all connected roads
        direction_vectors = []
        for connected_geom in connected_roads_geoms:
            connected_center = connected_geom.centroid
            dx = connected_center.x - center.x
            dy = connected_center.y - center.y
            direction_vectors.append((dx, dy))
        
        if not direction_vectors:
            return 0  # Default direction if no connections
        
        # Average the direction vectors
        avg_dx = np.mean([v[0] for v in direction_vectors])
        avg_dy = np.mean([v[1] for v in direction_vectors])
        
        # Calculate angle in degrees
        angle = np.degrees(np.arctan2(avg_dy, avg_dx))
        return angle
    
    # Calculate triangle size - increased for better visibility
    xmin, ymin, xmax, ymax = city_boundary.total_bounds
    map_width = xmax - xmin
    triangle_size = map_width * 0.04  # Increased from 0.025 to 0.04
    
    # Process each placement
    for road_id, direction in placement_placed:
        # Get the road geometry
        road_geom = roads[roads['id'] == road_id].geometry.iloc[0]
        center = road_geom.centroid
        
        # Get connected roads for this direction
        if road_id in direction_info and direction in direction_info[road_id]:
            connected_road_ids = direction_info[road_id][direction]
            connected_roads = roads[roads['id'].isin(connected_road_ids)]
            connected_geoms = connected_roads.geometry.tolist()
            
            # Calculate the direction angle
            angle = calculate_direction_angle(road_geom, connected_geoms)
        else:
            # If no direction info, use a default angle
            angle = 0
        
        # Create a more pointy triangle (longer and narrower)
        # The tip extends much further forward, while the base is narrower
        triangle = np.array([
            [triangle_size * 0.8, 0],                     # tip (further forward)
            [-triangle_size * 0.2, triangle_size * 0.25], # top back (narrower)
            [-triangle_size * 0.2, -triangle_size * 0.25] # bottom back (narrower)
        ])
        
        # Rotate triangle to point in the correct direction
        angle_rad = np.radians(angle)
        rotation_matrix = np.array([
            [np.cos(angle_rad), -np.sin(angle_rad)],
            [np.sin(angle_rad), np.cos(angle_rad)]
        ])
        rotated_triangle = triangle @ rotation_matrix.T
        
        # Translate to the center position
        rotated_triangle[:, 0] += center.x
        rotated_triangle[:, 1] += center.y
        
        # Plot the triangle with brighter cold blue color and thinner edge
        ax.fill(rotated_triangle[:, 0], rotated_triangle[:, 1], 
                color='#00A8E8', edgecolor='white', linewidth=1, 
                zorder=5, alpha=1.0)
    
    if draw_legend:
        """
        Add legend
        """
        legend_elements = []
        if pre_placement is not None:
            legend_elements.append(Line2D([0], [0], marker='o', color='w', 
                                        markerfacecolor='#9B59B6', markersize=10,
                                        label='Preconfigurations'))
        legend_elements.append(Line2D([0], [0], marker='^', color='w', 
                                    markerfacecolor='#00A8E8', markersize=10,
                                    label='Selections'))
        
        ax.legend(handles=legend_elements, loc=(0.05, 0.08), frameon=False, fontsize=18)
    
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


def ridge_plot_movement_over_sensor_counts_bk(plc_over_weeks, save_path=None, skip_sensor_counts=[]):
    import pandas as pd
    import numpy as np
    from ridgeplot import ridgeplot
    
    # Extract all sensor counts (assuming they're the same across weeks)
    first_week = next(iter(plc_over_weeks.values()))
    sensor_counts = sorted(first_week.keys())
    weeks = sorted(plc_over_weeks.keys())

    samples = []
    labels = []
    for sensor_count in sensor_counts:
        if sensor_count in skip_sensor_counts:
            continue
        changes_over_weeks = []
        
        for i in range(len(weeks) - 1):
            current_week = weeks[i]
            next_week = weeks[i + 1]
            
            # Convert lists to sets for comparison, directions are not considered
            current_set = set([i[0] for i in plc_over_weeks[current_week][sensor_count]])
            next_set = set([i[0] for i in plc_over_weeks[next_week][sensor_count]])
            
            # Count changes (sensor_count - overlap)
            overlap = len(current_set & next_set)
            num_changes = sensor_count - overlap
            changes_over_weeks.append(num_changes)
        
        # Add to samples and labels
        samples.append(np.array(changes_over_weeks))
        labels.append(f'{sensor_count} CCTVs')
    
    # Convert to numpy array for ridgeplot
    samples = np.array(samples)
    
    # Determine x-axis range
    all_changes = samples.flatten()
    x_min, x_max = all_changes.min(), all_changes.max()
    x_range = x_max - x_min
    kde_points = np.linspace(x_min - x_range * 0.1, x_max + x_range * 0.1, 500)
    
    # Create ridge plot
    fig = ridgeplot(
        samples=samples,
        bandwidth=x_range / 20,  # Adaptive bandwidth based on data range
        kde_points=kde_points,
        colorscale="Blues_r",  # Reversed: lighter for smaller sensor counts
        colormode="row-index",
        # opacity=0.9,
        labels=labels,
        spacing=3 / 9,
    )
    
    # Update layout
    fig.update_layout(
        height=max(560, len(sensor_counts) * 80),
        width=900,
        font=dict(family="Arial", size=19),
        plot_bgcolor="white",
        xaxis_gridcolor="rgba(0, 0, 0, 0.1)",
        yaxis_gridcolor="rgba(0, 0, 0, 0.1)",
        xaxis_title=dict(
            text="Number of Changes", 
            font=dict(family="Arial", size=19,)
        ),
        showlegend=False,
    )
    
    # Save or show
    if save_path is None:
        fig.show()
    else:
        fig.write_image(save_path, width=900, height=max(560, len(sensor_counts) * 80))

    return


def ridge_plot_movement_over_sensor_counts(plc_over_weeks_list, save_path=None, skip_sensor_counts=[]):
    import pandas as pd
    import numpy as np
    from ridgeplot import ridgeplot
    
    assert len(plc_over_weeks_list) == 2, "Currently supports exactly two datasets, as of the color scheme."

    # Extract all sensor counts (assuming they're the same across weeks and datasets)
    first_week = next(iter(plc_over_weeks_list[0].values()))
    sensor_counts = sorted(first_week.keys())
    
    # Prepare samples as list of lists of DataFrames
    samples = []
    
    for sensor_count in sensor_counts:
        if sensor_count in skip_sensor_counts:
            continue
        
        row_distributions = []
        
        # Process each plc_over_weeks dataset
        for idx, plc_data in enumerate(plc_over_weeks_list):
            weeks = sorted(plc_data.keys())
            changes_over_weeks = []
            
            for i in range(len(weeks) - 1):
                current_week = weeks[i]
                next_week = weeks[i + 1]
                
                # Convert lists to sets for comparison, directions are not considered
                current_set = set([i[0] for i in plc_data[current_week][sensor_count]])
                next_set = set([i[0] for i in plc_data[next_week][sensor_count]])
                
                # Count changes (sensor_count - overlap)
                overlap = len(current_set & next_set)
                num_changes = sensor_count - overlap
                changes_over_weeks.append(num_changes)
            
            # Convert to DataFrame (required by ridgeplot for multiple distributions per row)
            df = pd.DataFrame({f'changes_{idx}': changes_over_weeks})
            row_distributions.append(df[f'changes_{idx}'])

        samples.append(row_distributions)

    # Determine x-axis range from all data
    all_changes = []
    for row in samples:
        for dist in row:
            all_changes.extend(dist.values)
    all_changes = np.array(all_changes)
    x_min, x_max = all_changes.min(), all_changes.max()
    x_range = x_max - x_min
    kde_points = np.linspace(x_min - x_range * 0.1, x_max + x_range * 0.1, 500)

    fig = ridgeplot(
        samples=samples,
        bandwidth=x_range / 20,  # Adaptive bandwidth based on data range
        kde_points=kde_points,
        colorscale=['#79B4D9', '#D99B66'],
        colormode="trace-index-row-wise",
        labels=[["With preset", "Without preset"]] * (len(sensor_counts) - len(skip_sensor_counts)),
        row_labels=[f'{i} CCTVs' for i in sensor_counts if i not in skip_sensor_counts],
        spacing=3 / 9,
        opacity=0.8,
    )

    fig.update_layout(
        height=max(560, len(plc_over_weeks_list) * 80),
        width=900,
        font=dict(family="Arial", size=20),
        plot_bgcolor="white",
        xaxis_gridcolor="rgba(0, 0, 0, 0.1)",
        yaxis_gridcolor="rgba(0, 0, 0, 0.1)",
        xaxis_title=dict(
            text="Number of Changes", 
            font=dict(family="Arial", size=20)
        ),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="top",
            y=1.0,
            xanchor="center",
            x=0.5
        )
    )

    legend_names = set()
    for trace in fig.data:
        if trace.name in legend_names:
            trace.showlegend = False
        else:
            legend_names.add(trace.name)

    # Save or show
    if save_path is None:
        fig.show()
    else:
        fig.write_html(save_path.split('.png')[0] + '.html',)
        html_to_high_dpi_image(
            save_path.split('.png')[0] + '.html', save_path, width=900, height=max(560, len(plc_over_weeks_list) * 80)
            )
    return

from playwright.sync_api import sync_playwright
def html_to_high_dpi_image(html_path, output_path, width=900, height=560):
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(
            viewport={'width': width, 'height': height},
            device_scale_factor=3.125  # 300 DPI / 96 DPI ≈ 3.125
        )
        page.goto(f'file:///{html_path}')
        page.screenshot(path=output_path, full_page=True)
        browser.close()


def line_crime_count(df1, df2, df1Label, df2Label, resample, typeFilter=None, 
                     use_moving_avg=False, window=7, save_path=None):

    import matplotlib.dates as mdates
    
    # Set font to Arial and increase size BEFORE creating the plot
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size'] = 16
    
    # Filter by crime type if specified
    if typeFilter is not None:
        df1 = df1[df1.Type == typeFilter].copy()
        df2 = df2[df2.Type == typeFilter].copy()
    
    # Resample and count
    crimeCount_df1 = df1.resample(resample).count()
    crimeCount_df2 = df2.resample(resample).count()
    
    # Get the first column for counting
    count_col1 = crimeCount_df1.columns[0]
    count_col2 = crimeCount_df2.columns[0]
    
    # Prepare data for plotting (raw or moving average)
    if use_moving_avg:
        plot_data1 = crimeCount_df1[count_col1].rolling(window=window, center=True).mean()
        plot_data2 = crimeCount_df2[count_col2].rolling(window=window, center=True).mean()
    else:
        plot_data1 = crimeCount_df1[count_col1]
        plot_data2 = crimeCount_df2[count_col2]
    
    # Calculate averages (always from original data)
    avg_df1 = crimeCount_df1[count_col1].mean()
    avg_df2 = crimeCount_df2[count_col2].mean()
    
    # Custom colors (darker versions)
    colors = ['#D9534F', '#F5A623']
    
    # Create plot with shorter vertical size
    fig, ax = plt.subplots(figsize=(10, 4))
    
    # Plot crime count curves (raw or moving average) - only these appear in legend
    ax.plot(crimeCount_df1.index, plot_data1, 
            label='Before action', linewidth=2, alpha=0.8, color=colors[0])
    ax.plot(crimeCount_df2.index, plot_data2, 
            label='After action', linewidth=2, alpha=0.8, color=colors[1])
    
    # Plot average lines (based on original data) - no labels for legend
    ax.axhline(y=avg_df1, color=colors[0], linestyle='--', linewidth=1.5, alpha=0.7)
    ax.axhline(y=avg_df2, color=colors[1], linestyle='--', linewidth=1.5, alpha=0.7)
    
    # Get y-axis limits to calculate offset
    ylim = ax.get_ylim()
    y_range = ylim[1] - ylim[0]
    
    # Add annotations for average values at the left edge using axis coordinates
    # Annotate "before" line (further below the line)
    ax.text(0.2, avg_df1 - 0.05 * y_range, f'{avg_df1:.2f}',
            transform=ax.get_yaxis_transform(),
            color='black',
            fontsize=16,
            fontfamily='Arial',
            ha='left',
            va='top')
    
    # Annotate "after" line (above the line)
    ax.text(0.2, avg_df2, f'{avg_df2:.2f}',
            transform=ax.get_yaxis_transform(),
            color='black',
            fontsize=16,
            fontfamily='Arial',
            ha='left',
            va='bottom')
    
    # Y-axis label with line break and /day
    ax.set_ylabel('Crime count / day\n(moving averaged)', fontsize=16, fontfamily='Arial')
    
    # Set tick label font size to match reference (16)
    ax.tick_params(axis='both', labelsize=16)
    
    # Explicitly set tick label font to Arial
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontfamily('Arial')
    
    # Legend matching reference style
    legend = ax.legend(frameon=False, fontsize=16, loc='upper center', 
                       ncol=2, bbox_to_anchor=(0.5, 1.25),
                       handletextpad=0.3, columnspacing=1.5,
                       prop={'family': 'Arial', 'size': 16})
    
    # Turn off grid
    ax.grid(False)
    
    # Set x-axis limits to match data range
    all_dates = crimeCount_df1.index.union(crimeCount_df2.index)
    ax.set_xlim(all_dates.min(), all_dates.max())
    
    # Reduce number of x-axis ticks for dates
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    
    # Make box edge thicker (matching reference at 1.5)
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
    
    plt.tight_layout()
    
    # Print summary statistics
    print(f"\n{'='*50}")
    print(f"Summary Statistics:")
    print(f"{'='*50}")
    print(f"{df1Label}:")
    print(f"  Average: {avg_df1:.2f}")
    print(f"  Min: {crimeCount_df1[count_col1].min():.0f}")
    print(f"  Max: {crimeCount_df1[count_col1].max():.0f}")
    if use_moving_avg:
        print(f"  MA Min: {plot_data1.min():.2f}")
        print(f"  MA Max: {plot_data1.max():.2f}")
    
    print(f"\n{df2Label}:")
    print(f"  Average: {avg_df2:.2f}")
    print(f"  Min: {crimeCount_df2[count_col2].min():.0f}")
    print(f"  Max: {crimeCount_df2[count_col2].max():.0f}")
    if use_moving_avg:
        print(f"  MA Min: {plot_data2.min():.2f}")
        print(f"  MA Max: {plot_data2.max():.2f}")
    print(f"{'='*50}\n")
    
    # Save or show
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Figure saved to: {save_path}")
    else:
        plt.show()

    return


def map_plc_over_time(df_buffers, city_b, roads=None, save_path=None):
    import geopandas as gpd
    from matplotlib.colors import LinearSegmentedColormap, Normalize
    from matplotlib.cm import ScalarMappable
    
    # Set font to Arial
    plt.rcParams['font.family'] = 'Arial'
    
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Base map
    city_boundary = gpd.read_file(city_b)
    city_boundary = city_boundary[city_boundary['NAME'] == 'Warner Robins']
    city_boundary = city_boundary.to_crs(epsg=3857)
    city_boundary.plot(ax=ax, facecolor='lightgray', edgecolor='none')
    
    if roads is not None:
        roads = roads.to_crs(epsg=3857)
        roads_short = roads[roads.intersects(city_boundary.buffer(0.002).unary_union)]
        roads_short.plot(ax=ax, color='#3C535B', linewidth=1.5)
    
    # Plot buffer polygons on top
    df_buffers = df_buffers.to_crs(epsg=3857)
    max_freq = df_buffers['frequency'].max()
    alphas = df_buffers['frequency'] / max_freq
    df_buffers.plot(ax=ax, alpha=alphas, edgecolor='k', facecolor='#4A90B8', 
                   linewidth=0.5, zorder=4)
    
    # Create custom colormap matching the buffer color with varying alpha
    colors = [(1, 1, 1, 0), (0.29, 0.565, 0.722, 1)]  # Transparent to #4A90B8
    n_bins = 100
    cmap = LinearSegmentedColormap.from_list('custom_blue', colors, N=n_bins)
    
    # Add horizontal colorbar legend at the top
    norm = Normalize(vmin=0, vmax=max_freq)
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    
    # Create colorbar at the top using axes positioning
    cax = fig.add_axes([0.3, 0.86, 0.4, 0.02])  # [left, bottom, width, height]
    cbar = plt.colorbar(sm, cax=cax, orientation='horizontal')
    cbar.set_label('Placement frequency', labelpad=10, fontsize=14)
    cbar.ax.tick_params(labelsize=12)
    cbar.ax.xaxis.set_ticks_position('bottom')
    cbar.ax.xaxis.set_label_position('top')
    
    ax.set_aspect('equal')
    ax.axis('off')
    
    if save_path is None:
        plt.tight_layout()
        plt.show()
    else:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return


def line_crime_count_fitting(df, result, save_path=None):
    import matplotlib.dates as mdates
    import numpy as np
    
    # Get fitted values and residuals
    df['fitted'] = result.fittedvalues
    df['residuals'] = result.resid
    
    # Intervention point
    intervention_idx = df['intervention'].idxmax()
    intervention_date = df.index[df.index.get_loc(intervention_idx)]
    
    # Set font to Arial and increase size BEFORE creating the plot
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size'] = 16
    
    # Plot with shorter vertical size
    fig, ax = plt.subplots(figsize=(10, 4))
    
    # Use datetime index for x-axis with colors from the image
    ax.plot(df.index, df['y'], 'o', alpha=0.6, label='Actual', markersize=3, markeredgewidth=0, color='#5B8FA3')
    ax.plot(df.index, df['fitted'], '-', color='#D9534F', label='Fitted', linewidth=2)
    ax.axvline(intervention_date, color='#7B5C8F', linestyle='--', linewidth=1.5, label='Action date')
    
    # Calculate and plot trend lines for fitted values
    # Before intervention
    before_mask = df.index < intervention_date
    if before_mask.sum() > 1:
        x_before = np.arange(before_mask.sum())
        y_before = df.loc[before_mask, 'fitted'].values
        z_before = np.polyfit(x_before, y_before, 1)
        p_before = np.poly1d(z_before)
        ax.plot(df.index[before_mask], p_before(x_before), '-', color='grey', linewidth=1.5, label='Trend')
    
    # After intervention
    after_mask = df.index >= intervention_date
    if after_mask.sum() > 1:
        x_after = np.arange(after_mask.sum())
        y_after = df.loc[after_mask, 'fitted'].values
        z_after = np.polyfit(x_after, y_after, 1)
        p_after = np.poly1d(z_after)
        ax.plot(df.index[after_mask], p_after(x_after), '-', color='grey', linewidth=1.5)  # No label for second trend line
    
    ax.set_ylabel('Crime count / day', fontsize=16)
    legend = ax.legend(frameon=False, fontsize=16, loc='upper center', ncol=4, bbox_to_anchor=(0.5, 1.25), markerscale=3, columnspacing=1.0, handletextpad=0.3)
    ax.grid(False)
    
    # Set x-axis limits to match data range
    ax.set_xlim(df.index.min(), df.index.max())
    
    # Reduce number of x-axis ticks for dates
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    
    # Make box edge thicker
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
    
    plt.tight_layout()
    
    # Save or show
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def line_fitting_rediduals(df, result, save_path=None):
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import numpy as np
    from scipy import stats
    from statsmodels.stats.diagnostic import het_white
    
    # Set font to Arial globally
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size'] = 16
    
    # Get fitted values and residuals
    df['fitted'] = result.fittedvalues
    df['residuals'] = result.resid
    
    # Intervention point
    intervention_idx = df['intervention'].idxmax()
    intervention_date = df.index[df.index.get_loc(intervention_idx)]
    
    # Statistical tests for residuals
    resid = df['residuals'].values
    
    # Zero Mean test (t-test)
    t_stat, p_value_mean = stats.ttest_1samp(resid, 0)
    mean_resid = resid.mean()
    std_resid = resid.std()
    
    # Homoscedasticity tests
    # White test
    try:
        white_stat, white_pvalue, _, _ = het_white(resid, result.model.exog)
    except:
        white_stat, white_pvalue = np.nan, np.nan
    
    # Levene's test (pre vs post intervention)
    pre_resid = df[df['intervention'] == 0]['residuals']
    post_resid = df[df['intervention'] == 1]['residuals']
    levene_stat, levene_pvalue = stats.levene(pre_resid, post_resid)
    
    # Plot with reference styling
    fig, ax = plt.subplots(figsize=(10, 4))
    
    # Plot residuals using datetime index
    ax.plot(df.index, df['residuals'], 'o', alpha=0.6, markersize=3, 
            markeredgewidth=0, color='#5B8FA3', label='Residuals')
    
    # Add mean line
    ax.axhline(mean_resid, color='#D9534F', linestyle='-', linewidth=1.5, 
               label='Mean')
    
    # Zero line - same thickness and color as box edge (no label)
    ax.axhline(0, color='black', linestyle='-', linewidth=1.5)
    
    # Add ±2 std bands
    ax.axhline(mean_resid + 2*std_resid, color='#F5A623', linestyle=':', 
               linewidth=1.5, alpha=0.7, label='±2σ')
    ax.axhline(mean_resid - 2*std_resid, color='#F5A623', linestyle=':', 
               linewidth=1.5, alpha=0.7)
    
    # Intervention line
    ax.axvline(intervention_date, color='#7B5C8F', linestyle='--', 
               linewidth=1.5, label='Action date')
    
    # Labels and legend
    ax.set_ylabel('Residuals', fontsize=16, fontfamily='Arial')
    legend = ax.legend(frameon=False, fontsize=16, loc='upper center', ncol=4, 
                       bbox_to_anchor=(0.5, 1.3), markerscale=3, 
                       handletextpad=0.3, columnspacing=1.5,
                       prop={'family': 'Arial', 'size': 16})
    ax.grid(False)
    
    # Set tick label font size to match y-axis label
    ax.tick_params(axis='both', labelsize=16)
    
    # Explicitly set tick label font to Arial
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontfamily('Arial')
    
    # Set x-axis limits to match data range
    ax.set_xlim(df.index.min(), df.index.max())
    
    # Reduce number of x-axis ticks for dates
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    
    # Make box edge thicker
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
    
    plt.tight_layout()
    
    # Save or show
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
    
    # Print detailed test results
    print("\n" + "="*50)
    print("RESIDUAL DIAGNOSTIC TESTS")
    print("="*50)
    print(f"\n1. Zero Mean Test (One-sample t-test)")
    print(f"   H0: mean = 0")
    print(f"   Sample mean: {mean_resid:.6f}")
    print(f"   t-statistic: {t_stat:.4f}")
    print(f"   p-value: {p_value_mean:.4f}")
    print(f"   Result: {'PASS - Mean not significantly different from 0' if p_value_mean > 0.05 else 'FAIL - Mean significantly different from 0'}")
    
    print(f"\n2. Homoscedasticity Tests")
    if not np.isnan(white_pvalue):
        print(f"   a) White Test")
        print(f"      H0: Homoscedastic (constant variance)")
        print(f"      LM-statistic: {white_stat:.4f}")
        print(f"      p-value: {white_pvalue:.4f}")
        print(f"      Result: {'PASS - Constant variance' if white_pvalue > 0.05 else 'FAIL - Heteroscedasticity detected'}")
        print(f"\n   b) Levene Test (Pre vs Post Intervention)")
    else:
        print(f"   a) Levene Test (Pre vs Post Intervention)")
    
    print(f"      H0: Equal variances")
    print(f"      Statistic: {levene_stat:.4f}")
    print(f"      p-value: {levene_pvalue:.4f}")
    print(f"      Std (pre): {pre_resid.std():.4f}, Std (post): {post_resid.std():.4f}")
    print(f"      Result: {'PASS - Equal variances' if levene_pvalue > 0.05 else 'FAIL - Unequal variances'}")
    print("="*50)


def dist_fitting_residuals(df, result, save_path=None):
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy import stats
    
    # Set font to Arial globally
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size'] = 16
    
    # Get fitted values and residuals
    df['fitted'] = result.fittedvalues
    df['residuals'] = result.resid
    
    # Calculate statistics
    resid = df['residuals'].values
    mean_resid = resid.mean()
    std_resid = resid.std()
    
    # Normality tests
    shapiro_stat, shapiro_pvalue = stats.shapiro(resid)
    ks_stat, ks_pvalue = stats.kstest(resid, 'norm', args=(mean_resid, std_resid))
    jb_stat, jb_pvalue = stats.jarque_bera(resid)
    
    # Plot with matching style
    fig, ax = plt.subplots(figsize=(10, 4))
    
    # Histogram with matching color scheme
    ax.hist(df['residuals'], bins=30, density=True, alpha=0.6, 
            edgecolor='none', color='#5B8FA3', label='Residuals')
    
    # Add normal curve overlay - orange color
    x = np.linspace(df['residuals'].min(), df['residuals'].max(), 100)
    ax.plot(x, stats.norm.pdf(x, mean_resid, std_resid), 
            color='#F5A623', linewidth=2, label='Density line')
    
    # Mean line - red color
    ax.axvline(mean_resid, color='#D9534F', linestyle='--', 
               linewidth=1.5, label='Mean')
    
    # Labels and legend
    ax.set_xlabel('Residuals', fontsize=16, fontfamily='Arial')
    ax.set_ylabel('Density', fontsize=16, fontfamily='Arial')
    legend = ax.legend(frameon=False, fontsize=16, loc='upper center', ncol=3,
                       bbox_to_anchor=(0.5, 1.30), markerscale=3,
                       handletextpad=0.3, columnspacing=1.5,
                       prop={'family': 'Arial', 'size': 16})
    ax.grid(False)
    
    # Set tick label font size to match y-axis label
    ax.tick_params(axis='both', labelsize=16)
    
    # Explicitly set tick label font to Arial
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontfamily('Arial')
    
    # Make box edge thicker
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
    
    plt.tight_layout()
    
    # Save or show
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
    
    # Print detailed test results
    print("\n" + "="*50)
    print("NORMALITY DIAGNOSTIC TESTS")
    print("="*50)
    print(f"\nResidual Statistics:")
    print(f"   Mean: {mean_resid:.6f}")
    print(f"   Std Dev: {std_resid:.6f}")
    print(f"   Skewness: {stats.skew(resid):.6f}")
    print(f"   Kurtosis: {stats.kurtosis(resid):.6f}")
    
    print(f"\n1. Shapiro-Wilk Test")
    print(f"   H0: Data is normally distributed")
    print(f"   W-statistic: {shapiro_stat:.4f}")
    print(f"   p-value: {shapiro_pvalue:.4f}")
    print(f"   Result: {'PASS - Normally distributed' if shapiro_pvalue > 0.05 else 'FAIL - Not normally distributed'}")
    
    print(f"\n2. Kolmogorov-Smirnov Test")
    print(f"   H0: Data follows normal distribution")
    print(f"   KS-statistic: {ks_stat:.4f}")
    print(f"   p-value: {ks_pvalue:.4f}")
    print(f"   Result: {'PASS - Follows normal distribution' if ks_pvalue > 0.05 else 'FAIL - Does not follow normal distribution'}")
    
    print(f"\n3. Jarque-Bera Test")
    print(f"   H0: Data has skewness and kurtosis matching normal distribution")
    print(f"   JB-statistic: {jb_stat:.4f}")
    print(f"   p-value: {jb_pvalue:.4f}")
    print(f"   Result: {'PASS - Matches normal distribution' if jb_pvalue > 0.05 else 'FAIL - Does not match normal distribution'}")
    print("="*50)
