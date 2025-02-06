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
