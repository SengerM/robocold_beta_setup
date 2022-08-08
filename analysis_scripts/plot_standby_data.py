from huge_dataframe.SQLiteDataFrame import load_whole_dataframe # https://github.com/SengerM/huge_dataframe
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from monitor_standby_conditions import PATH_TO_DIRECTORY_WHERE_TO_STORE_DATA as path_to_directory_with_monitored_data

def do_IV_vs_when_plot(measured_data_df):
	IV_vs_when_plot = make_subplots(cols=1, rows=2, shared_xaxes=True, vertical_spacing=0.02)
	IV_vs_when_plot.update_xaxes(title_text="When", row=2, col=1)
	colors_iterator = iter(px.colors.qualitative.Plotly)
	for device_name in sorted(set(measured_data_df.index)):
		current_color = next(colors_iterator)
		for row_idx,var_name in enumerate(['Bias voltage (V)','Bias current (A)']):
			IV_vs_when_plot.update_yaxes(title_text=var_name, row=row_idx+1, col=1)
			IV_vs_when_plot.add_trace(
				go.Scatter(
					x = measured_data_df.loc[device_name,'When'],
					y = measured_data_df.loc[device_name,var_name],
					mode = 'lines+markers',
					name = device_name,
					legendgroup = device_name,
					showlegend = row_idx == 0,
					line_color = current_color,
					error_y=dict(
						type = 'data',
						array = measured_data_df.loc[device_name,f'{var_name} std'],
						visible = True,
					),
				),
				row = row_idx+1,
				col = 1,
			)
	return IV_vs_when_plot

measured_data_df = load_whole_dataframe(path_to_directory_with_monitored_data/'measured_data.sqlite')
measured_data_df['Bias voltage (V)'] *= -1

print(measured_data_df)

fig = do_IV_vs_when_plot(measured_data_df)
fig.write_html(
	str(path_to_directory_with_monitored_data/'bias current.html'),
	include_plotlyjs = 'cdn',
)