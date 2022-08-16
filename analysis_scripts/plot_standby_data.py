from bureaucrat.SmarterBureaucrat import NamedTaskBureaucrat # https://github.com/SengerM/bureaucrat
from huge_dataframe.SQLiteDataFrame import load_whole_dataframe # https://github.com/SengerM/huge_dataframe
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path

def do_IV_vs_when_plot(measured_data_df):
	IV_vs_when_plot = make_subplots(cols=1, rows=2, shared_xaxes=True, vertical_spacing=0.02)
	IV_vs_when_plot.update_xaxes(title_text="When", row=2, col=1)
	colors_iterator = iter(px.colors.qualitative.Plotly)
	for device_name in sorted(set(measured_data_df.index)):
		current_color = next(colors_iterator)
		for row_idx,var_name in enumerate(['Bias voltage (V)','Bias current (A)']):
			IV_vs_when_plot.update_yaxes(title_text=var_name, row=row_idx+1, col=1)
			IV_vs_when_plot.add_trace(
				go.Scattergl(
					x = measured_data_df.loc[device_name,'When'],
					y = measured_data_df.loc[device_name,var_name],
					mode = 'lines',
					name = device_name,
					legendgroup = device_name,
					showlegend = row_idx == 0,
					line_color = current_color,
					error_y=dict(
						type = 'data',
						array = measured_data_df.loc[device_name,f'{var_name} std'],
						visible = True,
						width = 1,
						thickness = .8,
					),
				),
				row = row_idx+1,
				col = 1,
			)
	return IV_vs_when_plot

Norbert = NamedTaskBureaucrat(
	Path('/home/sengerm/measurements_data')/'20220816000000_Robocold_setup_test_run',
	task_name = 'plot_standby_data',
	_locals = locals(),
)

with Norbert.do_your_magic():
	measured_data_df = load_whole_dataframe(Norbert.path_to_output_directory_of_task_named('monitor_standby_conditions')/'measured_data.sqlite')
	measured_data_df['Bias voltage (V)'] *= -1
	
	fig = do_IV_vs_when_plot(measured_data_df)
	fig.write_html(
		str(Norbert.path_to_default_output_directory/'bias_voltage_and_current.html'),
		include_plotlyjs = 'cdn',
	)
