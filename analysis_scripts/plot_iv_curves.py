import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from the_bureaucrat.bureaucrats import RunBureaucrat # https://github.com/SengerM/the_bureaucrat
from pathlib import Path
from huge_dataframe.SQLiteDataFrame import load_whole_dataframe # https://github.com/SengerM/huge_dataframe
import pandas
from summarize_parameters import read_summarized_data
import grafica.plotly_utils.utils # https://github.com/SengerM/grafica

def plot_iv_curve(bureaucrat:RunBureaucrat):
	raise NotImplementedError()
	# Finished, now do a plot ---
	measured_data_df = load_whole_dataframe(John.path_to_directory_of_task('measure_iv_curve')/Path('measured_data.sqlite'))
	measured_data_df.reset_index(inplace=True)
	measured_data_df['Bias voltage (V)'] *= -1
	fig = px.line(
		measured_data_df.groupby(by='n_voltage').mean(), 
		x = "Bias voltage (V)", 
		y = "Bias current (A)",
		title = f'IV curve<br><sup>Measurement: {John.measurement_name}</sup>',
		markers = True,
	)
	fig.write_html(str(John.path_to_default_output_directory/Path('iv_curve_lin_scale.html')), include_plotlyjs='cdn')
	
	fig.update_yaxes(type="log")
	fig.write_html(str(John.path_to_default_output_directory/Path('iv_curve_log_scale.html')), include_plotlyjs='cdn')
	
	fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.02)
	fig.update_layout(title =  f'Current/Voltage vs time with beta source<br><sup>Measurement: {John.measurement_name}</sup>')
	for row_minus_one, variable in enumerate(['Bias current (A)', 'Bias voltage (V)']):
		fig.add_trace(
			go.Scatter(
				x = measured_data_df['When'],
				y = measured_data_df[variable],
				name = variable,
				mode = 'lines+markers',
			),
			row = row_minus_one + 1,
			col = 1
		)
		fig.update_yaxes(title_text=variable, row=row_minus_one+1, col=1)
		
		if variable == 'Bias voltage (V)':
			fig.add_trace(
				go.Scatter(
					x = measured_data_df['When'],
					y = measured_data_df['Set voltage (V)'],
					name = 'Set voltage (V)',
					mode = 'lines+markers',
				),
				row = row_minus_one + 1,
				col = 1
			)
			fig.update_yaxes(title_text='Voltage (V)', row=row_minus_one+1, col=1)
	fig.update_xaxes(title_text='When', row=row_minus_one+1, col=1)
	fig.write_html(str(John.path_to_default_output_directory/Path(f'iv_vs_time.html')), include_plotlyjs='cdn')

	return John.path_to_measurement_base_directory

def plot_IV_curves_all_together(bureaucrat:RunBureaucrat):
	"""Do a plot of the IV curves measured previously all together."""
	Richard = bureaucrat
	
	Richard.check_these_tasks_were_run_successfully('measure_iv_curves_on_multiple_slots')
	
	with Richard.handle_task('plot_IV_curves_all_together') as plot_IV_curves_all_together_task_handler:
		measured_data_list = []
		for subrun in Richard.list_subruns_of_task('measure_iv_curves_on_multiple_slots'):
			subrun.check_these_tasks_were_run_successfully('measure_iv_curve')
			measured_data_list.append(
				load_whole_dataframe(subrun.path_to_directory_of_task('measure_iv_curve')/'measured_data.sqlite').reset_index(),
			)
		measured_data_df = pandas.concat(measured_data_list, ignore_index=True)
		measured_data_df['Bias voltage (V)'] *= -1
		grouped_thing = measured_data_df.groupby(['device_name','n_voltage'])
		averaged_data_df = grouped_thing.mean()
		standard_deviation_data_df = grouped_thing.std()
		for col in {'Bias voltage (V)','Bias current (A)'}:
			averaged_data_df[f'{col} error'] = standard_deviation_data_df[col]
		averaged_data_df.reset_index(inplace=True)
		for log_y in {True,False}:
			fig = px.line(
				averaged_data_df.sort_values(['device_name','n_voltage']).reset_index(),
				x = 'Bias voltage (V)',
				y = 'Bias current (A)',
				error_y = 'Bias current (A) error',
				error_x = 'Bias voltage (V) error',
				color = 'device_name',
				markers = True,
				title = f'IV curves<br><sup>{Richard.run_name}</sup>',
				hover_data = ['n_voltage'],
				log_y = log_y,
			)
			fig.update_traces(
				error_y = dict(
					width = 1,
					thickness = .8,
				)
			)
			fig.write_html(
				str(plot_IV_curves_all_together_task_handler.path_to_directory_of_my_task/Path(f'iv_curves_{"lin" if log_y==False else "log"}.html')),
				include_plotlyjs = 'cdn',
			)

def IV_curve_from_beta_scan_data(bureaucrat:RunBureaucrat):
	with bureaucrat.handle_task('IV_curve_from_beta_scan_data') as employee:
		bureaucrat.check_these_tasks_were_run_successfully(['beta_scan_sweeping_bias_voltage'])
		summary = read_summarized_data(bureaucrat)
		summary = summary[['Bias voltage (V)','Bias current (A)','Temperature (°C)','Humidity (%RH)']]
		summary.columns = [f'{col[0]} {col[1]}' for col in summary.columns]
		summary.to_pickle(employee.path_to_directory_of_my_task/'iv_data.pickle')
		
		fig = px.line(
			data_frame = summary.reset_index(drop=False).sort_values(['device_name','Bias voltage (V) mean']),
			x = 'Bias voltage (V) mean',
			y = 'Bias current (A) mean',
			error_x = 'Bias voltage (V) std',
			error_y = 'Bias current (A) std',
			color = 'device_name',
			title = f'IV curve from beta scan data<br><sup>{bureaucrat.run_name}</sup>',
			markers = True,
			hover_data = ['Temperature (°C) mean','Humidity (%RH) mean','run_name'],
		)
		fig.update_layout(xaxis = dict(autorange = "reversed"))
		fig.write_html(
			str(employee.path_to_directory_of_my_task/'iv_curve.html'),
			include_plotlyjs = 'cdn',
		)

if __name__ == '__main__':
	import argparse
	
	grafica.plotly_utils.utils.set_my_template_as_default()

	parser = argparse.ArgumentParser()
	parser.add_argument('--dir',
		metavar = 'path',
		help = 'Path to the base measurement directory.',
		required = True,
		dest = 'directory',
		type = str,
	)

	args = parser.parse_args()
	plot_IV_curves_all_together(RunBureaucrat(Path(args.directory)))
