from bureaucrat.SmarterBureaucrat import SmarterBureaucrat # https://github.com/SengerM/bureaucrat
from pathlib import Path
import pandas
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from huge_dataframe.SQLiteDataFrame import load_whole_dataframe # https://github.com/SengerM/huge_dataframe

def script_core(directory: Path):
	John = SmarterBureaucrat(
		directory,
		_locals = locals(),
	)
	
	John.check_required_scripts_were_run_before('beta_scan.py')
	
	measured_stuff_df = load_whole_dataframe(John.path_to_output_directory_of_script_named('beta_scan.py')/Path('measured_stuff.sqlite'))
	parsed_from_waveforms_df = load_whole_dataframe(John.path_to_output_directory_of_script_named('beta_scan.py')/Path('parsed_from_waveforms.sqlite'))
	
	with John.do_your_magic():
		df = measured_stuff_df.sort_values('When').reset_index()
		path_to_save_plots = John.path_to_default_output_directory/Path('measured_stuff_vs_time')
		path_to_save_plots.mkdir()
		for col in measured_stuff_df.columns:
			if col in {'device_name','When','n_trigger'}:
				continue
			fig = px.line(
				df,
				title = f'{col} vs time<br><sup>Measurement: {John.measurement_name}</sup>',
				x = 'When',
				y = col,
				color = 'device_name',
				markers = True,
			)
			fig.write_html(
				str(path_to_save_plots/Path(f'{col} vs time.html')),
				include_plotlyjs = 'cdn',
			)
		
		df = parsed_from_waveforms_df.reset_index().drop({'n_waveform'}, axis=1).sort_values('signal_name')
		path_to_save_plots = John.path_to_default_output_directory/Path('parsed_from_waveforms')
		path_to_save_plots.mkdir()
		for col in df.columns:
			if col in {'signal_name','n_trigger'}:
				continue
			fig = px.histogram(
				df,
				title = f'{col} histogram<br><sup>Measurement: {John.measurement_name}</sup>',
				x = col,
				facet_row = 'signal_name',
			)
			fig.write_html(
				str(path_to_save_plots/Path(f'{col} histogram.html')),
				include_plotlyjs = 'cdn',
			)
			
			fig = px.ecdf(
				df,
				title = f'{col} ECDF<br><sup>Measurement: {John.measurement_name}</sup>',
				x = col,
				facet_row = 'signal_name',
			)
			fig.write_html(
				str(path_to_save_plots/Path(f'{col} ecdf.html')),
				include_plotlyjs = 'cdn',
			)
			
			columns_for_scatter_matrix_plot = set(df.columns) - {'n_trigger','signal_name'}
			fig = px.scatter_matrix(
				df,
				dimensions = sorted(columns_for_scatter_matrix_plot),
				title = f'Scatter matrix plot<br><sup>Measurement: {John.measurement_name}</sup>',
				color = 'signal_name',
				hover_data = ['n_trigger'],
			)
			fig.update_traces(diagonal_visible=False, showupperhalf=False, marker = {'size': 3})
			for k in range(len(fig.data)):
				fig.data[k].update(
					selected = dict(
						marker = dict(
							opacity = 1,
							color = 'black',
						)
					),
				)
			fig.write_html(
				str(path_to_save_plots/Path('scatter matrix plot.html')),
				include_plotlyjs = 'cdn',
			)
		
########################################################################

if __name__ == '__main__':
	import argparse

	parser = argparse.ArgumentParser(description='Makes plots with the distributions of the quantities parsed by the script "parse_raw_data_of_single_beta_scan.py".')
	parser.add_argument('--dir',
		metavar = 'path', 
		help = 'Path to the base measurement directory.',
		required = True,
		dest = 'directory',
		type = str,
	)

	args = parser.parse_args()
	script_core(Path(args.directory))
