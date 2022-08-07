from bureaucrat.SmarterBureaucrat import SmarterBureaucrat # https://github.com/SengerM/bureaucrat
from pathlib import Path
import pandas
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from huge_dataframe.SQLiteDataFrame import load_whole_dataframe # https://github.com/SengerM/huge_dataframe

COLOR_DISCRETE_MAP = {
	False: '#ff5c5c',
	True: '#27c200',
}

def script_core(path_to_measurement_base_directory: Path):
	John = SmarterBureaucrat(
		path_to_measurement_base_directory,
		_locals = locals(),
	)
	
	John.check_required_scripts_were_run_before(['beta_scan.py','clean_beta_scan.py'])
	
	parsed_from_waveforms_df = load_whole_dataframe(John.path_to_output_directory_of_script_named('beta_scan.py')/Path('parsed_from_waveforms.sqlite'))
	clean_triggers_df = pandas.read_feather(John.path_to_output_directory_of_script_named('clean_beta_scan.py')/Path('result.fd')).set_index('n_trigger')
	
	with John.do_your_magic():
		df = parsed_from_waveforms_df.merge(right=clean_triggers_df, left_index=True, right_index=True)
		df = df.reset_index().drop({'n_waveform'}, axis=1).sort_values('signal_name')
		path_to_save_plots = John.path_to_default_output_directory/'distributions'
		path_to_save_plots.mkdir(exist_ok = True)
		for col in df.columns:
			if col in {'signal_name','n_trigger','accepted'}:
				continue
			fig = px.histogram(
				df,
				title = f'{col} histogram<br><sup>Measurement: {John.measurement_name}</sup>',
				x = col,
				facet_row = 'signal_name',
				color = 'accepted',
				color_discrete_map = COLOR_DISCRETE_MAP,
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
				color = 'accepted',
				color_discrete_map = COLOR_DISCRETE_MAP,
			)
			fig.write_html(
				str(path_to_save_plots/Path(f'{col} ecdf.html')),
				include_plotlyjs = 'cdn',
			)
			
		columns_for_scatter_matrix_plot = set(df.columns) 
		columns_for_scatter_matrix_plot -= {'n_trigger','signal_name','accepted'} 
		columns_for_scatter_matrix_plot -= {f't_{i} (s)' for i in [10,20,30,40,60,70,80,90]}
		columns_for_scatter_matrix_plot -= {f'Time over {i}% (s)' for i in [10,30,40,50,60,70,80,90]}
		fig = px.scatter_matrix(
			df,
			dimensions = sorted(columns_for_scatter_matrix_plot),
			title = f'Scatter matrix plot<br><sup>Measurement: {John.measurement_name}</sup>',
			symbol = 'signal_name',
			color = 'accepted',
			hover_data = ['n_trigger'],
			color_discrete_map = COLOR_DISCRETE_MAP,
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
			str(John.path_to_default_output_directory/Path('scatter matrix plot.html')),
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
