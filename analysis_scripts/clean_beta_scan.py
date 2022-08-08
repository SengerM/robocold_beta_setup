from bureaucrat.SmarterBureaucrat import NamedTaskBureaucrat # https://github.com/SengerM/bureaucrat
from pathlib import Path
import pandas
import shutil
from huge_dataframe.SQLiteDataFrame import load_whole_dataframe # https://github.com/SengerM/huge_dataframe
import plotly.graph_objects as go
import plotly.express as px
import numpy as np

def apply_cuts(data_df, cuts_df):
	"""
	Given a dataframe `cuts_df` with one cut per row, e.g.
	```
		  signal_name       variable cut_type  cut_value
					  DUT  Amplitude (V)    lower        0.1
		reference_trigger            SNR    lower       20.0

	```
	this function returns a series with the index `n_trigger` and the value
	either `True` or `False` stating if such trigger satisfies ALL the
	cuts at the same time. For example using the previous example a
	trigger with charge 3e-12 and t_50 6.45e-8 will be `True` but if any
	of the variables in any of the channels is outside the range, it will
	be `False`.
	"""
	
	set_of_signal_names_on_which_to_apply_cuts = set(cuts_df['signal_name'])
	set_of_measured_signals = set(data_df.index.get_level_values('signal_name'))
	if not set_of_signal_names_on_which_to_apply_cuts.issubset(set_of_measured_signals):
		raise ValueError(f'One (or more) `signal_name` on which you want to apply cuts is not present in the measurement data. You want to apply cuts on signal_names = {set_of_signal_names_on_which_to_apply_cuts} while the measured signal_names are {set_of_measured_signals}.')
	
	data_df = data_df.reset_index(drop=False).pivot(
		index = 'n_trigger',
		columns = 'signal_name',
		values = list(set(data_df.columns) - {'signal_name'}),
	)
	triggers_accepted_df = pandas.DataFrame({'is_background': True}, index=data_df.index)
	for idx, cut_row in cuts_df.iterrows():
		if cut_row['cut_type'] == 'lower':
			triggers_accepted_df['is_background'] &= data_df[(cut_row['variable'],cut_row['signal_name'])] < cut_row['cut_value']
		elif cut_row['cut_type'] == 'higher':
			triggers_accepted_df['is_background'] &= data_df[(cut_row['variable'],cut_row['signal_name'])] > cut_row['cut_value']
		else:
			raise ValueError('Received a cut of type `cut_type={}`, dont know that that is...'.format(cut_row['cut_type']))
	return triggers_accepted_df

def clean_beta_scan(path_to_measurement_base_directory:Path, path_to_cuts_file:Path=None)->Path:
	"""Clean the events from a beta scan, i.e. apply cuts to reject/accept 
	the events. The output is a file assigning an "is_background" `True`
	or `False` label to each trigger.
	
	Arguments
	---------
	path_to_measurement_base_directory: Path
		Path to the directory of the measurement you want to clean.
	path_to_cuts_file: Path, optional
		Path to a CSV file specifying the cuts, an example of such file
		is 
		```
			  signal_name       variable cut_type  cut_value
					  DUT  Amplitude (V)    lower        0.1
		reference_trigger            SNR    lower       20.0
		```
		If nothing is passed, a file named `cuts.csv` will try to be found
		in the measurement's base directory.
	"""
	John = NamedTaskBureaucrat(
		path_to_measurement_base_directory,
		task_name = 'clean_beta_scan',
		_locals = locals(),
	)
	
	John.check_required_scripts_were_run_before('beta_scan.py')
	
	if path_to_cuts_file is None:
		path_to_cuts_file = John.path_to_measurement_base_directory/Path('cuts.csv')
	elif not isinstance(path_to_cuts_file, Path):
		raise TypeError(f'`path_to_cuts_file` must be an instance of {Path}, received object of type {type(path_to_cuts_file)}.')
	cuts_df = pandas.read_csv(path_to_cuts_file)
	data_df = load_whole_dataframe(John.path_to_output_directory_of_script_named('beta_scan.py')/Path('parsed_from_waveforms.sqlite'))
	
	with John.do_your_magic():
		cuts_df.to_csv(John.path_to_default_output_directory/Path(f'cuts.backup.csv'), index=False) # Create a backup.
		filtered_triggers_df = apply_cuts(data_df, cuts_df)
		filtered_triggers_df.reset_index().to_feather(John.path_to_default_output_directory/Path('result.fd'))

def plot_beta_scan_after_cleaning(path_to_measurement_base_directory: Path):
	
	COLOR_DISCRETE_MAP = {
		True: '#ff5c5c',
		False: '#27c200',
	}
	
	John = NamedTaskBureaucrat(
		path_to_measurement_base_directory,
		task_name = 'plot_beta_scan_after_cleaning',
		_locals = locals(),
	)
	
	John.check_required_tasks_were_run_before(['beta_scan','clean_beta_scan'])
	
	parsed_from_waveforms_df = load_whole_dataframe(John.path_to_output_directory_of_script_named('beta_scan.py')/Path('parsed_from_waveforms.sqlite'))
	clean_triggers_df = pandas.read_feather(John.path_to_output_directory_of_script_named('clean_beta_scan.py')/Path('result.fd')).set_index('n_trigger')
	
	with John.do_your_magic():
		df = parsed_from_waveforms_df.merge(right=clean_triggers_df, left_index=True, right_index=True)
		df = df.reset_index().drop({'n_waveform'}, axis=1).sort_values('signal_name')
		path_to_save_plots = John.path_to_default_output_directory/'distributions'
		path_to_save_plots.mkdir(exist_ok = True)
		for col in df.columns:
			if col in {'signal_name','n_trigger','is_background'}:
				continue
			fig = px.histogram(
				df,
				title = f'{col} histogram<br><sup>Measurement: {John.measurement_name}</sup>',
				x = col,
				facet_row = 'signal_name',
				color = 'is_background',
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
				color = 'is_background',
				color_discrete_map = COLOR_DISCRETE_MAP,
			)
			fig.write_html(
				str(path_to_save_plots/Path(f'{col} ecdf.html')),
				include_plotlyjs = 'cdn',
			)
			
		columns_for_scatter_matrix_plot = set(df.columns) 
		columns_for_scatter_matrix_plot -= {'n_trigger','signal_name','is_background'} 
		columns_for_scatter_matrix_plot -= {f't_{i} (s)' for i in [10,20,30,40,60,70,80,90]}
		columns_for_scatter_matrix_plot -= {f'Time over {i}% (s)' for i in [10,30,40,50,60,70,80,90]}
		fig = px.scatter_matrix(
			df,
			dimensions = sorted(columns_for_scatter_matrix_plot),
			title = f'Scatter matrix plot<br><sup>Measurement: {John.measurement_name}</sup>',
			symbol = 'signal_name',
			color = 'is_background',
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

	parser = argparse.ArgumentParser(description='Cleans a beta scan according to some criterion.')
	parser.add_argument('--dir',
		metavar = 'path',
		help = 'Path to the base measurement directory.',
		required = True,
		dest = 'directory',
		type = str,
	)

	args = parser.parse_args()
	clean_beta_scan(Path(args.directory))
	plot_beta_scan_after_cleaning(Path(args.directory))