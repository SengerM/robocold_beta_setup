from the_bureaucrat.bureaucrats import RunBureaucrat # https://github.com/SengerM/the_bureaucrat
from pathlib import Path
import pandas
import shutil
from huge_dataframe.SQLiteDataFrame import load_whole_dataframe # https://github.com/SengerM/huge_dataframe
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from plot_beta_scan import draw_histogram_and_langauss_fit
import dominate # https://github.com/Knio/dominate
import grafica.plotly_utils.utils
from scipy.optimize import minimize

def create_cuts_file_template(bureaucrat:RunBureaucrat):
	"""Creates a file `cuts.csv` in the run directory with the template
	that is most commonly used.
	"""
	if bureaucrat.was_task_run_successfully('beta_scan_sweeping_bias_voltage'):
		path_to_cuts_file = bureaucrat.path_to_run_directory/'cuts.csv'
		if path_to_cuts_file.is_file():
			raise RuntimeError(f'A file with the cuts already exists.')
		with open(path_to_cuts_file, 'w') as ofile:
			print('run_name,signal_name,variable,cut_type,cut_value', file=ofile)
			for subrun in bureaucrat.list_subruns_of_task('beta_scan_sweeping_bias_voltage'):
				print(f'{subrun.run_name},DUT,Amplitude (V),lower,NaN', file=ofile)
	elif bureaucrat.was_task_run_successfully('beta_scan'):
		raise NotImplementedError(f'Not yet implemented for `beta_scan`.')
	else:
		raise RuntimeError(f'Dont know how to create cuts file template for run {repr(bureaucrat.run_name)}.')

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
			triggers_accepted_df['is_background'] &= data_df[(cut_row['variable'],cut_row['signal_name'])] > cut_row['cut_value']
		elif cut_row['cut_type'] == 'higher':
			triggers_accepted_df['is_background'] &= data_df[(cut_row['variable'],cut_row['signal_name'])] < cut_row['cut_value']
		else:
			raise ValueError('Received a cut of type `cut_type={}`, dont know that that is...'.format(cut_row['cut_type']))
	triggers_accepted_df['is_background'] = ~triggers_accepted_df['is_background']
	return triggers_accepted_df

def clean_beta_scan(bureaucrat:RunBureaucrat, path_to_cuts_file:Path=None)->Path:
	"""Clean the events from a beta scan, i.e. apply cuts to reject/accept 
	the events. The output is a file assigning an "is_background" `True`
	or `False` label to each trigger.
	
	Arguments
	---------
	bureaucrat: RunBureaucrat
		The bureaucrat to handle this run.
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
	John = bureaucrat
	
	John.check_these_tasks_were_run_successfully('beta_scan')
	
	if path_to_cuts_file is None:
		path_to_cuts_file = John.path_to_run_directory/Path('cuts.csv')
	elif not isinstance(path_to_cuts_file, Path):
		raise TypeError(f'`path_to_cuts_file` must be an instance of {Path}, received object of type {type(path_to_cuts_file)}.')
	
	with John.handle_task('clean_beta_scan') as task:
		cuts_df = pandas.read_csv(path_to_cuts_file)
		REQUIRED_COLUMNS = {'signal_name','variable','cut_type','cut_value'}
		if set.intersection(set(cuts_df.columns),REQUIRED_COLUMNS) != set(REQUIRED_COLUMNS):
			raise ValueError(f'The file with the cuts {path_to_cuts_file} must have the following columns: {REQUIRED_COLUMNS}, but it has columns {set(cuts_df.columns)}.')
		data_df = load_whole_dataframe(John.path_to_directory_of_task('beta_scan')/'parsed_from_waveforms.sqlite')
		cuts_df.to_csv(task.path_to_directory_of_my_task/Path(f'cuts.backup.csv'), index=False) # Create a backup.
		filtered_triggers_df = apply_cuts(data_df, cuts_df)
		filtered_triggers_df.reset_index().to_feather(task.path_to_directory_of_my_task/Path('result.fd'))

def clean_beta_scan_sweeping_bias_voltage(bureaucrat:RunBureaucrat, path_to_cuts_file:Path=None):
	"""Clean all sub- beta scans at once."""
	Eriberto = bureaucrat
	Eriberto.check_these_tasks_were_run_successfully('beta_scan_sweeping_bias_voltage')
	
	if path_to_cuts_file is None: # Try to locate it within the measurement's base directory.
		path_to_cuts_file = Eriberto.path_to_run_directory/'cuts.csv'
	if not path_to_cuts_file.is_file():
		raise FileNotFoundError(f'Cannot find file with the cuts in {path_to_cuts_file}.')
	cuts_df = pandas.read_csv(path_to_cuts_file)
	REQUIRED_COLUMNS = {'run_name','signal_name','variable','cut_type','cut_value'}
	if set.intersection(set(cuts_df.columns),REQUIRED_COLUMNS) != set(REQUIRED_COLUMNS):
		raise ValueError(f'The file with the cuts {path_to_cuts_file} must have the following columns: {REQUIRED_COLUMNS}, but it has columns {set(cuts_df.columns)}.')
	cuts_df.set_index('run_name',inplace=True)
	for Quique in Eriberto.list_subruns_of_task('beta_scan_sweeping_bias_voltage'):
		this_run_cuts_df = cuts_df.query(f'run_name=={repr(Quique.run_name)}')
		if len(this_run_cuts_df) == 0:
			raise RuntimeError(f'No cuts were found when cleaning beta scan for run {Eriberto.run_name} located in {Eriberto.path_to_run_directory}.')
		this_run_cuts_df.to_csv(Quique.path_to_temporary_directory/'cuts.cvs',index=False)
		clean_beta_scan(Quique, Quique.path_to_temporary_directory/'cuts.cvs')

def clean_beta_scan_plots(bureaucrat:RunBureaucrat, scatter_plot:bool=False, langauss_plots:bool=True, distributions:bool=False):
	COLOR_DISCRETE_MAP = {
		True: '#ff5c5c',
		False: '#27c200',
	}
	
	John = bureaucrat
	
	John.check_these_tasks_were_run_successfully(['beta_scan','clean_beta_scan'])
	
	with John.handle_task('clean_beta_scan_plots') as Johns_eployee:
		df = load_whole_dataframe(Johns_eployee.path_to_directory_of_task('beta_scan')/'parsed_from_waveforms.sqlite')
		
		df = tag_n_trigger_as_background_according_to_the_result_of_clean_beta_scan(John, df)
		df = df.reset_index().sort_values('signal_name')
		
		if distributions:
			path_to_save_plots = Johns_eployee.path_to_directory_of_my_task/'distributions'
			path_to_save_plots.mkdir(exist_ok = True)
			for col in df.columns:
				if col in {'signal_name','n_trigger','is_background'}:
					continue
				fig = px.histogram(
					df,
					title = f'{col} histogram<br><sup>Run: {John.run_name}</sup>',
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
					title = f'{col} ECDF<br><sup>Run: {John.run_name}</sup>',
					x = col,
					facet_row = 'signal_name',
					color = 'is_background',
					color_discrete_map = COLOR_DISCRETE_MAP,
				)
				fig.write_html(
					str(path_to_save_plots/Path(f'{col} ecdf.html')),
					include_plotlyjs = 'cdn',
				)
				
		if scatter_plot:
			columns_for_scatter_matrix_plot = set(df.columns) 
			columns_for_scatter_matrix_plot -= {'n_trigger','signal_name','is_background','n_waveform'} 
			columns_for_scatter_matrix_plot -= {f't_{i} (s)' for i in [10,20,30,40,60,70,80,90]}
			columns_for_scatter_matrix_plot -= {f'Time over {i}% (s)' for i in [10,30,40,50,60,70,80,90]}
			fig = px.scatter_matrix(
				df,
				dimensions = sorted(columns_for_scatter_matrix_plot),
				title = f'Scatter matrix plot<br><sup>Run: {John.run_name}</sup>',
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
				str(Johns_eployee.path_to_directory_of_my_task/Path('scatter matrix plot.html')),
				include_plotlyjs = 'cdn',
			)
			
			for variables in {('t_50 (s)','Amplitude (V)')}:
				fig = px.scatter(
					data_frame = df,
					x = variables[0],
					y = variables[1],
					title = f'Scatter plot<br><sup>{bureaucrat.run_name}</sup>',
					symbol = 'signal_name',
					color = 'is_background',
					color_discrete_map = COLOR_DISCRETE_MAP,
				)
				fig.write_html(
					str(Johns_eployee.path_to_directory_of_my_task/Path(' '.join(list(variables)).replace(' ','_') + '_scatter_plot.html')),
					include_plotlyjs = 'cdn',
				)
		
		if langauss_plots:
			for col in {'Amplitude (V)','Collected charge (V s)'}:
				fig = go.Figure()
				fig.update_layout(
					title = f'Langauss fit to {col} after cleaning<br><sup>Run: {John.run_name}</sup>',
					xaxis_title = col,
					yaxis_title = 'count',
				)
				colors = iter(px.colors.qualitative.Plotly)
				for signal_name in sorted(set(df['signal_name'])):
					draw_histogram_and_langauss_fit(
						fig = fig,
						parsed_from_waveforms_df = df.query('is_background==False').set_index(['n_waveform','signal_name']),
						signal_name = signal_name,
						column_name = col,
						line_color = next(colors),
					)
				fig.write_html(
					str(Johns_eployee.path_to_directory_of_my_task/f'langauss fit to {col}.html'),
					include_plotlyjs = 'cdn',
				)

def plots_of_clean_beta_scan_sweeping_bias_voltage(bureaucrat:RunBureaucrat, scatter_plot:bool=False, langauss_plots:bool=True, distributions:bool=False):
	Ernesto = bureaucrat
	Ernesto.check_these_tasks_were_run_successfully('beta_scan_sweeping_bias_voltage')
	
	with Ernesto.handle_task('plots_of_clean_beta_scan_sweeping_bias_voltage') as Ernestos_employee:
		for dummy_bureaucrat in Ernesto.list_subruns_of_task('beta_scan_sweeping_bias_voltage'):
			clean_beta_scan_plots(dummy_bureaucrat, scatter_plot=scatter_plot, langauss_plots=langauss_plots, distributions=distributions)
		path_to_subplots = []
		plot_types = [p.stem for p in (Ernesto.list_subruns_of_task('beta_scan_sweeping_bias_voltage')[0].path_to_directory_of_task('clean_beta_scan_plots')).iterdir() if p.suffix == '.html']
		for plot_type in plot_types:
			for dummy_bureaucrat in Ernestos_employee.list_subruns_of_task('beta_scan_sweeping_bias_voltage'):
				path_to_subplots.append(
					{
						'plot_type': plot_type,
						'path_to_plot': Path('..')/(dummy_bureaucrat.path_to_directory_of_task('clean_beta_scan_plots')/f'{plot_type}.html').relative_to(Ernesto.path_to_run_directory),
						'run_name': dummy_bureaucrat.run_name,
					}
				)
		path_to_subplots_df = pandas.DataFrame(path_to_subplots).set_index('plot_type')
		for plot_type in set(path_to_subplots_df.index.get_level_values('plot_type')):
			document_title = f'{plot_type} plots from clean_beta_scan_plots {Ernesto.run_name}'
			html_doc = dominate.document(title=document_title)
			with html_doc:
				dominate.tags.h1(document_title)
				if plot_type in {'scatter matrix plot'}: # This is because these kind of plots draw a lot of memory and will cause problems if they are loaded all together.
					with dominate.tags.ul():
						for idx,row in path_to_subplots_df.loc[plot_type].sort_values('run_name').iterrows():
							with dominate.tags.li():
								dominate.tags.a(row['run_name'], href=row['path_to_plot'])
				else:
					with dominate.tags.div(style='display: flex; flex-direction: column; width: 100%;'):
						for idx,row in path_to_subplots_df.loc[[plot_type]].sort_values('run_name').iterrows():
							dominate.tags.iframe(src=str(row['path_to_plot']), style=f'height: 100vh; min-height: 600px; width: 100%; min-width: 600px; border-style: none;')
			with open(Ernestos_employee.path_to_directory_of_my_task/f'{plot_type} together.html', 'w') as ofile:
				print(html_doc, file=ofile)

def script_core(bureaucrat:RunBureaucrat, path_to_cuts_file:Path=None):
	John = bureaucrat
	if John.was_task_run_successfully('beta_scan_sweeping_bias_voltage'):
		clean_beta_scan_sweeping_bias_voltage(John, path_to_cuts_file=path_to_cuts_file)
		plots_of_clean_beta_scan_sweeping_bias_voltage(John, langauss_plots=True)
	elif John.was_task_run_successfully('beta_scan'):
		clean_beta_scan(John, path_to_cuts_file=path_to_cuts_file)
		clean_beta_scan_plots(John, distributions=True)
	else:
		raise RuntimeError(f'Dont know how to process run {repr(John.run_name)} located in {John.path_to_run_directory}.')

def tag_n_trigger_as_background_according_to_the_result_of_clean_beta_scan(bureaucrat:RunBureaucrat, df:pandas.DataFrame)->pandas.DataFrame:
	"""If there was a "beta scan cleaning" performed on the measurement
	being managed by the `bureaucrat`, it will be used to tag each `n_trigger` 
	in `df`	as background or not background.
	Note that `df` must have `n_trigger` as an index in order for this
	to be possible. If no successful "clean_beta_scan" task is found by
	`bureaucrat`, an error is raised.
	
	Arguments
	---------
	bureaucrat: RunBureaucrat
		A bureaucrat pointing to a run in which there was a "beta_scan", 
		and possibly (but not mandatory) a "clean_beta_scan".
	df: pandas.DataFrame
		The data frame you want to clean according to the "clean beta scan"
		procedure. An index of this data frame must be the `n_trigger` column.
	
	Returns
	-------
	df: pandas.DataFrame
		A data frame identical to `df` with a new column named `is_background`
		that tags with `True` or `False` each `n_trigger` value.
	"""
	
	Ernesto = bureaucrat
	
	Ernesto.check_these_tasks_were_run_successfully(['beta_scan','clean_beta_scan'])
	
	if 'n_trigger' not in df.index.names:
		raise ValueError(f'`"n_trigger"` cannot be found in the index of `df`. I need it in order to match to the results of the `clean_beta_scan` task.')
	
	df = df.merge(
		right = pandas.read_feather(Ernesto.path_to_directory_of_task('clean_beta_scan')/'result.fd').set_index('n_trigger'),
		left_index = True, 
		right_index = True
	)
	return df

def automatic_cut_amplitude(bureaucrat:RunBureaucrat, DUT_signal_name:str):
	"""Automatically finds the threshold value for cutting in the amplitude.
	
	Arguments
	---------
	bureaucrat: RunBureaucrat
		A bureaucrat object handling the run in which to apply the automatic
		cuts on the amplitude.
	DUT_signal_name: str
		Name of the signal that is the DUT.
	"""
	bureaucrat.check_these_tasks_were_run_successfully('beta_scan')
	with bureaucrat.handle_task('automatic_cut_amplitude') as employee:
		data = load_whole_dataframe(bureaucrat.path_to_directory_of_task('beta_scan')/'parsed_from_waveforms.sqlite')
		
		if DUT_signal_name not in set(data.index.get_level_values('signal_name')):
			raise RuntimeError(f'`DUT_signal_name={repr(DUT_signal_name)}` not present in data, available signals names are {set(data.index.get_level_values("signal_name"))}. ')
		
		data = data.query(f'signal_name=="{DUT_signal_name}"')
		
		x = data['Amplitude (V)']
		x = x.dropna()
		x = x[(x>=x.quantile(.05)) & (x<=x.quantile(.95))]
		threshold_cut = data.loc[x.sort_values().diff().idxmax(),'Amplitude (V)']
		
		cuts = pandas.DataFrame(
			{
				'signal_name': DUT_signal_name,
				'variable': 'Amplitude (V)',
				'cut_type': 'lower',
				'cut_value': threshold_cut,
			},
			index = [0],
		)
		cuts.to_csv(employee.path_to_directory_of_my_task/'cuts.csv', index=False)
		
		fig = px.ecdf(
			data['Amplitude (V)'],
			title = f'Automatic threshold cut in amplitude<br><sup>{bureaucrat.run_name}</sup>',
		)
		fig.add_vline(
			x = threshold_cut,
			annotation_text = 'Threshold',
			annotation_textangle = 90,
		)
		fig.write_html(
			employee.path_to_directory_of_my_task/'cut_in_amplitude.html',
			include_plotlyjs = 'cdn',
		)

def automatic_cuts(bureaucrat:RunBureaucrat, DUT_signal_name:str):
	bureaucrat.check_these_tasks_were_run_successfully('beta_scan_sweeping_bias_voltage')
	with bureaucrat.handle_task('automatic_cuts') as employee:
		cuts = []
		path_to_subplots = []
		for subrun in bureaucrat.list_subruns_of_task('beta_scan_sweeping_bias_voltage'):
			automatic_cut_amplitude(subrun, DUT_signal_name=DUT_signal_name)
			df = pandas.read_csv(subrun.path_to_directory_of_task('automatic_cut_amplitude')/'cuts.csv')
			df['run_name'] = subrun.run_name
			cuts.append(df)
			path_to_subplots.append(
				pandas.DataFrame(
					{
						'path': Path('..')/(subrun.path_to_directory_of_task('automatic_cut_amplitude')/'cut_in_amplitude.html').relative_to(bureaucrat.path_to_run_directory),
						'run_name': subrun.run_name,
					},
					index = [0],
				).set_index('run_name', drop=True)
			)
		cuts = pandas.concat(cuts)
		cuts.to_csv(employee.path_to_directory_of_my_task/'cuts.csv', index=False)
		path_to_subplots = pandas.concat(path_to_subplots)
		
		document_title = f'Automatic amplitude cuts plots for {bureaucrat.run_name}'
		html_doc = dominate.document(title=document_title)
		with html_doc:
			dominate.tags.h1(document_title)
			with dominate.tags.div(style='display: flex; flex-direction: column; width: 100%;'):
				for i,row in path_to_subplots.sort_index().iterrows():
					dominate.tags.iframe(
						src = str(
							row['path']
						), 
						style = f'height: 100vh; min-height: 600px; width: 100%; min-width: 600px; border-style: none;'
					)
		with open(employee.path_to_directory_of_my_task/f'automatic_cuts_plots.html', 'w') as ofile:
			print(html_doc, file=ofile)

if __name__ == '__main__':
	import argparse
	
	grafica.plotly_utils.utils.set_my_template_as_default()

	parser = argparse.ArgumentParser(description='Cleans a beta scan according to some criterion.')
	parser.add_argument('--dir',
		metavar = 'path',
		help = 'Path to the base measurement directory.',
		required = True,
		dest = 'directory',
		type = str,
	)

	args = parser.parse_args()
	bureaucrat = RunBureaucrat(Path(args.directory))
	# ~ script_core(bureaucrat)
	automatic_cut_amplitude(bureaucrat, DUT_signal_name='DUT_CH1')
	clean_beta_scan(bureaucrat, bureaucrat.path_to_directory_of_task('automatic_cut_amplitude')/'cuts.csv')
	clean_beta_scan_plots(bureaucrat)
