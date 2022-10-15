from the_bureaucrat.bureaucrats import RunBureaucrat # https://github.com/SengerM/the_bureaucrat
from pathlib import Path
import pandas
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from huge_dataframe.SQLiteDataFrame import load_whole_dataframe # https://github.com/SengerM/huge_dataframe
from scipy.stats import median_abs_deviation
from scipy.optimize import curve_fit
from landaupy import langauss, landau # https://github.com/SengerM/landaupy
from grafica.plotly_utils.utils import scatter_histogram # https://github.com/SengerM/grafica
import warnings
import dominate # https://github.com/Knio/dominate

def hex_to_rgba(h, alpha):
    return tuple([int(h.lstrip('#')[i:i+2], 16) for i in (0, 2, 4)] + [alpha])

def binned_fit_langauss(samples, bins='auto', nan_policy='drop', maxfev=0):
	"""Perform a binned fit of a langauss distribution.
	
	Arguments
	---------
	samples: array
		Array of samples that are believed to follow a Langauss distribution.
	bins:
		Same as in numpy.histogram.
	nan_policy: str, default `'remove'`
		What to do with NaN values.
	
	Returns
	-------
	popt:
		See scipy.optimize.curve_fit.
	pcov:
		See scipy.optimize.curve_fit.
	hist:
		See numpy.histogram.
	bin_centers:
		Same idea as in numpy.histogram but centers instead of edges.
	"""
	if nan_policy == 'drop':
		samples = samples[~np.isnan(samples)]
	else:
		raise NotImplementedError(f'`nan_policy={nan_policy}` not implemented.')
	if len(samples) == 0:
		raise ValueError(f'`samples` is an empty array.')
	hist, bin_edges = np.histogram(samples, bins, density=True)
	bin_centers = bin_edges[:-1] + np.diff(bin_edges)/2
	# Add an extra bin to the left:
	hist = np.insert(hist, 0, sum(samples<bin_edges[0]))
	bin_centers = np.insert(bin_centers, 0, bin_centers[0]-np.diff(bin_edges)[0])
	# Add an extra bin to the right:
	hist = np.append(hist,sum(samples>bin_edges[-1]))
	bin_centers = np.append(bin_centers, bin_centers[-1]+np.diff(bin_edges)[0])
	landau_x_mpv_guess = bin_centers[np.argmax(hist)]
	landau_xi_guess = median_abs_deviation(samples)/5
	gauss_sigma_guess = landau_xi_guess/2 
	popt, pcov = curve_fit(
		lambda x, mpv, xi, sigma: langauss.pdf(x, mpv, xi, sigma),
		xdata = bin_centers,
		ydata = hist,
		p0 = [p*((np.random.rand()-.5)*.1+1) for p in [landau_x_mpv_guess, landau_xi_guess, gauss_sigma_guess]], # I am multiplying by this random number just for those cases in which it cannot converge by default, this introduces some noise that helps making it to converge by just running it another time.
		absolute_sigma = True,
		maxfev = maxfev,
		# ~ bounds = ([0]*3, [float('inf')]*3), # Don't know why setting the limits make this to fail.
	)
	return popt, pcov, hist, bin_centers

def draw_histogram_and_langauss_fit(fig, parsed_from_waveforms_df, signal_name, column_name, line_color, maxfev=0):
	samples = parsed_from_waveforms_df.loc[pandas.IndexSlice[:, signal_name], column_name]
	
	fig.add_trace(
		scatter_histogram(
			samples = samples,
			error_y = dict(type='auto'),
			density = False,
			name = f'Data {signal_name}',
			line = dict(color = line_color),
			legendgroup = signal_name,
		)
	)
	
	fit_successfull = False
	try:
		popt, _, hist, bin_centers = binned_fit_langauss(samples, maxfev=maxfev)
		fit_successfull = True
	except Exception as e:
		warnings.warn(f'Cannot fit langauss to data, reason: {repr(e)}.')
	if fit_successfull == True:
		x_axis = np.linspace(min(bin_centers),max(bin_centers),999)
		fig.add_trace(
			go.Scatter(
				x = x_axis,
				y = langauss.pdf(x_axis, *popt)*len(samples)*np.diff(bin_centers)[0],
				name = f'Langauss fit {signal_name}<br>x<sub>MPV</sub>={popt[0]:.2e}<br>ξ={popt[1]:.2e}<br>σ={popt[2]:.2e}',
				line = dict(color = line_color, dash='dash'),
				legendgroup = signal_name,
			)
		)
		fig.add_trace(
			go.Scatter(
				x = x_axis,
				y = landau.pdf(x_axis, popt[0], popt[1])*len(samples)*np.diff(bin_centers)[0],
				name = f'Landau component {signal_name}',
				line = dict(color = f'rgba{hex_to_rgba(line_color, .3)}', dash='dashdot'),
				legendgroup = signal_name,
			)
		)
	else:
		signal_names = sorted(set(parsed_from_waveforms_df.index.get_level_values('signal_name')))
		fig.add_annotation(
			text = f'Could not fit Langauss to {signal_name}',
			xref = "paper", 
			yref = "paper",
			x = 1,
			y = 1-signal_names.index(signal_name)/len(signal_names),
			showarrow = False,
			align = 'right',
		)

def plot_everything_from_beta_scan(bureaucrat:RunBureaucrat, measured_stuff_vs_when:bool=False, all_distributions:bool=False):
	John = bureaucrat
	
	John.check_these_tasks_were_run_successfully('beta_scan')
	
	measured_stuff_df = load_whole_dataframe(John.path_to_directory_of_task('beta_scan')/Path('measured_stuff.sqlite'))
	parsed_from_waveforms_df = load_whole_dataframe(John.path_to_directory_of_task('beta_scan')/Path('parsed_from_waveforms.sqlite'))
	
	with John.handle_task('plot_everything_from_beta_scan') as task_bureaucrat:
		if measured_stuff_vs_when:
			df = measured_stuff_df.sort_values('When').reset_index()
			path_to_save_plots = task_bureaucrat.path_to_directory_of_my_task/Path('measured_stuff_vs_time')
			path_to_save_plots.mkdir()
			for col in measured_stuff_df.columns:
				if col in {'device_name','When','n_trigger'}:
					continue
				fig = px.line(
					df,
					title = f'{col} vs time<br><sup>Run: {John.run_name}</sup>',
					x = 'When',
					y = col,
					color = 'signal_name',
					markers = True,
				)
				fig.write_html(
					str(path_to_save_plots/Path(f'{col} vs time.html')),
					include_plotlyjs = 'cdn',
				)
		
		df = parsed_from_waveforms_df.reset_index().drop({'n_waveform'}, axis=1).sort_values('signal_name')
		path_to_save_plots = task_bureaucrat.path_to_directory_of_my_task/Path('parsed_from_waveforms')
		path_to_save_plots.mkdir()
		for col in df.columns:
			if col in {'signal_name','n_trigger'}:
				continue
			if not all_distributions:
				if col not in {'Amplitude (V)','Collected charge (V s)','SNR','Noise (V)','t_50 (s)'}:
					continue
			fig = px.histogram(
				df,
				title = f'{col} histogram<br><sup>Run: {John.run_name}</sup>',
				x = col,
				color = 'signal_name',
			)
			fig.write_html(
				str(path_to_save_plots/Path(f'{col} histogram.html')),
				include_plotlyjs = 'cdn',
			)
			
			fig = px.ecdf(
				df,
				title = f'{col} ECDF<br><sup>Run: {John.run_name}</sup>',
				x = col,
				color = 'signal_name',
			)
			fig.write_html(
				str(path_to_save_plots/Path(f'{col} ecdf.html')),
				include_plotlyjs = 'cdn',
			)
			
		columns_for_scatter_matrix_plot = set(df.columns) 
		columns_for_scatter_matrix_plot -= {'n_trigger','signal_name'} 
		columns_for_scatter_matrix_plot -= {f't_{i} (s)' for i in [10,20,30,40,60,70,80,90]}
		columns_for_scatter_matrix_plot -= {f'Time over {i}% (s)' for i in [10,30,40,50,60,70,80,90]}
		fig = px.scatter_matrix(
			df,
			dimensions = sorted(columns_for_scatter_matrix_plot),
			title = f'Scatter matrix plot<br><sup>Run: {John.run_name}</sup>',
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
		
		path_to_save_plots = task_bureaucrat.path_to_directory_of_my_task/Path('parsed_from_waveforms')
		path_to_save_plots.mkdir(exist_ok=True)
		for col in {'Amplitude (V)','Collected charge (V s)'}:
			fig = go.Figure()
			fig.update_layout(
				title = f'Langauss fit to {col}<br><sup>Run: {John.run_name}</sup>',
				xaxis_title = col,
				yaxis_title = 'count',
			)
			colors = iter(px.colors.qualitative.Plotly)
			for signal_name in sorted(set(parsed_from_waveforms_df.index.get_level_values('signal_name'))):
				draw_histogram_and_langauss_fit(
					fig = fig,
					parsed_from_waveforms_df = parsed_from_waveforms_df,
					signal_name = signal_name,
					column_name = col,
					line_color = next(colors),
				)
			fig.write_html(
				str(path_to_save_plots/Path(f'{col} langauss fit.html')),
				include_plotlyjs = 'cdn',
			)
		
		for variables in {('t_50 (s)','Amplitude (V)')}:
			fig = px.scatter(
				data_frame = parsed_from_waveforms_df.reset_index().sort_values('signal_name'),
				x = variables[0],
				y = variables[1],
				title = f'Scatter plot<br><sup>{bureaucrat.run_name}</sup>',
				color = 'signal_name'
			)
			fig.write_html(
				str(path_to_save_plots/Path(' '.join(list(variables)).replace(' ','_') + '_scatter_plot.html')),
				include_plotlyjs = 'cdn',
			)

def plot_everything_from_beta_scans_recursively(bureaucrat:RunBureaucrat, measured_stuff_vs_when:bool=False, all_distributions:bool=False):
	Melvin = bureaucrat
	tasks_in_Melvins_run = [p.parts[-1] for p in Melvin.path_to_run_directory.iterdir() if p.is_dir()]
	if 'beta_scan' in tasks_in_Melvins_run:
		plot_everything_from_beta_scan(Melvin, measured_stuff_vs_when=measured_stuff_vs_when, all_distributions=all_distributions)
	else:
		for task_name in tasks_in_Melvins_run:
			if not Melvin.was_task_run_successfully(task_name):
				continue
			for subrun in Melvin.list_subruns_of_task(task_name):
				plot_everything_from_beta_scans_recursively(subrun)

def plot_everything_from_beta_scan_sweeping_bias_voltage(bureaucrat:RunBureaucrat, measured_stuff_vs_when:bool=False, all_distributions:bool=False):
	Ernesto = bureaucrat
	
	TASKS = {'beta_scan_sweeping_bias_voltage','kill_device_por_Narnia'}
	if not any([Ernesto.was_task_run_successfully(task) for task in TASKS]):
		raise RuntimeError(f'None of the tasks {TASKS} was completed for run {repr(Ernesto.run_name)} located in {repr(str(Ernesto.path_to_run_directory))}.')
	
	with Ernesto.handle_task('plot_everything_from_beta_scan_sweeping_bias_voltage') as Ernestos_employee:
		plot_everything_from_beta_scans_recursively(Ernesto, measured_stuff_vs_when=measured_stuff_vs_when, all_distributions=all_distributions)
		path_to_subplots = []
		plot_types = [p.stem for p in (Ernesto.list_subruns_of_task('beta_scan_sweeping_bias_voltage')[0].path_to_directory_of_task('plot_everything_from_beta_scan')/f'parsed_from_waveforms').iterdir()]
		for plot_type in plot_types:
			for subrun in Ernestos_employee.list_subruns_of_task('beta_scan_sweeping_bias_voltage'):
				path_to_subplots.append(
					{
						'plot_type': plot_type,
						'path_to_plot': Path('..')/(subrun.path_to_directory_of_task('plot_everything_from_beta_scan')/f'parsed_from_waveforms/{plot_type}.html').relative_to(Ernesto.path_to_run_directory),
						'run_name': subrun.run_name,
					}
				)
		path_to_subplots_df = pandas.DataFrame(path_to_subplots).set_index('plot_type')
		for plot_type in set(path_to_subplots_df.index.get_level_values('plot_type')):
			document_title = f'{plot_type} plots from beta_scan_sweeping_bias_voltage {Ernesto.run_name}'
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
						for idx,row in path_to_subplots_df.loc[plot_type].sort_values('run_name').iterrows():
							dominate.tags.iframe(src=str(row['path_to_plot']), style=f'height: 100vh; min-height: 600px; width: 100%; min-width: 600px; border-style: none;')
			with open(Ernestos_employee.path_to_directory_of_my_task/f'{plot_type} together.html', 'w') as ofile:
				print(html_doc, file=ofile)

def script_core(bureaucrat:RunBureaucrat):
	if bureaucrat.was_task_run_successfully('beta_scan'):
		plot_everything_from_beta_scan(bureaucrat = bureaucrat)
	elif any([bureaucrat.was_task_run_successfully(task) for task in {'beta_scan_sweeping_bias_voltage','kill_device_por_Narnia'}]):
		plot_everything_from_beta_scan_sweeping_bias_voltage(bureaucrat = bureaucrat)
	else:
		raise RuntimeError(f'Dont know how to process run {repr(bureaucrat.run_name)} located in {bureaucrat.path_to_run_directory}.')

if __name__ == '__main__':
	import argparse

	parser = argparse.ArgumentParser()
	parser.add_argument('--dir',
		metavar = 'path', 
		help = 'Path to the base measurement directory.',
		required = True,
		dest = 'directory',
		type = str,
	)
	
	args = parser.parse_args()
	
	Enrique = RunBureaucrat(Path(args.directory))
	script_core(Enrique)
