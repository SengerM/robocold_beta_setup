from bureaucrat.SmarterBureaucrat import NamedTaskBureaucrat # https://github.com/SengerM/bureaucrat
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

def hex_to_rgba(h, alpha):
    return tuple([int(h.lstrip('#')[i:i+2], 16) for i in (0, 2, 4)] + [alpha])

def binned_fit_langauss(samples, bins='auto', nan_policy='drop'):
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
	gauss_sigma_guess = landau_xi_guess/10
	popt, pcov = curve_fit(
		lambda x, mpv, xi, sigma: langauss.pdf(x, mpv, xi, sigma),
		xdata = bin_centers,
		ydata = hist,
		p0 = [landau_x_mpv_guess, landau_xi_guess, gauss_sigma_guess],
		absolute_sigma = True,
		# ~ bounds = ([0]*3, [float('inf')]*3), # Don't know why setting the limits make this to fail.
	)
	return popt, pcov, hist, bin_centers

def draw_histogram_and_langauss_fit(fig, parsed_from_waveforms_df, signal_name, column_name, line_color):
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
		popt, _, hist, bin_centers = binned_fit_langauss(samples)
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

def plot_everything_from_beta_scan(directory: Path):
	John = NamedTaskBureaucrat(
		directory,
		task_name = 'plot_everything_from_beta_scan',
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
		
		# ~ df = parsed_from_waveforms_df.reset_index().drop({'n_waveform'}, axis=1).sort_values('signal_name')
		# ~ path_to_save_plots = John.path_to_default_output_directory/Path('parsed_from_waveforms')
		# ~ path_to_save_plots.mkdir()
		# ~ for col in df.columns:
			# ~ if col in {'signal_name','n_trigger'}:
				# ~ continue
			# ~ fig = px.histogram(
				# ~ df,
				# ~ title = f'{col} histogram<br><sup>Measurement: {John.measurement_name}</sup>',
				# ~ x = col,
				# ~ facet_row = 'signal_name',
			# ~ )
			# ~ fig.write_html(
				# ~ str(path_to_save_plots/Path(f'{col} histogram.html')),
				# ~ include_plotlyjs = 'cdn',
			# ~ )
			
			# ~ fig = px.ecdf(
				# ~ df,
				# ~ title = f'{col} ECDF<br><sup>Measurement: {John.measurement_name}</sup>',
				# ~ x = col,
				# ~ facet_row = 'signal_name',
			# ~ )
			# ~ fig.write_html(
				# ~ str(path_to_save_plots/Path(f'{col} ecdf.html')),
				# ~ include_plotlyjs = 'cdn',
			# ~ )
			
			# ~ columns_for_scatter_matrix_plot = set(df.columns) 
			# ~ columns_for_scatter_matrix_plot -= {'n_trigger','signal_name'} 
			# ~ columns_for_scatter_matrix_plot -= {f't_{i} (s)' for i in [10,20,30,40,60,70,80,90]}
			# ~ columns_for_scatter_matrix_plot -= {f'Time over {i}% (s)' for i in [10,30,40,50,60,70,80,90]}
			# ~ fig = px.scatter_matrix(
				# ~ df,
				# ~ dimensions = sorted(columns_for_scatter_matrix_plot),
				# ~ title = f'Scatter matrix plot<br><sup>Measurement: {John.measurement_name}</sup>',
				# ~ color = 'signal_name',
				# ~ hover_data = ['n_trigger'],
			# ~ )
			# ~ fig.update_traces(diagonal_visible=False, showupperhalf=False, marker = {'size': 3})
			# ~ for k in range(len(fig.data)):
				# ~ fig.data[k].update(
					# ~ selected = dict(
						# ~ marker = dict(
							# ~ opacity = 1,
							# ~ color = 'black',
						# ~ )
					# ~ ),
				# ~ )
			# ~ fig.write_html(
				# ~ str(path_to_save_plots/Path('scatter matrix plot.html')),
				# ~ include_plotlyjs = 'cdn',
			# ~ )
		
		path_to_save_plots = John.path_to_default_output_directory/Path('parsed_from_waveforms')
		path_to_save_plots.mkdir(exist_ok=True)
		for col in {'Amplitude (V)','Collected charge (V s)'}:
			fig = go.Figure()
			fig.update_layout(
				title = f'Langauss fit to {col}<br><sup>Measurement: {John.measurement_name}</sup>',
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
	plot_everything_from_beta_scan(Path(args.directory))
