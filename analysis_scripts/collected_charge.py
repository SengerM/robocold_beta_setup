from the_bureaucrat.bureaucrats import RunBureaucrat # https://github.com/SengerM/the_bureaucrat
from pathlib import Path
import pandas
import plotly.graph_objects as go
import plotly.express as px
from huge_dataframe.SQLiteDataFrame import SQLiteDataFrameDumper, load_whole_dataframe # https://github.com/SengerM/huge_dataframe
from grafica.plotly_utils.utils import scatter_histogram, set_my_template_as_default, line # https://github.com/SengerM/grafica
import numpy
from landaupy import langauss, landau
from scipy.stats import gaussian_kde, median_abs_deviation
import warnings
from scipy.constants import elementary_charge
import multiprocessing
import sys
sys.path.insert(1, '/home/sengerm/scripts_and_codes/repos/robocold_beta_setup/analysis_scripts')
from summarize_parameters import read_summarized_data
from lmfit import Model, Parameter, report_fit
from uncertainties import ufloat
from utils import my_std, save_dataframe, kMAD, get_function_arguments_names, hex_to_rgba, resample_by_n_trigger, starmap_checking_arguments_names
from clean_beta_scan import tag_n_trigger_as_background_according_to_the_result_of_clean_beta_scan

def signal_probability_density_model(x, landau_x_mpv, landau_xi, landau_gauss_sigma):
	return langauss.pdf(x, landau_x_mpv, landau_xi, landau_gauss_sigma)

def fit_Landau_and_extract_MPV(bureaucrat:RunBureaucrat, time_from_trigger_background:dict, time_from_trigger_signal:dict, signal_name_trigger:str, n_bootstraps:int, collected_charge_variable_name:str, force:bool=False, use_clean_beta_scan:bool=True):
	"""Fit a Landau (Langauss) to a variable and extract the MPV.
	
	Arguments
	---------
	bureaucrat: RunBureaucrat
		An instance of this class pointing to the run to be analyzed.
	time_from_trigger_background: dict
		A dictionary where the keys are names of signals present in the
		data and the items are tuples of the form `(float,float)` specifying
		where to look for background events for each signal. E.g.:
		`
		time_from_trigger_background = {
			'DUT_CH1': (5e-9,10e-9),
			'DUT_CH2': (5e-9,10e-9),
		}
		`
		The 'time_from_trigger` for a signal named `signal_name` is defined 
		as 
		```
		time_from_trigger = parsed_from_waveforms.loc[signal_name,'t_50 (s)'] - parsed_from_waveforms.loc[signal_name_trigger,'t_50 (s)']
		```
	time_from_trigger_signal: dict
		A dictionary where the keys are names of signals present in the
		data and the items are tuples of the form `(float,float)` specifying
		where to look for background events for each signal. E.g.:
		`
		time_from_trigger_signal = {
			'DUT_CH1': (13e-9,14e-9),
			'DUT_CH2': (13e-9,14e-9),
		}
		`
		Note that the keys of this dictionary must be the same as the
		keys of `time_from_trigger_background`.
	signal_name_trigger: str
		The name of a signal present in the data to be used as trigger.
	n_bootstraps: int
		Number of fits to perform with resampled data to estimate the
		statistical uncertainty.
	collected_charge_variable_name: str
		Variable that best represents the collected charge in which one
		expects the Landau distribution. Usually this is `'Amplitude (V)'`
		or `'Collected charge (V s)'` or so.
	use_clean_beta_scan: bool, default `True`
		Whether to use the information from the task `clean_beta_scan`. If
		`False`, it is ignored. If `True` then only 'clean events' are used.
	"""
	bureaucrat.check_these_tasks_were_run_successfully('beta_scan')
	
	TASK_NAME = f'fit_Landau_and_extract_MPV_{collected_charge_variable_name.replace(" ","_")}'
	if force == False and bureaucrat.was_task_run_successfully(TASK_NAME): # If this was already done, don't do it again...
		return
	
	if set(time_from_trigger_signal.keys()) != set(time_from_trigger_background.keys()):
		raise ValueError(f'`time_from_trigger_background` and `time_from_trigger_signal` must have the same keys.')
	
	data = load_whole_dataframe(bureaucrat.path_to_directory_of_task('beta_scan')/'parsed_from_waveforms.sqlite')
	
	if collected_charge_variable_name not in data.columns:
		raise ValueError(f'`collected_charge_variable_name` {repr(collected_charge_variable_name)} not found among the colums of the data, which are {sorted(data.columns)}. ')
	
	with bureaucrat.handle_task(TASK_NAME) as employee:
		_A = data['t_50 (s)']
		_B = data.query(f'signal_name == "{signal_name_trigger}"')['t_50 (s)'].reset_index('signal_name', drop=True)
		data['Time from trigger (s)'] = _A-_B
		
		if use_clean_beta_scan:
			data = tag_n_trigger_as_background_according_to_the_result_of_clean_beta_scan(bureaucrat, data)
			data = data.query('is_background == False')
		
		if signal_name_trigger not in set(data.index.get_level_values('signal_name')):
			raise ValueError(f'`signal_name_trigger` {repr(signal_name_trigger)} not present among the signals of the dataset which are {sorted(set(data.index.get_level_values("signal_name")))}. ')
		
		variable = collected_charge_variable_name
		
		params_to_save = []
		# Because the scipy.curve_fit works better when the numbers are close to 1, I rescale the data. Otherwise it fails.
		scale_factor_for_comfortable_fit = numpy.nanmean(data[variable])**-1
		scaled_variable_name = variable + ' SCALED'
		with warnings.catch_warnings():
			warnings.filterwarnings("ignore")
			data[scaled_variable_name] = data[variable]*scale_factor_for_comfortable_fit
		
		fig_background = go.Figure()
		fig = go.Figure()
		colors = iter(px.colors.qualitative.Plotly)
		hist_max_for_the_plot_that_does_not_handle_it_automatically_yet = -1
		for n_bootstrap in range(n_bootstraps+1):
			for signal_name in sorted(time_from_trigger_signal.keys()):
				samples_with_signal_and_background = data.loc[(data['Time from trigger (s)']>time_from_trigger_signal[signal_name][0])&(data['Time from trigger (s)']<time_from_trigger_signal[signal_name][1])]
				samples_with_signal_and_background = samples_with_signal_and_background.query(f'signal_name=="{signal_name}"')
				samples_with_signal_and_background = samples_with_signal_and_background[scaled_variable_name]
				
				samples_background = data.loc[(data['Time from trigger (s)']>time_from_trigger_background[signal_name][0])&(data['Time from trigger (s)']<time_from_trigger_background[signal_name][1])]
				samples_background = samples_background.query(f'signal_name=="{signal_name}"')
				samples_background = samples_background[scaled_variable_name]
				
				if n_bootstrap == 0: # Means with real data.
					kind_of_data = 'original'
				else:
					samples_with_signal_and_background = resample_by_n_trigger(samples_with_signal_and_background)
					samples_background = resample_by_n_trigger(samples_background)
					kind_of_data = 'resampled'
				
				# Remove NaN values because fitting algorithms always cry otherwise...
				samples_with_signal_and_background = samples_with_signal_and_background.loc[~numpy.isnan(samples_with_signal_and_background)]
				samples_background = samples_background.loc[~numpy.isnan(samples_background)]
				
				if len(samples_with_signal_and_background)==0:
					raise RuntimeError(f'No signal events were found for `time_from_trigger_signal[signal_name]` = {repr(time_from_trigger_signal[signal_name])}. ')
				if len(samples_background)==0:
					raise RuntimeError(f'No background events were found for `time_from_trigger_background[signal_name]` = {repr(time_from_trigger_background[signal_name])}. ')
				
				try:
					background_probability_density_model = gaussian_kde(samples_background)
				except Exception:
					background_probability_density_model = lambda x: numpy.array([0]*len(x))
				
				if n_bootstrap == 0: # Means with the real data so do a plot.
					line_color = next(colors)
					hist, bin_edges = numpy.histogram(samples_background/scale_factor_for_comfortable_fit, bins='auto', density=False)
					fig_background.add_trace(
						scatter_histogram(
							samples = samples_background/scale_factor_for_comfortable_fit,
							error_y = dict(type='auto'),
							density = False,
							name = f'{signal_name} background',
							line = dict(color = line_color),
							legendgroup = signal_name,
							bins = bin_edges,
						)
					)
					x_axis = numpy.array(sorted(list(samples_background.sample(n=99, replace=True)) + list(numpy.linspace(samples_background.min(),samples_background.max(),99))))
					fig_background.add_trace(
						go.Scatter(
							x = x_axis/scale_factor_for_comfortable_fit,
							y = background_probability_density_model(x_axis)*numpy.diff(bin_edges)[0]*len(samples_background)*scale_factor_for_comfortable_fit,
							name = f'{signal_name} background model',
							line = dict(color = line_color, dash='dash'),
							legendgroup = signal_name,
						)
					)
				
				hist, bin_edges = numpy.histogram(samples_with_signal_and_background, bins='auto', density=False)
				bin_centers = bin_edges[:-1] + numpy.diff(bin_edges)/2
				# ~ # Add an extra bin to the left:
				hist = numpy.insert(hist, 0, sum(samples_with_signal_and_background<bin_edges[0]))
				bin_centers = numpy.insert(bin_centers, 0, bin_centers[0]-numpy.diff(bin_edges)[0])
				# Add an extra bin to the right:
				hist = numpy.append(hist,sum(samples_with_signal_and_background>bin_edges[-1]))
				bin_centers = numpy.append(bin_centers, bin_centers[-1]+numpy.diff(bin_edges)[0])
				
				def probability_density_model(x, landau_x_mpv, landau_xi, landau_gauss_sigma, signal_to_background_fraction):
					# This integrated to 1.
					return signal_to_background_fraction*signal_probability_density_model(x, landau_x_mpv, landau_xi, landau_gauss_sigma) + (1-signal_to_background_fraction)*background_probability_density_model(x)
				
				def events_count_model(x, landau_x_mpv, landau_xi, landau_gauss_sigma, n_signal, n_background):
					# This integrates to `(n_signal+n_background)*bin_size` with `bin_size` being the size of the first bin (assuming all identical).
					N = n_background+n_signal
					return probability_density_model(x, landau_x_mpv, landau_xi, landau_gauss_sigma, n_signal/N)*N*numpy.diff(bin_edges)[0]
				
				model = Model(events_count_model, independent_vars=['x'])
				params = model.make_params()
				params['landau_x_mpv'].value = bin_centers[2:][numpy.argmax(hist[2:])] # Remove the two first bin because usually the background is there.
				params['landau_x_mpv'].min = 0
				params['landau_x_mpv'].vary = True
				params['landau_xi'].value = median_abs_deviation(samples_with_signal_and_background)
				params['landau_xi'].min = params['landau_xi'].value/10
				params['landau_xi'].max = params['landau_xi'].value*10
				params['landau_xi'].vary = True
				params['landau_gauss_sigma'].value = median_abs_deviation(samples_with_signal_and_background)
				params['landau_gauss_sigma'].min = params['landau_gauss_sigma'].value/10
				params['landau_gauss_sigma'].max = params['landau_gauss_sigma'].value*10
				params['landau_gauss_sigma'].vary = True
				_background_rate_estimation = len(samples_background)/abs(numpy.diff(time_from_trigger_background[signal_name])[0])
				_sigmas = 1
				params['n_background'].value = _background_rate_estimation*abs(numpy.diff(time_from_trigger_signal[signal_name])[0])
				params['n_background'].min = max(0, params['n_background'].value - _sigmas*params['n_background'].value**.5)
				params['n_background'].max = params['n_background'].value + _sigmas*params['n_background'].value**.5
				params['n_signal'].value = len(samples_with_signal_and_background) - params['n_background'].value
				params['n_signal'].min = max(0, params['n_signal'].value - _sigmas*params['n_signal'].value)
				params['n_signal'].max = params['n_signal'].value + _sigmas*params['n_signal'].value
				params['n_background'].vary = True
				params['n_signal'].vary = True
				
				with warnings.catch_warnings():
					warnings.filterwarnings("ignore", message="overflow encountered in divide")
					warnings.filterwarnings("ignore", message="invalid value encountered in divide")
					warnings.filterwarnings("ignore", message="divide by zero encountered in divide")
					try:
						result = model.fit(hist, params, x=bin_centers)
						_ = {param_name: result.params[param_name].value for param_name in result.params}
						
					except Exception as e:
						warnings.warn(f'Cannot fit signal_name {repr(signal_name)}, variable {repr(variable)}, n_bootstrap {repr(n_bootstrap)}, run {repr(bureaucrat.run_name)}. Reason: {e}')
						_ = {param_name: numpy.nan for param_name in params}
				_['signal_name'] = signal_name
				_['variable'] = scaled_variable_name
				_['kind_of_data'] = kind_of_data
				params_to_save.append(_)
				
				if n_bootstrap == 0 and 'result' in locals(): # Means with real data.
					PARAMS_LABELS = {
						'landau_x_mpv': 'x<sub>MPV</sub>', 
						'landau_xi': 'ξ', 
						'landau_gauss_sigma': 'σ', 
						'n_signal': 'n<sub>signal</sub>', 
						'n_background': 'n<sub>background</sub>',
					}
					hist_max_for_the_plot_that_does_not_handle_it_automatically_yet = max(hist_max_for_the_plot_that_does_not_handle_it_automatically_yet,hist.max())
					fig.add_trace(
						scatter_histogram(
							samples = samples_with_signal_and_background/scale_factor_for_comfortable_fit,
							error_y = dict(type='auto'),
							density = False,
							name = f'{signal_name}',
							line = dict(color = line_color),
							legendgroup = signal_name,
							bins = bin_edges/scale_factor_for_comfortable_fit,
						)
					)
					x_axis = numpy.array(sorted(list(samples_with_signal_and_background.sample(n=99,replace=True)) + list(numpy.linspace(samples_with_signal_and_background.min(),samples_with_signal_and_background.max(),99))))
					fig.add_trace(
						go.Scatter(
							x = x_axis/scale_factor_for_comfortable_fit,
							y = events_count_model(x_axis, **{p:result.params[p].value for p in result.params}),
							name = f'{signal_name} model' + '<br>'.join([''] + [f'{PARAMS_LABELS[param_name]}: {int(result.params[param_name].value)}' for param_name in ['n_signal','n_background']]),
							line = dict(color = line_color, dash='dash'),
							legendgroup = signal_name,
						)
					)
					fig.add_trace(
						go.Scatter(
							x = x_axis/scale_factor_for_comfortable_fit,
							y = background_probability_density_model(x_axis)*result.params['n_background'].value*numpy.diff(bin_edges)[0],
							name = f'{signal_name} background model',
							line = dict(color = f'rgba{hex_to_rgba(line_color, .3)}', dash='dashdot'),
							legendgroup = signal_name,
						)
					)
					fig.add_trace(
						go.Scatter(
							x = x_axis/scale_factor_for_comfortable_fit,
							y = signal_probability_density_model(x_axis, **{param_name: result.params[param_name].value for param_name in result.params if param_name in get_function_arguments_names(signal_probability_density_model)})*result.params['n_signal'].value*numpy.diff(bin_edges)[0],
							name = f'{signal_name} signal model' + '<br>'.join([''] + [f'{PARAMS_LABELS[param_name]}: {result.params[param_name].value/scale_factor_for_comfortable_fit:.2e}' for param_name in ['landau_x_mpv', 'landau_xi', 'landau_gauss_sigma']]),
							line = dict(color = f'rgba{hex_to_rgba(line_color, .3)}', dash='longdashdot'),
							legendgroup = signal_name,
						)
					)
				
			if n_bootstrap == 0: # Means with real data.
				try:
					fig_background.update_layout(
						title = f'{scaled_variable_name.replace(" SCALED","")} background model<br><sup>{bureaucrat.run_name}</sup>',
						xaxis_title = scaled_variable_name.replace(" SCALED",""),
						yaxis_title = 'Count',
					)
					fig.update_layout(
						title = f'{scaled_variable_name.replace(" SCALED","")} fit<br><sup>{bureaucrat.run_name}</sup>',
						xaxis_title = scaled_variable_name.replace(" SCALED",""),
						yaxis_title = 'Count',
					)
					for _ in [fig,fig_background]:
						_.update_yaxes(type='log', range=[numpy.log10(.5),numpy.log10(hist_max_for_the_plot_that_does_not_handle_it_automatically_yet*1.3)])
					fig_background.write_html(
						employee.path_to_directory_of_my_task/f'{scaled_variable_name} background model.html',
						include_plotlyjs = 'cdn',
					)
					fig.write_html(
						employee.path_to_directory_of_my_task/f'{scaled_variable_name.replace(" SCALED","")} fit.html',
						include_plotlyjs = 'cdn',
					)
				except UnboundLocalError:
					pass
		
		params_to_save = pandas.DataFrame.from_records(params_to_save).set_index(['variable','signal_name'])
		
		unscaled_params_to_save = params_to_save.copy()
		for variable in {collected_charge_variable_name}:
			unscaled_params_to_save.loc[pandas.IndexSlice[variable+' SCALED',:],['landau_x_mpv','landau_xi','landau_gauss_sigma']] /= scale_factor_for_comfortable_fit
			unscaled_params_to_save.rename(index={variable+' SCALED':variable},inplace=True)
		params_to_save = unscaled_params_to_save
		
		for variable in unscaled_params_to_save.index.get_level_values('variable'):
			fig = px.violin(
				title = f'x<sub>MPV</sub> distribution for {variable}<br><sup>{bureaucrat.run_name}</sup>',
				data_frame = params_to_save.reset_index(drop=False).query(f'variable=="{variable}"').sort_values('signal_name'),
				x = 'signal_name',
				y = 'landau_x_mpv',
				points = "all",
				hover_data = None,
				labels = {
					'landau_x_mpv': 'x<sub>MPV</sub> ' + variable[variable.index('('):],
				}
			)
			fig.update_traces(
				hovertemplate = None,
				hoverinfo = 'skip'
			)
			fig.write_html(
				employee.path_to_directory_of_my_task/f'{variable} x_mpv violins.html',
				include_plotlyjs = 'cdn',
			)
		
		save_dataframe(df=params_to_save, location=employee.path_to_directory_of_my_task, name='fits_results')
		
		x_mpv = params_to_save['landau_x_mpv']
		x_mpv = x_mpv.groupby(x_mpv.index.names).agg([numpy.nanmean, numpy.nanmedian, numpy.nanstd, kMAD])
		save_dataframe(df=x_mpv, location=employee.path_to_directory_of_my_task, name='x_mpv')

def fit_Landau_and_extract_MPV_sweeping_voltage(bureaucrat:RunBureaucrat, time_from_trigger_background:dict, time_from_trigger_signal:dict, signal_name_trigger:str, n_bootstraps:int, collected_charge_variable_name:str, force:bool=False, number_of_processes:int=1):
	bureaucrat.check_these_tasks_were_run_successfully('beta_scan_sweeping_bias_voltage')
	
	with bureaucrat.handle_task(f'fit_Landau_and_extract_MPV_sweeping_voltage_{collected_charge_variable_name.replace(" ","_")}') as employee:
		subruns = bureaucrat.list_subruns_of_task('beta_scan_sweeping_bias_voltage')
		if number_of_processes == 1:
			for subrun in subruns:
				fit_Landau_and_extract_MPV(
					bureaucrat = subrun,
					time_from_trigger_background = time_from_trigger_background[subrun.run_name], 
					time_from_trigger_signal = time_from_trigger_signal[subrun.run_name],
					signal_name_trigger = signal_name_trigger, 
					n_bootstraps = n_bootstraps,
					collected_charge_variable_name = collected_charge_variable_name,
					force = force,
				)
		else:
			with multiprocessing.Pool(number_of_processes) as p:
				starmap_checking_arguments_names(
					pool = p,
					func = fit_Landau_and_extract_MPV,
					args = [
						dict(
							bureaucrat = subrun,
							time_from_trigger_background = time_from_trigger_background[subrun.run_name], 
							time_from_trigger_signal = time_from_trigger_signal[subrun.run_name],
							signal_name_trigger = signal_name_trigger, 
							n_bootstraps = n_bootstraps,
							collected_charge_variable_name = collected_charge_variable_name,
							force = force,
							use_clean_beta_scan = True,
						)
						for subrun in subruns
					],
				)
		
		x_mpv, fits_results = read_MPV_data_from_Landau_fits(bureaucrat, collected_charge_variable_name)
		summarized_data = read_summarized_data(bureaucrat)
		summarized_data.columns = [' '.join(col) for col in summarized_data.columns]
		summarized_data.reset_index('device_name',drop=False,inplace=True)
		
		for col in ['Bias voltage (V) mean','Bias voltage (V) std']:
			fits_results = fits_results.join(summarized_data[col], on='run_name')
			x_mpv = x_mpv.join(summarized_data[col], on='run_name')
		
		for name,df in {'fits_results':fits_results,'x_mpv':x_mpv}.items():
			save_dataframe(df=df,name=name,location=employee.path_to_directory_of_my_task)
		
		for variable in set(x_mpv.index.get_level_values('variable')):
			fig = line(
				title = f'{variable} x<sub>MPV</sub><br><sup>{bureaucrat.run_name}</sup>',
				data_frame = x_mpv.reset_index(drop=False).query(f'variable=="{variable}"').sort_values(['Bias voltage (V) mean','signal_name']),
				x = 'Bias voltage (V) mean',
				y = 'nanmedian',
				error_x = 'Bias voltage (V) std',
				error_y = 'kMAD',
				color = 'signal_name',
				markers = True,
				labels = {
					'nanmedian': f'{variable} x<sub>MPV</sub>',
					'nanmean': f'{variable} x<sub>MPV</sub>',
					'Bias voltage (V) mean': 'Bias voltage (V)',
				}
			)
			fig.update_layout(xaxis = dict(autorange = "reversed"))
			fig.write_html(employee.path_to_directory_of_my_task/f'x_mpv vs bias voltage.html', include_plotlyjs='cdn')

def read_MPV_data_from_Landau_fits(bureaucrat:RunBureaucrat, collected_charge_variable_name:str):
	if bureaucrat.was_task_run_successfully('beta_scan'):
		task_name = f'fit_Landau_and_extract_MPV_{collected_charge_variable_name.replace(" ","_")}'
		bureaucrat.check_these_tasks_were_run_successfully(task_name)
		x_mpv = pandas.read_pickle(bureaucrat.path_to_directory_of_task(task_name)/'x_mpv.pickle')
		fits_results = pandas.read_pickle(bureaucrat.path_to_directory_of_task(task_name)/'fits_results.pickle')
		return x_mpv, fits_results
	elif bureaucrat.was_task_run_successfully('beta_scan_sweeping_bias_voltage'):
		fits_results = []
		x_mpv = []
		for subrun in bureaucrat.list_subruns_of_task('beta_scan_sweeping_bias_voltage'):
			x, f = read_MPV_data_from_Landau_fits(subrun, collected_charge_variable_name)
			for _ in [f,x]:
				_['run_name'] = subrun.run_name
				_.set_index('run_name',inplace=True,append=True)
			fits_results.append(f)
			x_mpv.append(x)
		fits_results = pandas.concat(fits_results)
		x_mpv = pandas.concat(x_mpv)
		return x_mpv, fits_results
	else:
		raise RuntimeError(f'Dont know how to read the Coulomb calibration factors in run {repr(bureaucrat.run_name)} located in {repr(str(bureaucrat.path_to_run_directory))}. ')

def collected_charge_in_Coulomb(bureaucrat:RunBureaucrat, collected_charge_variable_name:str, conversion_factor_to_divide_by:float, conversion_factor_to_divide_by_error:float=float('NaN')):
	task_name_where_to_look_for_data = f'fit_Landau_and_extract_MPV_sweeping_voltage_{collected_charge_variable_name.replace(" ","_")}'
	bureaucrat.check_these_tasks_were_run_successfully(task_name_where_to_look_for_data)
	
	with bureaucrat.handle_task(f'collected_charge_in_Coulomb_using_{collected_charge_variable_name.replace(" ","_")}') as employee:
		data = pandas.read_pickle(bureaucrat.path_to_directory_of_task(task_name_where_to_look_for_data)/'x_mpv.pickle')
		
		conversion_factor_to_divide_by = ufloat(conversion_factor_to_divide_by,conversion_factor_to_divide_by_error)
		data['Collected charge (C) ufloat'] = numpy.array([ufloat(val,err) for val,err in zip(data['nanmedian'],data['nanstd'])])/conversion_factor_to_divide_by
		data['Collected charge (C)'] = data['Collected charge (C) ufloat'].apply(lambda x: x.nominal_value)
		data['Collected charge (C) error'] = data['Collected charge (C) ufloat'].apply(lambda x: x.std_dev)
		
		data.drop(columns=['nanmedian','nanstd','kMAD','nanmean','Collected charge (C) ufloat'], inplace=True)
		
		save_dataframe(
			df = data,
			name = 'collected_charge_vs_bias_voltage',
			location = employee.path_to_directory_of_my_task,
		)
		
		for variable in set(data.index.get_level_values('variable')):
			fig = line(
				title = f'Collected charge using {collected_charge_variable_name}<br><sup>{bureaucrat.run_name}</sup>',
				data_frame = data.reset_index(drop=False).query(f'variable=="{variable}"').sort_values(['Bias voltage (V) mean','signal_name']),
				x = 'Bias voltage (V) mean',
				y = 'Collected charge (C)',
				error_x = 'Bias voltage (V) std',
				error_y = 'Collected charge (C) error',
				color = 'signal_name',
				markers = True,
				labels = {
					'Bias voltage (V) mean': 'Bias voltage (V)',
				}
			)
			fig.update_layout(
				xaxis = dict(autorange = "reversed"), 
				yaxis = dict(type='log'),
			)
			fig.write_html(employee.path_to_directory_of_my_task/f'charge vs bias voltage.html', include_plotlyjs='cdn')

if __name__ == '__main__':
	
	import argparse
	
	set_my_template_as_default()
	
	parser = argparse.ArgumentParser()
	parser.add_argument('--dir',
		metavar = 'path', 
		help = 'Path to the base measurement directory.',
		required = True,
		dest = 'directory',
		type = str,
	)
	parser.add_argument('--n_bootstraps',
		metavar = 'N', 
		help = 'Number of fittings to perform to accumulate statistics. Default is 0. Higher values can take long time to compute.',
		default = 0,
		dest = 'n_bootstraps',
		type = int,
	)
	parser.add_argument(
		'--force',
		help = 'If this flag is passed, it will force the calculation even if it was already done beforehand. Old data will be deleted.',
		required = False,
		dest = 'force',
		action = 'store_true'
	)
	
	args = parser.parse_args()
	
	bureaucrat = RunBureaucrat(Path(args.directory))
	subruns = bureaucrat.list_subruns_of_task('beta_scan_sweeping_bias_voltage')
	SIGNALS_NAMES = {'DUT'}
	
	CALIBRATION_FACTORS_COULOMB = {
		'Amplitude (V)': {'val': 5.67e12, 'err': .25e12},
		'Collected charge (V s)': {'val': 5050, 'err': 250},
	}
	
	for variable in ['Amplitude (V)']:
		with warnings.catch_warnings():
			warnings.filterwarnings("ignore", 'Your `task_name` is ')
			fit_Landau_and_extract_MPV_sweeping_voltage(
				bureaucrat = bureaucrat,
				time_from_trigger_background = {subrun.run_name:{_:(1e-9,1.5e-9) for _ in SIGNALS_NAMES} for subrun in subruns},
				time_from_trigger_signal = {subrun.run_name:{_:(1.5e-9,1.83e-9) for _ in SIGNALS_NAMES} for subrun in subruns},
				signal_name_trigger = 'MCP-PMT',
				n_bootstraps = args.n_bootstraps,
				force = args.force,
				collected_charge_variable_name = variable,
				number_of_processes = max(multiprocessing.cpu_count()-2,1),
			)
			collected_charge_in_Coulomb(
				bureaucrat = bureaucrat,
				collected_charge_variable_name = variable,
				conversion_factor_to_divide_by = CALIBRATION_FACTORS_COULOMB[variable]['val'],
				conversion_factor_to_divide_by_error = CALIBRATION_FACTORS_COULOMB[variable]['err'],
			)
