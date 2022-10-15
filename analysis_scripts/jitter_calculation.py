import pandas
from the_bureaucrat.bureaucrats import RunBureaucrat # https://github.com/SengerM/the_bureaucrat
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from grafica.plotly_utils.utils import scatter_histogram
from scipy.stats import median_abs_deviation
from scipy.optimize import curve_fit
from huge_dataframe.SQLiteDataFrame import load_whole_dataframe # https://github.com/SengerM/huge_dataframe
import shutil
from clean_beta_scan import tag_n_trigger_as_background_according_to_the_result_of_clean_beta_scan
import multiprocessing
from summarize_parameters import read_summarized_data
import dominate # https://github.com/Knio/dominate

N_BOOTSTRAP = 99
STATISTIC_TO_USE_FOR_THE_FINAL_JITTER_CALCULATION = 'sigma_from_gaussian_fit' # For the time resolution I will use the `sigma_from_gaussian_fit` because in practice ends up being the most robust and reliable of all.

def kMAD(x,nan_policy='omit'):
	"""Calculates the median absolute deviation multiplied by 1.4826... 
	which should converge to the standard deviation for Gaussian distributions,
	but is much more robust to outliers than the std."""
	k_MAD_TO_STD = 1.4826 # https://en.wikipedia.org/wiki/Median_absolute_deviation#Relation_to_standard_deviation
	return k_MAD_TO_STD*median_abs_deviation(x,nan_policy=nan_policy)

def gaussian(x, mu, sigma, amplitude=1):
	return amplitude/sigma/(2*np.pi)**.5*np.exp(-((x-mu)/sigma)**2/2)

def fit_gaussian_to_samples(samples, bins='auto', nan_policy='drop'):
	if nan_policy == 'drop':
		samples = samples[~np.isnan(samples)]
	else:
		raise NotImplementedError(f'`nan_policy={nan_policy}` not implemented.')
	hist, bins_edges = np.histogram(
		samples,
		bins = bins,
	)
	x_values = bins_edges[:-1] + np.diff(bins_edges)[0]/2
	y_values = hist
	try:
		popt, pcov = curve_fit(
			gaussian,
			x_values,
			y_values,
			p0 = [np.median(samples),kMAD(samples),max(y_values)],
		)
		return popt[0], popt[1], popt[2]
	except RuntimeError: # This happens when the fit fails because there are very few samples.
		return float('NaN'),float('NaN'),float('NaN')

def resample_measured_data(data_df):
	"""Produce a new sample of `data_df` using the value of `n_trigger`
	to group rows. Returns a new data frame of the same size."""
	resampled_df = data_df.reset_index(drop=False).pivot(
		index = 'n_trigger',
		columns = 'signal_name',
		values = set(data_df.columns),
	)
	resampled_df = resampled_df.sample(frac=1, replace=True)
	resampled_df = resampled_df.stack()
	return resampled_df

def calculate_Δt(data_df:pandas.DataFrame)->pandas.DataFrame:
	"""Calculate the time difference between each `t_whatever (s)` column
	between two signals.
	
	Arguments
	---------
	data_df: pandas.DataFrame
		A data frame with index `('n_trigger','signal_name')` and columns
		`('t_10 (s)','t_20 (s)',...,'t_90 (s)')`.
	
	Returns
	-------
	Δt: pandas.DataFrame
		A data frame of the form:
	```
													   Δt (s)
	n_trigger k_DUT (%) k_reference_trigger (%)              
	0         10        10                       2.668779e-09
	1         10        10                       2.747474e-09
	2         10        10                       2.749357e-09
	3         10        10                       2.724985e-09
	4         10        10                       2.651759e-09
	...                                                   ...
	6661      90        90                       3.121407e-09
	6662      90        90                       2.960962e-09
	6663      90        90                       3.046798e-09
	6664      90        90                       3.003820e-09
	6665      90        90                       2.953823e-09
	```
	where the columns `"k_{whatever} (%)"` indicate the constant fraction 
	discriminator value and get their names from the values in the 
	`"signal_name"` column from the input data frame. For this example 
	`data_df` was
	```
									 t_10 (s)      t_20 (s)      t_30 (s)  ...      t_70 (s)      t_80 (s)      t_90 (s)
	n_trigger signal_name                                                  ...                                          
	0         reference_trigger  2.538063e-08  2.543204e-08  2.546866e-08  ...  2.559847e-08  2.565264e-08  2.572007e-08
			  DUT                2.804941e-08  2.814947e-08  2.823715e-08  ...  2.849985e-08  2.857490e-08  2.865827e-08
	1         reference_trigger  2.546233e-08  2.551147e-08  2.554806e-08  ...  2.565830e-08  2.568933e-08  2.572637e-08
			  DUT                2.820980e-08  2.831695e-08  2.839530e-08  ...  2.863696e-08  2.870580e-08  2.878720e-08
	2         reference_trigger  2.530641e-08  2.536126e-08  2.540537e-08  ...  2.552016e-08  2.554677e-08  2.558508e-08
	...                                   ...           ...           ...  ...           ...           ...           ...
	6663      DUT                2.793967e-08  2.805513e-08  2.813528e-08  ...  2.839191e-08  2.845853e-08  2.853970e-08
	6664      reference_trigger  2.547273e-08  2.552157e-08  2.555375e-08  ...  2.567229e-08  2.570345e-08  2.574207e-08
			  DUT                2.814452e-08  2.824212e-08  2.832217e-08  ...  2.857825e-08  2.865634e-08  2.874589e-08
	6665      reference_trigger  2.525602e-08  2.532906e-08  2.536762e-08  ...  2.550622e-08  2.553539e-08  2.557395e-08
			  DUT                2.789984e-08  2.801477e-08  2.810459e-08  ...  2.839165e-08  2.845277e-08  2.852777e-08
	```
	"""
	
	if data_df.index.names != ['n_trigger','signal_name']:
		raise ValueError(f"I am expecting a data frame with a multi index with columns `['n_trigger','signal_name']`, but instead received one with columns `{data_df.index.names}`")
	TIME_THRESHOLD_COLUMNS = {f't_{i} (s)' for i in [10,20,30,40,50,60,70,80,90]}
	if not TIME_THRESHOLD_COLUMNS.issubset(set(data_df.columns)):
		raise ValueError(f'I am expecting a data frame with the columns `{sorted(TIME_THRESHOLD_COLUMNS)}`, but at least one of them is not in `data_df`, which columns are {sorted(data_df.columns)}.')
	set_of_signal_names = set(data_df.index.get_level_values('signal_name'))
	if len(set_of_signal_names) != 2:
		raise ValueError(f'Cannot calculate Δt in a data frame that does not have exactly two signals. The data frame you gave me has signal names {sorted(set_of_signal_names)}. I need exactly two, not {len(set_of_signal_names)}.')

	df = data_df.copy() # Don't want to touch the original...
	signal_1_name = sorted(set_of_signal_names)[0]
	signal_2_name = sorted(set_of_signal_names)[1]

	Δts_list = []
	for k1 in [10,20,30,40,50,60,70,80,90]:
		for k2 in [10,20,30,40,50,60,70,80,90]:
			t1 = df.loc[pandas.IndexSlice[:, signal_1_name], f't_{k1} (s)']
			t2 = df.loc[pandas.IndexSlice[:, signal_2_name], f't_{k2} (s)']
			for t in [t1,t2]:
				t.index = t.index.droplevel('signal_name')
			Δt = t1 - t2
			Δt.rename('Δt (s)', inplace=True)
			Δt = Δt.to_frame()
			Δt[f'k_{signal_1_name} (%)'] = k1
			Δt[f'k_{signal_2_name} (%)'] = k2
			Δt.reset_index(drop=False, inplace=True)
			Δt.set_index(['n_trigger',f'k_{signal_1_name} (%)',f'k_{signal_2_name} (%)'], inplace=True)
			Δts_list.append(Δt)
	Δt = pandas.concat(Δts_list)
	return Δt

def sigma_from_gaussian_fit(x, nan_policy='drop')->float:
	"""Estimates the standard deviation of samples `x` by fitting a 
	Gaussian function and getting the value of sigma."""
	_, sigma, _ = fit_gaussian_to_samples(samples=x, bins='auto', nan_policy=nan_policy)
	return sigma

def plot_cfd(jitter_df, constant_fraction_discriminator_thresholds_to_use_for_the_jitter):
	"""Plot the color map of the constant fraction discriminator thresholds
	and the resulting jitter.
	
	Arguments
	---------
	jitter_df: pandas.DataFrame
		A data frame of the form
		```
												 Δt (s)                                      
												   kMAD           std sigma_from_gaussian_fit
		k_DUT (%) k_reference_trigger (%)                                                    
		10        10                       4.090410e-11  5.364598e-11            3.973729e-11
				  20                       4.071987e-11  5.819378e-11            3.919612e-11
				  30                       4.097219e-11  5.862690e-11            3.929521e-11
				  40                       4.152937e-11  5.897231e-11            3.960742e-11
				  50                       4.160544e-11  5.931032e-11            3.989404e-11
		...                                         ...           ...                     ...
		90        50                       4.064725e-11  6.696063e-11            3.901498e-11
				  60                       4.137183e-11  6.756106e-11            3.955964e-11
				  70                       4.178150e-11  6.808789e-11            4.025378e-11
				  80                       4.266790e-11  6.865003e-11            4.105474e-11
				  90                       4.437570e-11  6.945268e-11            4.225482e-11
		```
		The number of "sub columns" of the column "Δt (s)" is arbitrary,
		for each subcolumn a color map plot will be produced. The names
	"""
	
	jitter_column_name = jitter_df.columns[0][0]
	df = jitter_df.copy()
	df.columns = df.columns.droplevel()
	
	figs = {}
	for col in df.columns:
		pivot_table_df = pandas.pivot_table(
			df,
			values = col,
			index = jitter_df.index.names[0],
			columns = jitter_df.index.names[1],
			aggfunc = np.mean,
		)
		fig = go.Figure(
			data = go.Contour(
				z = pivot_table_df.to_numpy(),
				x = pivot_table_df.index,
				y = pivot_table_df.columns,
				contours = dict(
					coloring ='heatmap',
					showlabels = True, # show labels on contours
				),
				colorbar = dict(
					title = f'{jitter_column_name} {col}',
					titleside = 'right',
				),
				hovertemplate = f'{pivot_table_df.index.name}: %{{x:.0f}} %<br>{pivot_table_df.columns.name}: %{{y:.0f}} %<br>{jitter_column_name} {col}: %{{z:.2e}}',
				name = '',
			),
		)
		fig.add_trace(
			go.Scatter(
				x = [constant_fraction_discriminator_thresholds_to_use_for_the_jitter[1]],
				y = [constant_fraction_discriminator_thresholds_to_use_for_the_jitter[0]],
				mode = 'markers',
				hovertext = [f'<b>Minimum</b><br>{pivot_table_df.index.name}: {constant_fraction_discriminator_thresholds_to_use_for_the_jitter[0]:.0f} %<br>{pivot_table_df.columns.name}: {constant_fraction_discriminator_thresholds_to_use_for_the_jitter[1]:.0f} %<br>{jitter_column_name} {col}: {df.loc[constant_fraction_discriminator_thresholds_to_use_for_the_jitter,col]*1e12:.2f} ps'],
				hoverinfo = 'text',
				marker = dict(
					color = '#70ff96',
					line_width = 2,
					line_color = '#000000',
					symbol = 'x',
					size = 22,
				),
				name = '',
			)
		)
		fig.update_yaxes(
			scaleanchor = "x",
			scaleratio = 1,
		)
		fig.update_layout(
			xaxis_title = pivot_table_df.columns.name,
			yaxis_title = pivot_table_df.index.name,
		)
		figs[col] = fig
	return figs

def jitter_calculation_beta_scan(bureaucrat:RunBureaucrat, CFD_thresholds='best', force:bool=False):
	"""Calculates the jitter from a beta scan between two devices.
	
	Arguments
	---------
	bureaucrat: RunBureaucrat
		A bureaucrat to handle this run.
	CFD_thresholds: str or dict, default `'best'`
		If `'best'`, then the CFD thresholds are chosen such that the
		jitter is minimized. If a dictionary, it must be of the form:
		`{signal_name_1: float, signal_name_2: float}` where each `signal_name`
		is a string with the name of the signal and each `float` is a 
		number between 0 and 100 specifying the CFD threshold height in 
		percentage, e.g. `{'DUT': 20, 'reference_trigger': 50}`.
	force: bool, default `True`
		If `True` the calculation is done, no matter if it was done before
		or not. If `False` the calculation will be done only if there is
		no previous calculation of it.
	"""
	
	Norberto = bureaucrat
	
	Norberto.check_these_tasks_were_run_successfully('beta_scan')
	
	if not (CFD_thresholds != 'best' or not isinstance(CFD_thresholds, dict)):
		raise ValueError('Wrong value for `CFD_thresholds`, please read the documentation of this function.')
	
	TASK_NAME = 'jitter_calculation_beta_scan'
	
	if force == False and Norberto.was_task_run_successfully(TASK_NAME): # If this was already done, don't do it again...
		return
	
	with Norberto.handle_task(TASK_NAME) as Norbertos_employee:
		data_df = load_whole_dataframe(Norberto.path_to_directory_of_task('beta_scan')/'parsed_from_waveforms.sqlite')
		
		if Norberto.check_these_tasks_were_run_successfully('clean_beta_scan', raise_error=False): # If there was a cleaning done, let's take it into account...
			shutil.copyfile( # Put a copy of the cuts in the output directory so there is a record of what was done.
				Norberto.path_to_directory_of_task('clean_beta_scan')/Path('cuts.backup.csv'),
				Norbertos_employee.path_to_directory_of_my_task/'cuts_that_were_applied.csv',
			)
			data_df = tag_n_trigger_as_background_according_to_the_result_of_clean_beta_scan(Norberto, data_df).query('is_background==False').drop(columns='is_background')
		
		# I always expect that there are only two measured signals from the beta setup, let's check it:
		set_of_measured_signals = set(data_df.index.get_level_values('signal_name'))
		if len(set_of_measured_signals) == 2:
			signal_names = set_of_measured_signals
		else:
			raise RuntimeError(f'A time resolution calculation requires two signals (DUT and reference, or two DUTs), but this beta scan has the following signal names `{set_of_measured_signals}`. Dont know how to handle this, sorry dude...')
		
		jitter_results = []
		bootstrapped_replicas_data = []
		for k_bootstrap in range(N_BOOTSTRAP+1):
			bootstrapped_iteration = False
			if k_bootstrap > 0:
				bootstrapped_iteration = True

			if bootstrapped_iteration == False:
				df = data_df.copy()
			else:
				df = resample_measured_data(data_df)

			Δt_df = calculate_Δt(df)
			Δt_df.index = Δt_df.index.droplevel('n_trigger')
			jitter_df = Δt_df.groupby(level=Δt_df.index.names).agg(func=(kMAD, np.std, sigma_from_gaussian_fit))
			
			if CFD_thresholds == 'best':
				optimum_constant_fraction_discriminator_thresholds = jitter_df.idxmin()
				optimum_constant_fraction_discriminator_thresholds = optimum_constant_fraction_discriminator_thresholds[('Δt (s)',STATISTIC_TO_USE_FOR_THE_FINAL_JITTER_CALCULATION)]
				constant_fraction_discriminator_thresholds_to_use_for_the_jitter = optimum_constant_fraction_discriminator_thresholds
			else: # It is a dictionary specifying the threshold for each signal.
				set_of_signals_in_this_measurement = set(data_df.index.get_level_values('signal_name'))
				if set_of_signals_in_this_measurement != set(CFD_thresholds.keys()):
					raise ValueError(f'`CFD_thresholds` specifies signal names that are not found in the data. According to the data the signal names are {set_of_signals_in_this_measurement} and `CFD_thresholds` specifies signal names {set(CFD_thresholds.keys())}.')
				if any([not 0 <=CFD_thresholds[s] <= 100 for s in set_of_signals_in_this_measurement]):
					raise ValueError(f'`CFD_thresholds` contains values outside the range from 0 to 100, which is wrong. Received `CFD_thresholds = {CFD_thresholds}`.')
				constant_fraction_discriminator_thresholds_to_use_for_the_jitter = tuple([CFD_thresholds[k_name[2:-4]] for k_name in jitter_df.index.names]) # The `k_name` is e.g. "'k_DUT (%)'" and here we want just `"DUT"`.
			
			jitter_final_number = jitter_df.loc[constant_fraction_discriminator_thresholds_to_use_for_the_jitter,('Δt (s)',STATISTIC_TO_USE_FOR_THE_FINAL_JITTER_CALCULATION)]
			
			jitter_results.append(
				{
					'measured_on': 'real data' if bootstrapped_iteration == False else 'resampled data',
					'Jitter (s)': jitter_final_number,
					jitter_df.index.names[0]: constant_fraction_discriminator_thresholds_to_use_for_the_jitter[0],
					jitter_df.index.names[1]: constant_fraction_discriminator_thresholds_to_use_for_the_jitter[1],
				}
			)
			
			if bootstrapped_iteration == True:
				continue
			else: # Do some plots
				figs = plot_cfd(jitter_df, constant_fraction_discriminator_thresholds_to_use_for_the_jitter)
				for key,fig in figs.items():
					fig.update_layout(title=f'Jitter vs k<sub>CFD</sub><br><sup>Run: {Norberto.run_name}</sup>')
				figs[STATISTIC_TO_USE_FOR_THE_FINAL_JITTER_CALCULATION].write_html(str(Norbertos_employee.path_to_directory_of_my_task/f'CFD jitter using {key}.html'), include_plotlyjs='cdn')
				
				fig = go.Figure()
				fig.update_layout(
					yaxis_title = 'count',
					xaxis_title = 'Δt (s)',
					title = f'Δt for {Δt_df.index.names[0]}={constant_fraction_discriminator_thresholds_to_use_for_the_jitter[0]} and {Δt_df.index.names[1]}={constant_fraction_discriminator_thresholds_to_use_for_the_jitter[1]}<br><sup>Run: {Norberto.run_name}</sup>'
				)
				selected_Δt_samples = Δt_df.loc[constant_fraction_discriminator_thresholds_to_use_for_the_jitter]
				selected_Δt_samples = selected_Δt_samples.to_numpy()
				selected_Δt_samples = selected_Δt_samples[~np.isnan(selected_Δt_samples)] # Remove NaN values because otherwise the histograms complain...
				fig.add_trace(
					scatter_histogram(
						samples = selected_Δt_samples,
						name = f'Measured data',
						error_y = dict(type='auto'),
					)
				)
				fitted_mu, fitted_sigma, fitted_amplitude = fit_gaussian_to_samples(selected_Δt_samples)
				x_axis_values = sorted(list(np.linspace(min(selected_Δt_samples),max(selected_Δt_samples),99)) + list(np.linspace(fitted_mu-5*fitted_sigma,fitted_mu+5*fitted_sigma,99)))
				fig.add_trace(
					go.Scatter(
						x = x_axis_values,
						y = gaussian(x_axis_values, fitted_mu, fitted_sigma, fitted_amplitude),
						name = f'Fitted Gaussian (σ={fitted_sigma*1e12:.2f} ps)',
					)
				)
				fig.add_trace(
					go.Scatter(
						x = [fitted_mu, fitted_mu+jitter_final_number],
						y = 2*[gaussian(x_axis_values, fitted_mu, fitted_sigma, fitted_amplitude).max()*.6],
						name = f'Jitter ({jitter_final_number*1e12:.2f} ps)',
						mode = 'lines+markers',
						hoverinfo = 'none',
					)
				)
				fig.write_html(
					str(Norbertos_employee.path_to_directory_of_my_task/Path(f'Delta_t distribution and fit where jitter was obtained from.html')),
					include_plotlyjs = 'cdn',
				)

		jitter_results = pandas.DataFrame.from_records(jitter_results)
		jitter_results.set_index('measured_on', inplace=True)
		
		jitter = pandas.Series(
			{
				'Jitter (s)': jitter_results.loc['real data','Jitter (s)'],
				'Jitter (s) error': jitter_results['Jitter (s)'].std(),
				'signals_names': tuple(sorted(set_of_measured_signals)),
				jitter_df.index.names[0]: constant_fraction_discriminator_thresholds_to_use_for_the_jitter[0],
				jitter_df.index.names[1]: constant_fraction_discriminator_thresholds_to_use_for_the_jitter[1],
			}
		)
		jitter.to_pickle(Norbertos_employee.path_to_directory_of_my_task/'jitter.pickle')
		
		fig = go.Figure()
		fig.update_layout(
			title = f'Statistics for the jitter<br><sup>Run: {Norberto.run_name}</sup>',
			xaxis_title = 'Jitter (s)',
			yaxis_title = 'count',
		)
		fig.add_trace(
			scatter_histogram(
				samples = jitter_results.loc['resampled data','Jitter (s)'],
				name = 'Bootstrapped jitter replicas',
				error_y = dict(type='auto'),
			)
		)
		fig.add_vline(
			x = jitter['Jitter (s)'],
			annotation_text = f"Jitter = ({jitter['Jitter (s)']*1e12:.2f}±{jitter['Jitter (s) error']*1e12:.2f}) ps", 
			annotation_position = "bottom left",
			annotation_textangle = -90,
		)
		fig.add_vrect(
			x0 = jitter['Jitter (s)'] - jitter['Jitter (s) error'],
			x1 = jitter['Jitter (s)'] + jitter['Jitter (s) error'],
			opacity = .1,
			line_width = 0,
			fillcolor = 'black',
		)
		fig.write_html(
			str(Norbertos_employee.path_to_directory_of_my_task/Path(f'histogram bootstrap.html')),
			include_plotlyjs = 'cdn',
		)

def jitter_calculation_beta_scan_sweeping_voltage(bureaucrat:RunBureaucrat, CFD_thresholds='best', force_calculation_on_submeasurements:bool=False, number_of_processes:int=1):
	Norberto = bureaucrat
	
	Norberto.check_these_tasks_were_run_successfully('beta_scan_sweeping_bias_voltage')
	
	with Norberto.handle_task('jitter_calculation_beta_scan_sweeping_voltage') as Norbertos_employee:
		subruns = Norberto.list_subruns_of_task('beta_scan_sweeping_bias_voltage')
		with multiprocessing.Pool(number_of_processes) as p:
			p.starmap(
				jitter_calculation_beta_scan,
				[(bur,thrshld,frc) for bur,thrshld,frc in zip(subruns, [CFD_thresholds]*len(subruns), [force_calculation_on_submeasurements]*len(subruns))]
			)
		
		jitter = []
		for Raúl in Norberto.list_subruns_of_task('beta_scan_sweeping_bias_voltage'):
			Raúl.check_these_tasks_were_run_successfully(['jitter_calculation_beta_scan','summarize_beta_scan_measured_stuff'])
			_ = pandas.read_pickle(Raúl.path_to_directory_of_task('jitter_calculation_beta_scan')/'jitter.pickle')
			_['run_name'] = Raúl.run_name
			jitter.append(_)
		jitter = pandas.DataFrame.from_records(jitter)
		jitter.set_index('run_name',drop=True,inplace=True)
		
		signals_names = set(jitter['signals_names'])
		if len(signals_names) != 1 or len(list(signals_names)[0]) != 2:
			raise RuntimeError(f'There is a weird error with the signals, check this!')
		signals_names = list(signals_names)[0]
		
		jitter.drop(columns='signals_names',inplace=True)
		
		summary = read_summarized_data(bureaucrat)
		summary.columns = [f'{col[0]} {col[1]}' for col in summary.columns]
		
		measured_device = set(summary.index.get_level_values('device_name'))
		if len(measured_device) != 1:
			raise RuntimeError(f'I was expecting only a single device measured, but found {len(measured_device)} devices measured, check this!')
		measured_device = list(measured_device)[0]
		
		summary.reset_index(level='device_name', inplace=True, drop=False)
		
		jitter.to_pickle(Norbertos_employee.path_to_directory_of_my_task/'jitter.pickle')
		
		fig = px.line(
			title = f'Jitter vs bias voltage<br><sup>Run: {Norberto.run_name}</sup>',
			data_frame = jitter.join(summary).sort_values('Bias voltage (V) mean'),
			x = 'Bias voltage (V) mean',
			y = 'Jitter (s)',
			error_y = 'Jitter (s) error',
			error_x = 'Bias voltage (V) std',
			markers = True,
		)
		fig.update_layout(xaxis = dict(autorange = "reversed"))
		fig.write_html(
			str(Norbertos_employee.path_to_directory_of_my_task/'jitter_vs_bias_voltage.html'),
			include_plotlyjs = 'cdn',
		)
		
		k_CFD = jitter[[col for col in jitter.columns if col[:2]=='k_' and col[-4:]==' (%)']]
		k_CFD = k_CFD.stack()
		k_CFD.rename('k_CFD (%)',inplace=True)
		k_CFD = k_CFD.to_frame()
		k_CFD.index.set_names(k_CFD.index.names[:-1] + ['CFD'], inplace=True)
		
		fig = px.line(
			title = f'Jitter vs bias voltage<br><sup>Run: {Norberto.run_name}</sup>',
			data_frame = k_CFD.reset_index(drop=False).sort_values('run_name'),
			x = 'run_name',
			y = 'k_CFD (%)',
			color = 'CFD',
			markers = True,
		)
		fig.update_yaxes(range=[0,100])
		fig.write_html(
			str(Norbertos_employee.path_to_directory_of_my_task/'CFD_vs_bias_voltage.html'),
			include_plotlyjs = 'cdn',
		)
		
		# Collect all plots in documents so it is easy to view them all together.
		path_to_subplots = []
		plot_types = [p.stem for p in (Norberto.list_subruns_of_task('beta_scan_sweeping_bias_voltage')[0].path_to_directory_of_task('jitter_calculation_beta_scan')).iterdir() if p.suffix == '.html']
		for plot_type in plot_types:
			for subrun in Norbertos_employee.list_subruns_of_task('beta_scan_sweeping_bias_voltage'):
				path_to_the_plot = Path('../..')/(subrun.path_to_directory_of_task('jitter_calculation_beta_scan')/f'{plot_type}.html').relative_to(Norberto.path_to_run_directory)
				path_to_subplots.append(
					{
						'plot_type': plot_type,
						'path_to_plot': path_to_the_plot,
						'run_name': subrun.run_name,
					}
				)
		path_to_subplots_df = pandas.DataFrame(path_to_subplots).set_index('plot_type')
		for plot_type in set(path_to_subplots_df.index.get_level_values('plot_type')):
			document_title = f'{plot_type} plots from beta_scan_sweeping_bias_voltage {Norberto.run_name}'
			html_doc = dominate.document(title=document_title)
			with html_doc:
				dominate.tags.h1(document_title)
				with dominate.tags.div(style='display: flex; flex-direction: column; width: 100%;'):
					for idx,row in path_to_subplots_df.loc[plot_type].sort_values('run_name').iterrows():
						dominate.tags.iframe(src=str(row['path_to_plot']), style=f'height: 100vh; min-height: 600px; width: 100%; min-width: 600px; border-style: none;')
			path_for_saving_plots_all_together = Norbertos_employee.path_to_directory_of_my_task/f'plots_of_subruns'
			path_for_saving_plots_all_together.mkdir(exist_ok=True)
			with open(path_for_saving_plots_all_together/f'{plot_type} together.html', 'w') as ofile:
				print(html_doc, file=ofile)

def script_core(bureaucrat:RunBureaucrat, CFD_thresholds, force:bool=False):
	Nestor = bureaucrat
	if Nestor.was_task_run_successfully('beta_scan_sweeping_bias_voltage'):
		jitter_calculation_beta_scan_sweeping_voltage(
			bureaucrat = Nestor,
			CFD_thresholds = CFD_thresholds,
			force_calculation_on_submeasurements = force,
			number_of_processes = max(multiprocessing.cpu_count()-1,1),
		)
	elif Nestor.was_task_run_successfully('beta_scan'):
		jitter_calculation_beta_scan(
			bureaucrat = Nestor,
			CFD_thresholds = CFD_thresholds,
			force = True,
		)
	elif Nestor.was_task_run_successfully('automatic_beta_scans'):
		for b in Nestor.list_subruns_of_task('automatic_beta_scans'):
			jitter_calculation_beta_scan_sweeping_voltage(
				bureaucrat = b,
				CFD_thresholds = CFD_thresholds,
				force_calculation_on_submeasurements = force,
				number_of_processes = max(multiprocessing.cpu_count()-1,1),
			)
	else:
		raise RuntimeError(f'Cannot process run {repr(Nestor.run_name)} becasue I cannot find any of my known scripts to have ended successfully.')

if __name__ == '__main__':
	import argparse

	parser = argparse.ArgumentParser()
	parser.add_argument(
		'--dir',
		metavar = 'path',
		help = 'Path to the base directory of a measurement.',
		required = True,
		dest = 'directory',
		type = str,
	)
	parser.add_argument(
		'--force',
		help = 'If this flag is passed, it will force the calculation even if it was already done beforehand. Old data will be deleted.',
		required = False,
		dest = 'force',
		action = 'store_true'
	)
	args = parser.parse_args()
	script_core(
		RunBureaucrat(Path(args.directory)),
		CFD_thresholds = 'best',
		force = args.force,
	)
