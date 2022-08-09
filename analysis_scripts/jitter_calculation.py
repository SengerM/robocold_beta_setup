import pandas
from bureaucrat.SmarterBureaucrat import NamedTaskBureaucrat # https://github.com/SengerM/bureaucrat
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from grafica.plotly_utils.utils import scatter_histogram
from scipy.stats import median_abs_deviation
from scipy.optimize import curve_fit
from huge_dataframe.SQLiteDataFrame import load_whole_dataframe # https://github.com/SengerM/huge_dataframe
import shutil

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

def plot_cfd(jitter_df):
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
		optimum_constant_fraction_discriminator_thresholds = df[col].idxmin()
		fig.add_trace(
			go.Scatter(
				x = [optimum_constant_fraction_discriminator_thresholds[0]],
				y = [optimum_constant_fraction_discriminator_thresholds[1]],
				mode = 'markers',
				hovertext = [f'<b>Minimum</b><br>{pivot_table_df.index.name}: {optimum_constant_fraction_discriminator_thresholds[0]:.0f} %<br>{pivot_table_df.columns.name}: {optimum_constant_fraction_discriminator_thresholds[1]:.0f} %<br>{jitter_column_name} {col}: {df.loc[optimum_constant_fraction_discriminator_thresholds,col]*1e12:.2f} ps'],
				hoverinfo = 'text',
				marker = dict(
					color = '#61ff5c',
				),
				name = '',
			)
		)
		fig.update_yaxes(
			scaleanchor = "x",
			scaleratio = 1,
		)
		fig.update_layout(
			xaxis_title = pivot_table_df.index.name,
			yaxis_title = pivot_table_df.columns.name,
		)
		figs[col] = fig
	return figs

def jitter_calculation_beta_scan_single_voltage(path_to_measurement_base_directory:Path, force:bool=False):
	Norberto = NamedTaskBureaucrat(
		path_to_measurement_base_directory,
		task_name = 'jitter_calculation_beta_scan_single_voltage',
		_locals = locals(),
	)
	
	Norberto.check_required_scripts_were_run_before('beta_scan.py')
	
	if force == False and Norberto.task_was_applied_without_errors(): # If this was already done, don't do it again...
		return
	
	with Norberto.do_your_magic():
		data_df = load_whole_dataframe(Norberto.path_to_output_directory_of_script_named('beta_scan.py')/Path('parsed_from_waveforms.sqlite'))
		
		if Norberto.check_required_scripts_were_run_before('clean_beta_scan.py', raise_error=False): # If there was a cleaning done, let's take it into account...
			shutil.copyfile( # Put a copy of the cuts in the output directory so there is a record of what was done.
				Norberto.path_to_output_directory_of_script_named('clean_beta_scan.py')/Path('cuts.backup.csv'),
				Norberto.path_to_default_output_directory/Path('cuts_that_were_applied.csv')
			)
			data_df = data_df.merge(
				right = pandas.read_feather(Norberto.path_to_output_directory_of_script_named('clean_beta_scan.py')/Path('result.fd')).set_index('n_trigger'),
				left_index = True, 
				right_index = True
			)
			data_df = data_df.query('is_background==False').copy().drop('is_background',axis=1) # From now on we drop all background triggers.
		
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
			
			optimum_constant_fraction_discriminator_thresholds = jitter_df.idxmin()
			
			optimum_constant_fraction_discriminator_thresholds = optimum_constant_fraction_discriminator_thresholds[('Δt (s)',STATISTIC_TO_USE_FOR_THE_FINAL_JITTER_CALCULATION)]
			
			constant_fraction_discriminator_thresholds_to_use_for_the_jitter = optimum_constant_fraction_discriminator_thresholds
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
				figs = plot_cfd(jitter_df)
				for key,fig in figs.items():
					fig.update_layout(title=f'CFD jitter measured using {key} from Δt<br><sup>Measurement: {Norberto.measurement_name}</sup>')
				figs[STATISTIC_TO_USE_FOR_THE_FINAL_JITTER_CALCULATION].write_html(str(Norberto.path_to_default_output_directory/f'CFD jitter using {key}.html'), include_plotlyjs='cdn')
				
				fig = go.Figure()
				fig.update_layout(
					yaxis_title = 'count',
					xaxis_title = 'Δt (s)',
					title = f'Δt for {Δt_df.index.names[0]}={constant_fraction_discriminator_thresholds_to_use_for_the_jitter[0]} and {Δt_df.index.names[1]}={constant_fraction_discriminator_thresholds_to_use_for_the_jitter[1]}<br><sup>Measurement: {Norberto.measurement_name}</sup>'
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
					str(Norberto.path_to_default_output_directory/Path(f'Delta_t distribution and fit where jitter was obtained from.html')),
					include_plotlyjs = 'cdn',
				)

		jitter_results_df = pandas.DataFrame.from_records(jitter_results)
		df = jitter_results_df.set_index('measured_on')
		
		jitter = pandas.Series(
			{
				'Jitter (s)': df.loc['real data','Jitter (s)'],
				'Jitter (s) error': df['Jitter (s)'].std(),
			}
		)
		jitter.to_csv(Norberto.path_to_default_output_directory/'jitter.csv', header=False)
		
		fig = go.Figure()
		fig.update_layout(
			title = f'Statistics for the jitter<br><sup>Measurement: {Norberto.measurement_name}</sup>',
			xaxis_title = 'Jitter (s)',
			yaxis_title = 'count',
		)
		fig.add_trace(
			scatter_histogram(
				samples = df.loc['resampled data','Jitter (s)'],
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
			str(Norberto.path_to_default_output_directory/Path(f'histogram bootstrap.html')),
			include_plotlyjs = 'cdn',
		)

def jitter_calculation_beta_scan_sweeping_voltage(path_to_measurement_base_directory:Path, force_calculation_on_submeasurements:bool=False):
	Norberto = NamedTaskBureaucrat(
		path_to_measurement_base_directory,
		task_name = 'jitter_calculation_beta_scan_sweeping_voltage',
		_locals = locals(),
	)
	
	Norberto.check_required_tasks_were_run_before('beta_scan')
	
	with Norberto.do_your_magic():
		jitters = []
		for submeasurement_name, path_to_submeasurement in Norberto.find_submeasurements_of_task('beta_scan').items():
			jitter_calculation_beta_scan_single_voltage(
				path_to_measurement_base_directory = path_to_submeasurement, 
				force = force_calculation_on_submeasurements,
			)
			Raul = NamedTaskBureaucrat(path_to_submeasurement, task_name='no_task', _locals=locals())
			submeasurement_jitter = pandas.read_csv(
				Raul.path_to_output_directory_of_task_named('jitter_calculation_beta_scan_single_voltage')/'jitter.csv',
				names = ['variable_name','value'],
			)
			submeasurement_jitter.set_index('variable_name', inplace=True)
			submeasurement_jitter = submeasurement_jitter['value']
			submeasurement_jitter['submeasurement_name'] = submeasurement_name
			submeasurement_jitter['Bias voltage (V)'] = float(submeasurement_name.split('_')[-1].replace('V',''))
			jitters.append(submeasurement_jitter)
		
		jitter_df = pandas.DataFrame.from_records(jitters)
		jitter_df.columns.rename('', inplace=True)
		jitter_df.to_csv(Norberto.path_to_default_output_directory/'jitter_vs_bias_voltage.csv', index=False)
		
		fig = px.line(
			jitter_df.sort_values('Bias voltage (V)'),
			x = 'Bias voltage (V)',
			y = 'Jitter (s)',
			error_y = 'Jitter (s) error',
			markers = True,
			title = f'Jitter vs bias voltage<br><sup>Measurement: {Norberto.measurement_name}</sup>',
		)
		fig.write_html(
			str(Norberto.path_to_default_output_directory/'jitter_vs_bias_voltage.html'),
			include_plotlyjs = 'cdn',
		)

def script_core(path_to_measurement_base_directory:Path, force:bool=False):
	Nestor = NamedTaskBureaucrat(
		path_to_measurement_base_directory,
		task_name = 'jitter_calculation_script_core',
		_locals = locals(),
	)
	if Nestor.task_was_applied_without_errors('beta_scan_sweeping_bias_voltage'):
		jitter_calculation_beta_scan_sweeping_voltage(
			path_to_measurement_base_directory = path_to_measurement_base_directory,
			force_calculation_on_submeasurements = force,
		)
	elif Nestor.task_was_applied_without_errors('beta_scan'):
		jitter_calculation_beta_scan_single_voltage(
			path_to_measurement_base_directory = path_to_measurement_base_directory,
			force = force,
		)
	else:
		raise RuntimeError(f'Cannot process {path_to_measurement_base_directory} becasue I cannot find any of my known scripts to have ended successfully.')

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
		Path(args.directory), 
		force = args.force,
	)
