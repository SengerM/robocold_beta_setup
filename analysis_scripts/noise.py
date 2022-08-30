from the_bureaucrat.bureaucrats import RunBureaucrat # https://github.com/SengerM/the_bureaucrat
from pathlib import Path
import pandas
import plotly.express as px
import grafica.plotly_utils.utils # https://github.com/SengerM/grafica
import numpy
from huge_dataframe.SQLiteDataFrame import load_whole_dataframe # https://github.com/SengerM/huge_dataframe
from clean_beta_scan import tag_n_trigger_as_background_according_to_the_result_of_clean_beta_scan
from jitter_calculation import resample_measured_data
from grafica.plotly_utils.utils import scatter_histogram # https://github.com/SengerM/grafica
import warnings

grafica.plotly_utils.utils.set_my_template_as_default()

def noise_in_beta_scan(bureaucrat:RunBureaucrat, force:bool=False):
	Norberto = bureaucrat
	Norberto.check_these_tasks_were_run_successfully('beta_scan')
	
	TASK_NAME = 'noise_in_beta_scan'
	
	if force == False and Norberto.was_task_run_successfully(TASK_NAME): # If this was already done, don't do it again...
		return
	
	with Norberto.handle_task(TASK_NAME) as task_handler:
		data_df = load_whole_dataframe(Norberto.path_to_directory_of_task('beta_scan')/'parsed_from_waveforms.sqlite')
		if Norberto.was_task_run_successfully('clean_beta_scan'):
			data_df = tag_n_trigger_as_background_according_to_the_result_of_clean_beta_scan(Norberto, data_df).query('is_background==False').drop(columns='is_background')
		data_df.sort_index(inplace=True)
		
		noise_results = []
		for n_bootstrap in range(99):
			if n_bootstrap == 0:
				bootstrapped_iteration = False
			else:
				bootstrapped_iteration = True
			
			if not bootstrapped_iteration:
				df = data_df.copy()
			else: # if bootstrapped iteration
				df = resample_measured_data(data_df)
			df.index = df.index.droplevel('n_trigger')
			
			for signal_name in set(df.index.get_level_values('signal_name')):
				noise_results.append(
					{
						'measured_on': 'real data' if bootstrapped_iteration == False else 'resampled data',
						'Noise (V)': numpy.median(df.loc[signal_name,'Noise (V)']),
						'signal_name': signal_name,
					}
				)
			
		noise_results_df = pandas.DataFrame(noise_results).set_index(['signal_name','measured_on'])
		noise_results_df.sort_index(inplace=True)
		noise_final_results = []
		for signal_name in set(noise_results_df.index.get_level_values('signal_name')):
			noise_final_results.append(
				{
					'Noise (V)': noise_results_df.loc[(signal_name,'real data'),'Noise (V)'][0],
					'Noise (V) error': noise_results_df.loc[(signal_name,'resampled data'),'Noise (V)'].std(),
					'signal_name': signal_name,
				}
			)
		noise_final_results_df = pandas.DataFrame(noise_final_results).set_index('signal_name')
		noise_final_results_df.to_csv(task_handler.path_to_directory_of_my_task/'noise.csv')

def noise_vs_bias_voltage(bureaucrat:RunBureaucrat, force_calculation_on_submeasurements:bool=False):
	Romina = bureaucrat
	
	Romina.check_these_tasks_were_run_successfully('beta_scan_sweeping_bias_voltage')
	with Romina.handle_task('noise_vs_bias_voltage') as task_handler:
		collected_noises = []
		for submeasurement_name, path_to_submeasurement in Romina.list_subruns_of_task('beta_scan_sweeping_bias_voltage').items():
			Raúl = RunBureaucrat(path_to_submeasurement)
			noise_in_beta_scan(
				bureaucrat = Raúl,
				force = force_calculation_on_submeasurements,
			)
			submeasurement_noise = pandas.read_csv(Raúl.path_to_directory_of_task('noise_in_beta_scan')/'noise.csv')
			submeasurement_noise['measurement_name'] = submeasurement_name
			submeasurement_noise['Bias voltage (V)'] = float(submeasurement_name.split('_')[-1].replace('V',''))
			collected_noises.append(submeasurement_noise)
		noise_df = pandas.concat(collected_noises, ignore_index=True)
		
		noise_df.to_csv(
			task_handler.path_to_directory_of_my_task/'noise_vs_bias_voltage.csv',
			index = False,
		)
		
		fig = px.line(
			noise_df.sort_values(['Bias voltage (V)','signal_name']),
			x = 'Bias voltage (V)',
			y = 'Noise (V)',
			error_y = 'Noise (V) error',
			color = 'signal_name',
			title = f'Noise vs bias voltage<br><sup>Run: {Romina.run_name}</sup>',
			markers = True,
		)
		fig.write_html(
			str(task_handler.path_to_directory_of_my_task/'noise_vs_bias_voltage.html'),
			include_plotlyjs = 'cdn',
		)

def noise_vs_bias_voltage_comparison(bureaucrat:RunBureaucrat, force:bool=False):
	Spencer = bureaucrat
	
	Spencer.check_these_tasks_were_run_successfully('automatic_beta_scans')
	
	with Spencer.handle_task('noise_vs_bias_voltage_comparison') as Spencers_employee:
		noises = []
		for submeasurement_name, path_to_submeasurement in Spencers_employee.list_subruns_of_task('automatic_beta_scans').items():
			Raúl = RunBureaucrat(path_to_submeasurement)
			noise_vs_bias_voltage(
				bureaucrat = Raúl,
				force_calculation_on_submeasurements = force,
			)
			noise = pandas.read_csv(Raúl.path_to_directory_of_task('noise_vs_bias_voltage')/'noise_vs_bias_voltage.csv')
			noise['beta_scan_vs_bias_voltage'] = submeasurement_name
			noises.append(noise)
		df = pandas.concat(noises, ignore_index=True)
		
		df.to_csv(Spencers_employee.path_to_directory_of_my_task/'noise.csv', index=False)
		
		df['measurement_timestamp'] = df['beta_scan_vs_bias_voltage'].apply(lambda x: x.split('_')[0])
		fig = px.line(
			df.sort_values(['measurement_timestamp','Bias voltage (V)','signal_name']),
			x = 'Bias voltage (V)',
			y = 'Noise (V)',
			error_y = 'Noise (V) error',
			color = 'measurement_timestamp',
			facet_col = 'signal_name',
			markers = True,
			title = f'Noise comparison<br><sup>Run: {Spencer.run_name}</sup>',
			hover_data = ['beta_scan_vs_bias_voltage','measurement_name'],
			labels = {
				'measurement_name': 'Beta scan',
				'beta_scan_vs_bias_voltage': 'Beta scan vs bias voltage',
				'measurement_timestamp': 'Measurement timestamp',
			}
		)
		fig.write_html(
			str(Spencers_employee.path_to_directory_of_my_task/'noise_vs_bias_voltage_comparison.html'),
			include_plotlyjs = 'cdn',
		)

def script_core(bureaucrat:RunBureaucrat, force:bool):
	Manuel = bureaucrat
	if Manuel.was_task_run_successfully('automatic_beta_scans'):
		noise_vs_bias_voltage_comparison(Manuel, force=force)
	elif Manuel.was_task_run_successfully('beta_scan_sweeping_bias_voltage'):
		noise_vs_bias_voltage(Manuel, force_calculation_on_submeasurements=force)
	elif Manuel.was_task_run_successfully('beta_scan'):
		noise_in_beta_scan(Manuel, force=True)
	else:
		raise RuntimeError(f'Dont know how to process run {repr(Manuel.run_name)} located in `{Manuel.path_to_run_directory}`...')

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
		force = args.force,
	)
