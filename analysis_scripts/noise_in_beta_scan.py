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
from summarize_parameters import read_summarized_data
import multiprocessing

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
		noise_final_results_df.to_pickle(task_handler.path_to_directory_of_my_task/'noise.pickle')

def read_noise_in_beta_scan(bureaucrat:RunBureaucrat):
	if bureaucrat.was_task_run_successfully('beta_scan'):
		bureaucrat.check_these_tasks_were_run_successfully('noise_in_beta_scan')
		return pandas.read_pickle(bureaucrat.path_to_directory_of_task('noise_in_beta_scan')/'noise.pickle')
	elif bureaucrat.was_task_run_successfully('beta_scan_sweeping_bias_voltage'):
		noise = []
		for subrun in bureaucrat.list_subruns_of_task('beta_scan_sweeping_bias_voltage'):
			_ = read_noise_in_beta_scan(subrun)
			_['run_name'] = subrun.run_name
			_.set_index('run_name',inplace=True,append=True)
			noise.append(_)
		noise = pandas.concat(noise)
		return noise
	else:
		raise RuntimeError(f'Dont know how to read the noise in run {repr(bureaucrat.run_name)} located in {repr(str(bureaucrat.path_to_run_directory))}')

def noise_vs_bias_voltage(bureaucrat:RunBureaucrat, force_calculation_on_submeasurements:bool=False, number_of_processes:int=1):
	Romina = bureaucrat
	
	Romina.check_these_tasks_were_run_successfully('beta_scan_sweeping_bias_voltage')
	with Romina.handle_task('noise_vs_bias_voltage') as task_handler:
		subruns = Romina.list_subruns_of_task('beta_scan_sweeping_bias_voltage')
		with multiprocessing.Pool(number_of_processes) as p:
			p.starmap(
				noise_in_beta_scan,
				[(bur,frc) for bur,frc in zip(subruns, [force_calculation_on_submeasurements]*len(subruns))]
			)
		
		noise = read_noise_in_beta_scan(bureaucrat)
		
		noise.to_pickle(task_handler.path_to_directory_of_my_task/'noise.pickle')
		
		summary = read_summarized_data(bureaucrat)
		summary.columns = [' '.join(col) for col in summary.columns]
		
		noise = noise.query('signal_name=="DUT"').reset_index('signal_name',drop=True).join(summary.reset_index('device_name',drop=False))
		
		fig = px.line(
			noise.sort_values(['device_name','Bias voltage (V) mean']),
			x = 'Bias voltage (V) mean',
			y = 'Noise (V)',
			error_y = 'Noise (V) error',
			error_x = 'Bias voltage (V) std',
			color = 'device_name',
			title = f'Noise vs bias voltage<br><sup>Run: {Romina.run_name}</sup>',
			markers = True,
		)
		fig.update_layout(xaxis = dict(autorange = "reversed"))
		fig.write_html(
			str(task_handler.path_to_directory_of_my_task/'noise_vs_bias_voltage.html'),
			include_plotlyjs = 'cdn',
		)

def script_core(bureaucrat:RunBureaucrat, force:bool):
	if bureaucrat.was_task_run_successfully('beta_scan'):
		noise_in_beta_scan(bureaucrat, force=True)
	elif bureaucrat.was_task_run_successfully('beta_scan_sweeping_bias_voltage'):
		noise_vs_bias_voltage(
			bureaucrat = bureaucrat,
			force_calculation_on_submeasurements = force,
			number_of_processes = max(multiprocessing.cpu_count()-1,1),
		)
	else:
		raise RuntimeError(f'Dont know how to process run {repr(bureaucrat.run_name)} located in `{bureaucrat.path_to_run_directory}`...')

if __name__ == '__main__':
	import argparse
	
	grafica.plotly_utils.utils.set_my_template_as_default()

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
