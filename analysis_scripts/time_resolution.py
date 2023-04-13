import pandas
from the_bureaucrat.bureaucrats import RunBureaucrat # https://github.com/SengerM/the_bureaucrat
from pathlib import Path
import plotly.express as px
import grafica.plotly_utils.utils # https://github.com/SengerM/grafica
from uncertainties import ufloat
from huge_dataframe.SQLiteDataFrame import load_whole_dataframe # https://github.com/SengerM/huge_dataframe
from summarize_parameters import read_summarized_data
from utils import save_dataframe

def time_resolution_DUT_and_reference(bureaucrat:RunBureaucrat, DUT_signal_name:str, reference_signal_name:str, reference_signal_time_resolution:float, reference_signal_time_resolution_error:float):
	bureaucrat.check_these_tasks_were_run_successfully(['beta_scan','jitter_calculation_beta_scan'])
	
	with bureaucrat.handle_task('time_resolution_DUT_and_reference') as employee:
		jitter = pandas.read_pickle(bureaucrat.path_to_directory_of_task('jitter_calculation_beta_scan')/'jitter.pickle')
		
		signals_names = {col[2:-4] for col in jitter.columns if col[:2]=='k_' and col[-4:]==' (%)'}
		if len(signals_names) != 2:
			raise RuntimeError(f'Something is wrong with the data, I was expecting two signals but I am finding {len(signals_names)}...')
		
		if reference_signal_name not in signals_names:
			raise RuntimeError(f'Cannot find reference signal name within the measured signals...')
		
		if DUT_signal_name not in signals_names:
			raise RuntimeError(f'Cannot find `DUT_signal_name` {repr(DUT_signal_name)} within the signals available in run {repr(bureaucrat.run_name)} located in {bureaucrat.path_to_run_directory}. The signals measured in this run are {signals_names}. ')
		
		jitter = ufloat(jitter.loc[0,'Jitter (s)'], jitter.loc[0,'Jitter (s) error'])
		reference_contribution = ufloat(reference_signal_time_resolution,reference_signal_time_resolution_error)
		
		try:
			DUT_time_resolution = (jitter**2-reference_contribution**2)**.5
		except ValueError as e:
			if 'The uncertainties module does not handle complex results' in str(e):
				DUT_time_resolution = ufloat(float('NaN'), float('NaN'))
		
		DUT_time_resolution = pandas.Series(
			{
				'Time resolution (s)': DUT_time_resolution.nominal_value,
				'Time resolution (s) error': DUT_time_resolution.std_dev,
				'signal_name': DUT_signal_name,
			}
		)
		reference_time_resolution = pandas.Series(
			{
				'Time resolution (s)': reference_signal_time_resolution,
				'Time resolution (s) error': reference_signal_time_resolution_error,
				'signal_name': reference_signal_name,
			}
		)
		time_resolution = pandas.DataFrame.from_records([DUT_time_resolution,reference_time_resolution])
		time_resolution.set_index('signal_name',inplace=True)
		save_dataframe(time_resolution, name='time_resolution', location=employee.path_to_directory_of_my_task)

def read_time_resolution(bureaucrat:RunBureaucrat):
	if bureaucrat.was_task_run_successfully('beta_scan'):
		bureaucrat.check_these_tasks_were_run_successfully('time_resolution_DUT_and_reference')
		return pandas.read_pickle(bureaucrat.path_to_directory_of_task('time_resolution_DUT_and_reference')/'time_resolution.pickle')
	elif bureaucrat.was_task_run_successfully('beta_scan_sweeping_bias_voltage'):
		time_resolution = []
		for subrun in bureaucrat.list_subruns_of_task('beta_scan_sweeping_bias_voltage'):
			_ = read_time_resolution(subrun)
			_['run_name'] = subrun.run_name
			_.set_index('run_name',inplace=True,append=True)
			time_resolution.append(_)
		time_resolution = pandas.concat(time_resolution)
		return time_resolution
	else:
		raise RuntimeError(f'Dont know how to read the time resolution in run {repr(bureaucrat.run_name)} located in {repr(str(bureaucrat.path_to_run_directory))}')

def time_resolution_DUT_and_reference_vs_bias_voltage(bureaucrat:RunBureaucrat, DUT_signal_name:str, reference_signal_name:str, reference_signal_time_resolution:float, reference_signal_time_resolution_error:float, force:bool=False):
	bureaucrat.check_these_tasks_were_run_successfully('beta_scan_sweeping_bias_voltage')
	with bureaucrat.handle_task('time_resolution_DUT_and_reference_vs_bias_voltage') as employee:
		for subrun in bureaucrat.list_subruns_of_task('beta_scan_sweeping_bias_voltage'):
			time_resolution_DUT_and_reference(
				bureaucrat = subrun,
				DUT_signal_name = DUT_signal_name,
				reference_signal_name = reference_signal_name,
				reference_signal_time_resolution = reference_signal_time_resolution,
				reference_signal_time_resolution_error = reference_signal_time_resolution_error,
			)
		
		time_resolution = read_time_resolution(bureaucrat)
		
		if DUT_signal_name not in time_resolution.index.get_level_values('signal_name'):
			raise RuntimeError(f'Cannot find {repr(DUT_signal_name)} signal within the calculated time resolution data...')
		time_resolution = time_resolution.query(f'signal_name=="{DUT_signal_name}"')
		
		summary = read_summarized_data(bureaucrat)
		summary.columns = [f'{col[0]} {col[1]}' for col in summary.columns]
		
		time_resolution.reset_index(level='signal_name', inplace=True, drop=True)
		summary.reset_index(level='device_name', inplace=True)
		
		
		time_resolution = time_resolution.join(summary).sort_values('Bias voltage (V) mean')
		save_dataframe(time_resolution, name='time_resolution', location=employee.path_to_directory_of_my_task)
		
		fig = px.line(
			title = f'Time resolution vs bias voltage with beta source<br><sup>Run: {bureaucrat.run_name}</sup>',
			data_frame = time_resolution,
			x = 'Bias voltage (V) mean',
			y = 'Time resolution (s)',
			error_x = 'Bias voltage (V) std',
			error_y = 'Time resolution (s) error',
			markers = True,
			labels = {
				'Bias voltage (V) mean': 'Bias voltage (V)'
			},
		)
		fig.update_layout(xaxis = dict(autorange = "reversed"))
		fig.write_html(
			str(employee.path_to_directory_of_my_task/'time_resolution_vs_bias_voltage.html'),
			include_plotlyjs = 'cdn',
		)

def script_core(bureaucrat:RunBureaucrat, DUT_signal_name:str, reference_signal_name:str, reference_signal_time_resolution:float, reference_signal_time_resolution_error:float):
	if bureaucrat.was_task_run_successfully('beta_scan'):
		time_resolution_DUT_and_reference(
			bureaucrat = bureaucrat,
			DUT_signal_name = DUT_signal_name,
			reference_signal_name = reference_signal_name,
			reference_signal_time_resolution = reference_signal_time_resolution,
			reference_signal_time_resolution_error = reference_signal_time_resolution_error,
		)
	elif bureaucrat.was_task_run_successfully('beta_scan_sweeping_bias_voltage'):
		time_resolution_DUT_and_reference_vs_bias_voltage(
			bureaucrat = bureaucrat,
			DUT_signal_name = DUT_signal_name,
			reference_signal_name = reference_signal_name,
			reference_signal_time_resolution = reference_signal_time_resolution,
			reference_signal_time_resolution_error = reference_signal_time_resolution_error,
		)
	else:
		raise RuntimeError(f'Dont know how to process run {repr(Manuel.run_name)} located in {Manuel.path_to_run_directory}.')

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

	args = parser.parse_args()
	
	script_core(
		bureaucrat = RunBureaucrat(Path(args.directory)),
		DUT_signal_name = 'DUT',
		reference_signal_name = 'MCP-PMT',
		reference_signal_time_resolution = 17.32e-12, # My best characterization of the Photonis PMT.
		reference_signal_time_resolution_error = 2.16e-12, # My best characterization of the Photonis PMT.
	)
