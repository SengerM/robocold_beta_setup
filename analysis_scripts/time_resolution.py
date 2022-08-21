import pandas
from bureaucrat.SmarterBureaucrat import NamedTaskBureaucrat # https://github.com/SengerM/bureaucrat
from pathlib import Path
import plotly.express as px
import grafica.plotly_utils.utils # https://github.com/SengerM/grafica
from uncertainties import ufloat
from huge_dataframe.SQLiteDataFrame import load_whole_dataframe # https://github.com/SengerM/huge_dataframe

grafica.plotly_utils.utils.set_my_template_as_default()

def time_resolution_vs_bias_voltage_DUT_and_reference_trigger(path_to_measurement_base_directory:Path, reference_signal_name:str, reference_signal_time_resolution:float, reference_signal_time_resolution_error:float):
	Norberto = NamedTaskBureaucrat(
		path_to_measurement_base_directory,
		task_name = 'time_resolution_vs_bias_voltage',
		_locals = locals(),
	)
	
	Norberto.check_required_tasks_were_run_before(['jitter_calculation_beta_scan_sweeping_voltage','beta_scan_sweeping_bias_voltage'])
	
	path_to_any_submeasurement = [p for _,p in Norberto.find_submeasurements_of_task('beta_scan_sweeping_bias_voltage').items()][0]
	signal_names = set(load_whole_dataframe(path_to_any_submeasurement/'beta_scan/parsed_from_waveforms.sqlite').index.get_level_values('signal_name'))
	
	if reference_signal_name not in signal_names:
		raise ValueError(f'`reference_signal_name` is `{repr(reference_signal_name)}` which cannot be found in the measured signal names which are `{repr(signal_names)}`.')
	
	DUT_signal_name = signal_names - {reference_signal_name}
	if len(DUT_signal_name) != 1:
		raise RuntimeError(f'Cannot find the name of the DUT.')
	DUT_signal_name = list(DUT_signal_name)[0]
	
	with Norberto.do_your_magic():
		jitter_df = pandas.read_csv(Norberto.path_to_output_directory_of_task_named('jitter_calculation_beta_scan_sweeping_voltage')/'jitter_vs_bias_voltage.csv')
		jitter_df['Jitter (s) ufloat'] = jitter_df.apply(lambda x: ufloat(x['Jitter (s)'],x['Jitter (s) error']), axis=1)
		reference_signal_time_resolution_ufloat = ufloat(reference_signal_time_resolution, reference_signal_time_resolution_error)
		jitter_df.set_index(['Bias voltage (V)', 'measurement_name'], inplace=True)
		DUT_time_resolution = (jitter_df['Jitter (s) ufloat']**2-reference_signal_time_resolution_ufloat**2)**.5
		DUT_time_resolution.rename(f'Time resolution (s) ufloat', inplace=True)
		DUT_time_resolution_df = DUT_time_resolution.to_frame()
		DUT_time_resolution_df[f'Time resolution (s)'] = DUT_time_resolution_df[f'Time resolution (s) ufloat'].apply(lambda x: x.nominal_value)
		DUT_time_resolution_df[f'Time resolution (s) error'] = DUT_time_resolution_df[f'Time resolution (s) ufloat'].apply(lambda x: x.std_dev)
		DUT_time_resolution_df.drop(columns=f'Time resolution (s) ufloat', inplace=True)
		DUT_time_resolution_df['signal_name'] = DUT_signal_name
		
		reference_signal_time_resolution_df = pandas.DataFrame(
			{
				'Time resolution (s)': reference_signal_time_resolution,
				'Time resolution (s) error': reference_signal_time_resolution_error,
				'signal_name': reference_signal_name,
			},
			index = DUT_time_resolution_df.index,
		)
		for df in [DUT_time_resolution_df, reference_signal_time_resolution_df]:
			df.set_index('signal_name', append=True, inplace=True)
		
		time_resolution_df = pandas.concat([reference_signal_time_resolution_df, DUT_time_resolution_df])
		
		time_resolution_df.to_csv(Norberto.path_to_default_output_directory/'time_resolution.csv')
		
		fig = px.line(
			time_resolution_df.sort_index(level='Bias voltage (V)').reset_index(drop=False),
			x = 'Bias voltage (V)',
			y = f'Time resolution (s)',
			error_y = f'Time resolution (s) error',
			color = 'signal_name',
			markers = True,
			title = f'Time resolution vs bias voltage<br><sup>Measurement: {Norberto.measurement_name}</sup>',
		)
		fig.update_traces(error_y = dict(width = 1, thickness = .8))
		fig.write_html(
			str(Norberto.path_to_default_output_directory/'time_resolution_vs_bias_voltage.html'),
			include_plotlyjs = 'cdn',
		)

def time_resolution_vs_bias_voltage_comparison(path_to_measurement_base_directory:Path):
	Nicanor = NamedTaskBureaucrat(
		path_to_measurement_base_directory,
		task_name = 'time_resolution_vs_bias_voltage_comparison',
		_locals = locals(),
	)
	
	Nicanor.check_required_tasks_were_run_before('beta_scans')
	
	with Nicanor.do_your_magic():
		time_resolutions = []
		for submeasurement_name, path_to_submeasurement in Nicanor.find_submeasurements_of_task('beta_scans').items():
			Raul = NamedTaskBureaucrat(path_to_submeasurement, task_name='dummy_task_deleteme', _locals=locals())
			Raul.check_required_tasks_were_run_before('time_resolution_vs_bias_voltage')
			submeasurement_time_resolution = pandas.read_csv(Raul.path_to_output_directory_of_task_named('time_resolution_vs_bias_voltage')/'time_resolution.csv')
			submeasurement_time_resolution['beta_scan_vs_bias_voltage'] = submeasurement_name
			time_resolutions.append(submeasurement_time_resolution)
		df = pandas.concat(time_resolutions, ignore_index=True)
		
		df.to_csv(Nicanor.path_to_default_output_directory/'time_resolution.csv', index=False)
		
		df['measurement_timestamp'] = df['beta_scan_vs_bias_voltage'].apply(lambda x: x.split('_')[0])
		fig = px.line(
			df.sort_values(['measurement_timestamp','Bias voltage (V)','signal_name']),
			x = 'Bias voltage (V)',
			y = 'Time resolution (s)',
			error_y = 'Time resolution (s) error',
			color = 'measurement_timestamp',
			facet_col = 'signal_name',
			markers = True,
			title = f'Time resolution comparison<br><sup>Measurement: {Nicanor.measurement_name}</sup>',
			hover_data = ['beta_scan_vs_bias_voltage','measurement_name'],
			labels = {
				'measurement_name': 'Beta scan',
				'beta_scan_vs_bias_voltage': 'Beta scan vs bias voltage',
				'measurement_timestamp': 'Measurement timestamp',
			}
		)
		fig.write_html(
			str(Nicanor.path_to_default_output_directory/'time_resolution_vs_bias_voltage_comparison.html'),
			include_plotlyjs = 'cdn',
		)

def script_core(path_to_measurement_base_directory:Path):
	Manuel = NamedTaskBureaucrat(
		path_to_measurement_base_directory,
		task_name = 'dummy_task_deleteme',
		_locals = locals(),
	)
	if Manuel.check_required_tasks_were_run_before('beta_scan_sweeping_bias_voltage', raise_error=False):
		time_resolution_vs_bias_voltage_DUT_and_reference_trigger(
			path_to_measurement_base_directory = path_to_measurement_base_directory,
			reference_signal_name = 'reference_trigger',
			reference_signal_time_resolution = 17.32e-12, # My best characterization of the Photonis PMT.
			reference_signal_time_resolution_error = 2.16e-12, # My best characterization of the Photonis PMT.
		)
	elif Manuel.check_required_tasks_were_run_before('beta_scans', raise_error=False):
		time_resolution_vs_bias_voltage_comparison(path_to_measurement_base_directory)
	else:
		raise RuntimeError(f'Dont know how to process measurement {repr(Manuel.measurement_name)} located in `{Manuel.path_to_measurement_base_directory}`...')

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
	script_core(Path(args.directory))
