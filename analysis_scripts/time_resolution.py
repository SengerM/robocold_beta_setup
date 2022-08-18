import pandas
from bureaucrat.SmarterBureaucrat import NamedTaskBureaucrat # https://github.com/SengerM/bureaucrat
from pathlib import Path
import plotly.express as px
from uncertainties import ufloat
from huge_dataframe.SQLiteDataFrame import load_whole_dataframe # https://github.com/SengerM/huge_dataframe

def time_resolution_vs_bias_voltage_DUT_and_reference_trigger(path_to_measurement_base_directory: Path, reference_signal_name:str, reference_signal_time_resolution:float, reference_signal_time_resolution_error:float):
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
	time_resolution_vs_bias_voltage_DUT_and_reference_trigger(
		Path(args.directory),
		reference_signal_name = 'reference_trigger',
		reference_signal_time_resolution = 17.32e-12,
		reference_signal_time_resolution_error = 2.16e-12,
	)
