from the_bureaucrat.bureaucrats import RunBureaucrat # https://github.com/SengerM/the_bureaucrat
from pathlib import Path
import pandas
import plotly.graph_objects as go
import plotly.express as px
import grafica.plotly_utils.utils
from huge_dataframe.SQLiteDataFrame import load_only_index_without_repeated_entries, load_whole_dataframe
from clean_beta_scan import tag_n_trigger_as_background_according_to_the_result_of_clean_beta_scan
import sqlite3
import dominate # https://github.com/Knio/dominate
import numpy

def frequency_domain_analysis(bureaucrat:RunBureaucrat, force:bool=False):
	bureaucrat.check_these_tasks_were_run_successfully('beta_scan')
	
	if force == False and bureaucrat.was_task_run_successfully('frequency_domain_analysis'):
		return
	
	with bureaucrat.handle_task('frequency_domain_analysis') as employee:
		data = load_whole_dataframe(bureaucrat.path_to_directory_of_task('beta_scan')/'parsed_from_waveforms.sqlite')
		data = tag_n_trigger_as_background_according_to_the_result_of_clean_beta_scan(bureaucrat, data)
		
		waveforms_connection = sqlite3.connect(bureaucrat.path_to_directory_of_task('beta_scan')/'waveforms.sqlite')
		n_waveforms_that_are_signal = set(data.query('is_background==False')['n_waveform'])
		waveforms = pandas.read_sql(
			sql = f"SELECT * from dataframe_table WHERE n_waveform IN {tuple(sorted(n_waveforms_that_are_signal))}",
			con = waveforms_connection,
			index_col = 'n_waveform',
		)
		
		waveforms = waveforms.merge(
			data.reset_index(drop=False).set_index('n_waveform')[['n_trigger','signal_name','t_50 (s)']],
			left_index = True,
			right_index = True,
		)
		
		fft = waveforms['Amplitude (V)'].groupby('n_waveform').apply(numpy.fft.rfft)
		_ = fft.index
		fft = pandas.DataFrame(fft.tolist())
		fft.index = _
		fft = fft.stack()
		fft.index.names = ['n_waveform','n_sample']
		fft.name = 'FFT'
		
		sampling_frequency = (waveforms['Time (s)'].groupby('n_waveform').apply(numpy.diff).apply(numpy.mean)**-1).mean() # I am assuming it is the same for all of them.
		frequency_axis = numpy.fft.rfftfreq(
			n = len(waveforms.query(f'n_waveform=={waveforms.index.get_level_values("n_waveform")[0]}')),
			d = sampling_frequency**-1,
		)
		frequency_axis = pandas.Series(
			frequency_axis*1e-9,
			name = 'Frequency (GHz)',
		)
		frequency_axis.index.names = ['n_sample']
		fft = fft.to_frame().join(frequency_axis, on='n_sample')
		
		fft = fft.merge(
			data.reset_index(drop=False).set_index('n_waveform')[['n_trigger','signal_name']],
			left_index = True,
			right_index = True,
		)
		
		fft['abs(FFT)'] = abs(fft['FFT'])
		
		fft = fft.drop(fft[fft.index.get_level_values('n_sample')==0].index) # Drop the DC value
		# ~ fft = fft.query('signal_name != "MCP-PMT"')
		
		fig = px.line(
			title = f'FFTs<br><sup>{bureaucrat.run_name}</sup>',
			data_frame = fft.reset_index(drop=False),
			x = 'Frequency (GHz)',
			y = 'abs(FFT)',
			facet_row = 'signal_name',
			line_group = 'n_waveform',
			labels = {
				'Time (s) corrected': 'Time (s)',
			},
		)
		fig.update_xaxes(
			type = "log",
			range = [numpy.log10(.01),numpy.log10(5)]
		)
		fig.update_yaxes(
			range = [numpy.log10(1000e-6),numpy.log10(1)]
		)
		fig.update_traces(opacity=.1)
		fig.update_yaxes(matches=None)
		fig.write_image(
			employee.path_to_directory_of_my_task/'FFTs_lin.pdf',
		)
		fig.update_yaxes(type="log")
		fig.write_image(
			employee.path_to_directory_of_my_task/'FFTs_log.pdf',
		)
		a

if __name__ == '__main__':
	import argparse
	
	grafica.plotly_utils.utils.set_my_template_as_default()

	parser = argparse.ArgumentParser(description='Perform a frequency domain analysis.')
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
	bureaucrat = RunBureaucrat(Path(args.directory))
	frequency_domain_analysis(
		bureaucrat, 
		force = args.force,
	)
