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

def plot_waveforms(bureaucrat:RunBureaucrat, force:bool=False):
	bureaucrat.check_these_tasks_were_run_successfully('beta_scan')
	
	if force == False and bureaucrat.was_task_run_successfully('plot_waveforms'):
		return
	
	with bureaucrat.handle_task('plot_waveforms') as employee:
		data = load_whole_dataframe(bureaucrat.path_to_directory_of_task('beta_scan')/'parsed_from_waveforms.sqlite')[['n_waveform']]
		data = tag_n_trigger_as_background_according_to_the_result_of_clean_beta_scan(bureaucrat, data)
		
		waveforms_connection = sqlite3.connect(bureaucrat.path_to_directory_of_task('beta_scan')/'waveforms.sqlite')
		n_waveforms_that_are_signal = set(data.query('is_background==False')['n_waveform'])
		waveforms = pandas.read_sql(
			sql = f"SELECT * from dataframe_table WHERE n_waveform IN {tuple(sorted(n_waveforms_that_are_signal))}",
			con = waveforms_connection,
			index_col = 'n_waveform',
		)
		
		waveforms = waveforms.merge(
			data.reset_index(drop=False).set_index('n_waveform')[['n_trigger','signal_name']],
			left_index = True,
			right_index = True,
		)
		
		df = waveforms.reset_index(drop=False).sort_values(['n_waveform','Time (s)'])
		fig = px.line(
			title = f'Waveforms<br><sup>{bureaucrat.run_name}</sup>',
			data_frame = df,
			x = 'Time (s)',
			y = 'Amplitude (V)',
			facet_row = 'signal_name',
			line_group = 'n_waveform',
		)
		fig.update_traces(opacity=.1)
		fig.update_yaxes(matches=None)
		fig.write_image(
			employee.path_to_directory_of_my_task/'signal_waveforms_plotly.pdf',
		)

def plot_waveforms_sweeping_bias_voltage(bureaucrat:RunBureaucrat, force:bool=False):
	bureaucrat.check_these_tasks_were_run_successfully('beta_scan_sweeping_bias_voltage')
	
	with bureaucrat.handle_task('plot_waveforms') as employee:
		subruns = bureaucrat.list_subruns_of_task('beta_scan_sweeping_bias_voltage')
		for b in subruns:
			plot_waveforms(b, force=force)
		
		path_to_subplots = []
		document_title = f'Waveforms {bureaucrat.run_name}'
		html_doc = dominate.document(title=document_title)
		with html_doc:
			dominate.tags.h1(document_title)
			with dominate.tags.div(style='display: flex; flex-direction: column; width: 100%;'):
				for subrun in subruns:
					dominate.tags.iframe(
						src = str(Path('..')/(subrun.path_to_directory_of_task('plot_waveforms')/'signal_waveforms_plotly.pdf').relative_to(bureaucrat.path_to_run_directory)),
						style = f'height: 80vh; min-height: 333px; width: 100%; min-width: 600px; border-style: none; border: none;'
					)
		with open(employee.path_to_directory_of_my_task/f'waveforms all together.html', 'w') as ofile:
			print(html_doc, file=ofile)

if __name__ == '__main__':
	import argparse
	
	grafica.plotly_utils.utils.set_my_template_as_default()

	parser = argparse.ArgumentParser(description='Plot the waveforms.')
	parser.add_argument('--dir',
		metavar = 'path',
		help = 'Path to the base measurement directory.',
		required = True,
		dest = 'directory',
		type = str,
	)

	args = parser.parse_args()
	bureaucrat = RunBureaucrat(Path(args.directory))
	plot_waveforms_sweeping_bias_voltage(bureaucrat, force=False)
