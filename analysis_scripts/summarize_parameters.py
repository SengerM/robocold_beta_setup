import pandas
from the_bureaucrat.bureaucrats import RunBureaucrat # https://github.com/SengerM/the_bureaucrat
from pathlib import Path
import numpy
from huge_dataframe.SQLiteDataFrame import load_whole_dataframe # https://github.com/SengerM/huge_dataframe

def summarize_beta_scan_measured_stuff(bureaucrat:RunBureaucrat, force:bool=False):
	bureaucrat.check_these_tasks_were_run_successfully('beta_scan')
	
	if force == False and bureaucrat.was_task_run_successfully('summarize_beta_scan_measured_stuff'): # If this was already done, don't do it again...
		return
	
	with bureaucrat.handle_task('summarize_beta_scan_measured_stuff') as employee:
		bureaucrat.check_these_tasks_were_run_successfully('beta_scan')
		
		extra_stuff = load_whole_dataframe(employee.path_to_directory_of_task('beta_scan')/'measured_stuff.sqlite')
		extra_stuff.set_index('device_name', inplace=True, drop=True)
		extra_stuff.drop(columns=['When'], inplace=True)
		
		summary = extra_stuff.groupby(['device_name']).agg([numpy.mean,numpy.std])
		summary.to_pickle(employee.path_to_directory_of_my_task/'summary.pickle')

def summarize_beta_scan_measured_stuff_recursively(bureaucrat:RunBureaucrat, force:bool=False):
	if bureaucrat.was_task_run_successfully('beta_scan'):
		summarize_beta_scan_measured_stuff(bureaucrat, force)
	else:
		for task in bureaucrat.path_to_run_directory.iterdir():
			if not task.is_dir():
				continue
			for sub_bureaucrat in bureaucrat.list_subruns_of_task(task.parts[-1]):
				summarize_beta_scan_measured_stuff_recursively(sub_bureaucrat, force=force)

def read_summarized_data(bureaucrat:RunBureaucrat):
	if bureaucrat.was_task_run_successfully('beta_scan'): # This is the most fundamental kind of run, so just read.
		bureaucrat.check_these_tasks_were_run_successfully('summarize_beta_scan_measured_stuff')
		return pandas.read_pickle(bureaucrat.path_to_directory_of_task('summarize_beta_scan_measured_stuff')/'summary.pickle')
	elif bureaucrat.was_task_run_successfully('beta_scan_sweeping_bias_voltage'):
		summarized = []
		for subrun in bureaucrat.list_subruns_of_task('beta_scan_sweeping_bias_voltage'):
			subrun.check_these_tasks_were_run_successfully('summarize_beta_scan_measured_stuff')
			_ = read_summarized_data(subrun)
			_['run_name'] = subrun.run_name
			_.set_index('run_name',append=True,inplace=True)
			summarized.append(_)
		return pandas.concat(summarized)
	else:
		raise RuntimeError(f'Dont know how to read summarized data from run {repr(bureaucrat.run_name)} located in {repr(str(bureaucrat.path_to_run_directory))}.')

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
	summarize_beta_scan_measured_stuff_recursively(
		RunBureaucrat(Path(args.directory)),
		force = args.force,
	)
