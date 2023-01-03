from the_bureaucrat.bureaucrats import RunBureaucrat # https://github.com/SengerM/the_bureaucrat
from pathlib import Path
from clean_beta_scan import script_core as clean_beta_scan, create_cuts_file_template, automatic_cuts
from collected_charge import script_core as collected_charge
from jitter_calculation import script_core as jitter_calculation
from time_resolution import script_core as time_resolution
from summarize_parameters import summarize_beta_scan_measured_stuff_recursively as summarize_parameters
from plot_iv_curves import IV_curve_from_beta_scan_data
from noise_in_beta_scan import script_core as noise_in_beta_scan
from events_rate import events_rate_vs_bias_voltage
from plot_beta_scan import script_core as plot_beta_scan
import multiprocessing
import grafica.plotly_utils.utils

def do_all(bureaucrat:RunBureaucrat, CFD_thresholds:dict, path_to_cuts_file:Path=None, skip_charge:bool=False, skip_jitter:bool=False, force:bool=True, number_of_processes:int=1):
	clean_beta_scan(bureaucrat, path_to_cuts_file=path_to_cuts_file)
	summarize_parameters(bureaucrat, force=force)
	IV_curve_from_beta_scan_data(bureaucrat)
	events_rate_vs_bias_voltage(bureaucrat, force=force, number_of_processes=number_of_processes)
	if not skip_charge:
		collected_charge(bureaucrat, force=force, number_of_processes=number_of_processes)
	if not skip_jitter:
		jitter_calculation(
			bureaucrat,
			CFD_thresholds = CFD_thresholds,
			force = force,
			number_of_processes = number_of_processes,
		)
	time_resolution(
		bureaucrat = bureaucrat,
		reference_signal_name = 'MCP-PMT',
		reference_signal_time_resolution = 17.32e-12, # My best characterization of the Photonis PMT.
		reference_signal_time_resolution_error = 2.16e-12, # My best characterization of the Photonis PMT.
	)
	noise_in_beta_scan(
		bureaucrat = bureaucrat,
		force = force,
		number_of_processes = number_of_processes,
	)

if __name__ == '__main__':
	import argparse
	
	grafica.plotly_utils.utils.set_my_template_as_default()

	parser = argparse.ArgumentParser(description='Cleans a beta scan according to some criterion.')
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
	
	path_to_cuts_file = None
	if bureaucrat.was_task_run_successfully('beta_scan_sweeping_bias_voltage') and not (bureaucrat.path_to_run_directory/'cuts.csv').is_file():
		if input('Automatically find cuts in amplitude? (yes) ') == 'yes':
			automatic_cuts(bureaucrat)
			path_to_cuts_file = bureaucrat.path_to_directory_of_task('automatic_cuts')/'cuts.csv'
		else:
			create_cuts_file_template(bureaucrat)
			print(f'`cuts.csv` file template was created on {repr(str(bureaucrat.path_to_run_directory))}.')
			print(f'Will now exit, after doing the plots needed to configure the cuts manually...')
			plot_beta_scan(bureaucrat)
			exit()
	
	do_all(
		bureaucrat,
		path_to_cuts_file = path_to_cuts_file,
		skip_charge = False,
		skip_jitter = False,
		CFD_thresholds = {'DUT': 'best', 'MCP-PMT': 'best'},
		force = args.force,
		number_of_processes = max(multiprocessing.cpu_count()-1,1),
	)
