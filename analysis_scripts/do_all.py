from the_bureaucrat.bureaucrats import RunBureaucrat # https://github.com/SengerM/the_bureaucrat
from pathlib import Path
from clean_beta_scan import script_core as clean_beta_scan, create_cuts_file_template, automatic_cuts
from collected_charge import fit_Landau_and_extract_MPV_sweeping_voltage, collected_charge_in_Coulomb
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
	_ = set(CFD_thresholds)
	_.remove('MCP-PMT')
	if len(_) != 1:
		raise RuntimeError('`CFD_thresholds` does not have two elements. It must have two, e.g. `{"DUT": 10, "MCP-PMT": 20}`. ')
	DUT_signal_name = list(_)[0]
	
	clean_beta_scan(bureaucrat, path_to_cuts_file=path_to_cuts_file)
	summarize_parameters(bureaucrat, force=force)
	IV_curve_from_beta_scan_data(bureaucrat)
	events_rate_vs_bias_voltage(bureaucrat, force=force, number_of_processes=number_of_processes)
	if not skip_charge:
		subruns = list(bureaucrat.list_subruns_of_task('beta_scan_sweeping_bias_voltage'))
		
		CALIBRATION_FACTORS_COULOMB = { # This comes from https://sengerm.github.io/Chubut_2/doc/testing/index.html
			'Amplitude (V)': {'val': 5.67e12, 'err': .25e12},
			'Collected charge (V s)': {'val': 5050, 'err': 250},
		}
		for variable in ['Amplitude (V)','Collected charge (V s)']:
			fit_Landau_and_extract_MPV_sweeping_voltage(
				bureaucrat = bureaucrat,
				time_from_trigger_signal = {subrun.run_name:{_:(1.4e-9,1.9e-9) for _ in [DUT_signal_name]} for subrun in subruns},
				time_from_trigger_background = {subrun.run_name:{_:(1e-9,1.4e-9) for _ in [DUT_signal_name]} for subrun in subruns},
				signal_name_trigger = 'MCP-PMT',
				n_bootstraps = 22,
				force = force,
				collected_charge_variable_name = variable,
				number_of_processes = number_of_processes,
				use_clean_beta_scan = True,
				fit_R_squared_threshold_when_aggregating_x_mpv = .94,
			)
			collected_charge_in_Coulomb(
				bureaucrat = bureaucrat,
				collected_charge_variable_name = variable,
				conversion_factor_to_divide_by = CALIBRATION_FACTORS_COULOMB[variable]['val'],
				conversion_factor_to_divide_by_error = CALIBRATION_FACTORS_COULOMB[variable]['err'],
			)
	if not skip_jitter:
		jitter_calculation(
			bureaucrat,
			CFD_thresholds = CFD_thresholds,
			force = force,
			number_of_processes = number_of_processes,
		)
	
	time_resolution(
		bureaucrat = bureaucrat,
		DUT_signal_name = DUT_signal_name,
		reference_signal_name = 'MCP-PMT',
		reference_signal_time_resolution = 17.5e-12, # Photonis PMT.
		reference_signal_time_resolution_error = 2.5e-12, # Photonis PMT.
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
	parser.add_argument(
		'--dir',
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
	parser.add_argument(
		'--automatic-cuts',
		help = 'Whether to use cuts from a file or estimate automatic cuts.',
		required = False,
		dest = 'automatic_cuts',
		action = 'store_true'
	)

	args = parser.parse_args()
	bureaucrat = RunBureaucrat(Path(args.directory))
	
	DUT_SIGNAL_NAME = 'DUT'
	
	path_to_cuts_file = None
	if not args.automatic_cuts:
		if bureaucrat.was_task_run_successfully('beta_scan_sweeping_bias_voltage') and not (bureaucrat.path_to_run_directory/'cuts.csv').is_file():
			create_cuts_file_template(bureaucrat)
			print(f'`cuts.csv` file template was created on {repr(str(bureaucrat.path_to_run_directory))}.')
			print(f'Will now exit, after doing the plots needed to configure the cuts manually...')
			plot_beta_scan(bureaucrat)
			exit()
	else:
		automatic_cuts(bureaucrat, DUT_signal_name=DUT_SIGNAL_NAME)
		path_to_cuts_file = bureaucrat.path_to_directory_of_task('automatic_cuts')/'cuts.csv'
	
	do_all(
		bureaucrat,
		path_to_cuts_file = path_to_cuts_file,
		skip_charge = True,
		skip_jitter = False,
		CFD_thresholds = {DUT_SIGNAL_NAME: 'best', 'MCP-PMT': 20},
		force = args.force,
		number_of_processes = max(multiprocessing.cpu_count()-1,1),
	)
