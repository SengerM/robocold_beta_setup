from the_bureaucrat.bureaucrats import RunBureaucrat # https://github.com/SengerM/the_bureaucrat
from pathlib import Path
from clean_beta_scan import script_core as clean_beta_scan
from collected_charge import script_core as collected_charge
from jitter_calculation import script_core as jitter_calculation
from time_resolution import script_core as time_resolution
from summarize_parameters import summarize_beta_scan_measured_stuff_recursively as summarize_parameters
from plot_iv_curves import IV_curve_from_beta_scan_data

def do_all(bureaucrat:RunBureaucrat, force:bool=False):
	clean_beta_scan(bureaucrat)
	summarize_parameters(bureaucrat, force=True)
	IV_curve_from_beta_scan_data(bureaucrat)
	collected_charge(bureaucrat, force=force)
	jitter_calculation(
		bureaucrat,
		CFD_thresholds = {'DUT': 'best', 'MCP-PMT': 20},
		force = force,
	)
	time_resolution(
		bureaucrat = RunBureaucrat(Path(args.directory)),
		reference_signal_name = 'MCP-PMT',
		reference_signal_time_resolution = 17.32e-12, # My best characterization of the Photonis PMT.
		reference_signal_time_resolution_error = 2.16e-12, # My best characterization of the Photonis PMT.
	)

if __name__ == '__main__':
	import argparse

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
	
	do_all(
		bureaucrat,
		force = args.force,
	)
