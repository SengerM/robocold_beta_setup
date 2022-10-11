from the_bureaucrat.bureaucrats import RunBureaucrat # https://github.com/SengerM/the_bureaucrat
from pathlib import Path
from clean_beta_scan import script_core as clean_beta_scan
from collected_charge import script_core as collected_charge
from jitter_calculation import script_core as jitter_calculation
from time_resolution import script_core as time_resolution
from summarize_parameters import summarize_beta_scan_measured_stuff_recursively as summarize_parameters

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

	args = parser.parse_args()
	bureaucrat = RunBureaucrat(Path(args.directory))
	
	summarize_parameters(bureaucrat, force=True)
	clean_beta_scan(bureaucrat)
	collected_charge(bureaucrat, force=True)
	jitter_calculation(
		bureaucrat,
		CFD_thresholds = 'best',
		force = True,
	)
	time_resolution(bureaucrat)
