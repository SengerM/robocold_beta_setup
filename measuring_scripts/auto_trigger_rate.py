from the_bureaucrat.bureaucrats import RunBureaucrat # https://github.com/SengerM/the_bureaucrat
from pathlib import Path
import pandas
import time
from TheSetup import connect_me_with_the_setup
import plotly.express as px
import plotly.graph_objects as go
import numpy
import sys
sys.path.append('../analysis_scripts')
from events_rate import fit_exponential_to_time_differences
from scipy.stats import expon

def auto_trigger_rate(bureaucrat:RunBureaucrat, name_to_access_to_the_setup:str, slot_number:int, bias_voltage:float, trigger_level:float, n_bootstraps:int, timeout_seconds:float, n_measurements_per_trigger:int, silent:bool=True):
	the_setup = connect_me_with_the_setup()
	
	if not silent:
		print('Waiting for acquiring control of the hardware...')
	with the_setup.hold_signal_acquisition(who=name_to_access_to_the_setup), the_setup.hold_control_of_bias_for_slot_number(slot_number, who=name_to_access_to_the_setup), the_setup.hold_control_of_robocold(who=name_to_access_to_the_setup):
		if not silent:
			print('Control of hardware acquired.')
		with bureaucrat.handle_task('auto_trigger_rate') as employee:
			if not silent:
				print('Removing the beta source from the DUT...')
			the_setup.move_to_slot(slot_number=3, who=name_to_access_to_the_setup)
			if not silent:
				print('Setting bias voltage...')
			the_setup.set_bias_voltage(slot_number=slot_number, volts=bias_voltage, who=name_to_access_to_the_setup)
			
			the_setup.configure_oscilloscope_for_auto_trigger_study(
				who = name_to_access_to_the_setup, 
				trigger_level = trigger_level,
				timeout_seconds = timeout_seconds,
				n_measurements_per_trigger = n_measurements_per_trigger,
			)
			
			measurements_results = []
			for n_bootstrap in range(n_bootstraps):
				if not silent:
					print(f'Measuring data for n_bootstrap {n_bootstrap}/{n_bootstraps-1}...')
				try:
					the_setup.wait_for_trigger(who=name_to_access_to_the_setup, timeout=timeout_seconds*(n_measurements_per_trigger+1))
					triggers_times = the_setup.get_triggers_times()
				except RuntimeError as e:
					if 'Timed out waiting for oscilloscope to trigger' in str(e):
						triggers_times = [] # It could not be measured, because the trigger is too high and the oscilloscope never triggers.
				
				
				times_between_triggers = numpy.diff(triggers_times)
				if len(times_between_triggers) > 0: # This is the normal case.
					fit_params = fit_exponential_to_time_differences(time_differences=times_between_triggers, measurement_seconds=triggers_times[-1])
				else: # This happens when the trigger is too high and cannot measure because the oscilloscope never ever triggers.
					fit_params = {
						'Rate (events s^-1)': float('NaN'),
						'Offset (s)': float('NaN'),
					}
				fit_params['n_bootstrap'] = n_bootstrap
				measurements_results.append(fit_params)
			measurements_results = pandas.DataFrame.from_records(measurements_results).set_index('n_bootstrap')
			
			fig = px.ecdf(
				title = f'Auto trigger rate distribution<br><sup>{bureaucrat.run_name}</sup>',
				data_frame = measurements_results,
				x = 'Rate (events s^-1)',
				marginal = 'histogram',
				labels = {
					'Rate (events s^-1)': 'Trigger rate (triggers/s)',
				}
			)
			fig.write_html(
				employee.path_to_directory_of_my_task/'auto_trigger_rate_distribution.html',
				include_plotlyjs = 'cdn',
			)
			
			final_results = measurements_results.agg([numpy.nanmean, numpy.nanstd])
			final_results.rename(index={'nanmean':'value', 'nanstd':'error'},inplace=True)
			final_results.to_csv(employee.path_to_directory_of_my_task/'results.csv')
			final_results.to_pickle(employee.path_to_directory_of_my_task/'results.pickle')

if __name__ == '__main__':
	from configuration_files.current_run import Alberto
	from utils import create_a_timestamp
	import os
	
	NAME_TO_ACCESS_TO_THE_SETUP = f'auto trigger rate PID: {os.getpid()}'
	
	with Alberto.handle_task('auto_trigger', drop_old_data=False) as employee:
		John = employee.create_subrun('just_testing')
		auto_trigger_rate(
			bureaucrat = John, 
			name_to_access_to_the_setup = NAME_TO_ACCESS_TO_THE_SETUP,
			slot_number = 1,
			bias_voltage = 300,
			trigger_level = 5e-3,
			n_bootstraps = 4,
			timeout_seconds = 2,
			n_measurements_per_trigger = 111,
			silent = False,
		)
	
