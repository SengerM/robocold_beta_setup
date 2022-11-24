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
import grafica.plotly_utils.utils # https://github.com/SengerM/grafica

def auto_trigger_rate(bureaucrat:RunBureaucrat, name_to_access_to_the_setup:str, slot_number:int, bias_voltage:float, trigger_level:float, n_bootstraps:int, timeout_seconds:float, n_measurements_per_trigger:int, silent:bool=True):
	the_setup = connect_me_with_the_setup()
	
	if not isinstance(trigger_level, (int,float)):
		raise TypeError(f'`trigger_level` must be a float (or int) number, received object of type {type(trigger_level)}.')
	
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
				if len(times_between_triggers) > 1: # This is the normal case.
					try:
						fit_params = fit_exponential_to_time_differences(time_differences=times_between_triggers, measurement_seconds=triggers_times[-1])
						
						if fit_params['Offset (s)'] > 10e-6:
							# This I do because:
							# 1) The time it takes our current oscilloscope (LeCroy 9254M) to trigger
							#    again is less than 1 µs, and the resolution seems to be 1 µs. So if 
							#    things go well this parameter "Offset (s)" should be close to 1 µs.
							# 2) When the threshold gets too high and we have very low statistics because
							#    the oscilloscope always times-out, the fit tends to converge to a very
							#    high offset value, which is clearly wrong and is only an artifact of the
							#    very low statistics. In this case, however, usually we can approximate
							#    offset = 0 and use the MLE estimator for the rate of an exponential. 
							fit_params = {
								'Rate (events s^-1)': numpy.nanmean(triggers_times)**-1,
								'Offset (s)': 0,
							}
					except RuntimeError as e:
						if 'Optimal parameters not found' in str(e):
							fit_params = {
								'Rate (events s^-1)': float('NaN'),
								'Offset (s)': float('NaN'),
							}
							continue
					
					if numpy.random.rand() < 5/n_bootstraps: # Do a plot for this iteration.
						fig = px.ecdf(
							title = f'Auto trigger rate measurement (n_bootstrap={n_bootstrap}<br><sup>{bureaucrat.run_name}</sup>',
							x = times_between_triggers,
						)
						x_axis_values = numpy.logspace(max(numpy.log10(min(times_between_triggers)),-7),numpy.log10(max(times_between_triggers)))
						fig.add_trace(
							go.Scatter(
								x = x_axis_values,
								y = expon.cdf(x_axis_values, scale = fit_params['Rate (events s^-1)']**-1, loc = fit_params['Offset (s)']),
								mode = 'lines',
								name = f"λ={fit_params['Rate (events s^-1)']:.2e} evnts/s, offset={fit_params['Offset (s)']:.2e} s",
								line = dict(
									dash = 'dash',
									color = 'black',
								),
							)
						)
						fig.update_layout(
							xaxis_title = 't<sub>trigger i</sub> - t<sub>trigger i-1</sub> (s)',
						)
						path_for_these_plots = employee.path_to_directory_of_my_task/'some_random_plots_of_exponential_fits'
						path_for_these_plots.mkdir(exist_ok=True)
						fig.write_html(
							path_for_these_plots/f'n_bootstrap_{n_bootstrap}.html',
							include_plotlyjs = 'cdn',
						)
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
			
			final_results = measurements_results.agg([numpy.nanmedian, numpy.nanstd])
			final_results.rename(index={'nanmedian':'value', 'nanstd':'error'},inplace=True)
			final_results.to_csv(employee.path_to_directory_of_my_task/'results.csv')
			final_results.to_pickle(employee.path_to_directory_of_my_task/'results.pickle')

def auto_trigger_rate_sweeping_trigger_level(bureaucrat:RunBureaucrat, name_to_access_to_the_setup:str, slot_number:int, bias_voltage:float, trigger_levels:list, n_bootstraps:int, timeout_seconds:float, n_measurements_per_trigger:int, silent:bool=True):
	the_setup = connect_me_with_the_setup()
	
	with bureaucrat.handle_task('auto_trigger_rate_sweeping_trigger_level',drop_old_data=False) as employee:
		# ~ for trigger_level in trigger_levels:
			# ~ if not silent:
				# ~ print(f'Measuring at trigger level {trigger_level}...')
			# ~ auto_trigger_rate(
				# ~ bureaucrat = employee.create_subrun(f'{bureaucrat.run_name}_Threshold{trigger_level*1e3:.2f}mV'), 
				# ~ name_to_access_to_the_setup = name_to_access_to_the_setup,
				# ~ slot_number = slot_number,
				# ~ bias_voltage = bias_voltage,
				# ~ trigger_level = trigger_level,
				# ~ n_bootstraps = n_bootstraps,
				# ~ timeout_seconds = timeout_seconds,
				# ~ n_measurements_per_trigger = n_measurements_per_trigger,
				# ~ silent = silent,
			# ~ )
		data = read_auto_trigger_rate(bureaucrat)
		
		rate = data.query('variable == "Rate (events s^-1)"').set_index('type',append=True)
		rate = rate.unstack()['value']
		rate.columns = [f'Rate (events s^-1) {col}'.replace(' value','') for col in rate.columns]
		rate.columns.name = None
		
		offset = data.query('variable == "Offset (s)"').set_index('type',append=True)
		offset = offset.unstack()['value']
		offset.columns = [f'Offset (s) {col}'.replace(' value','') for col in offset.columns]
		offset.columns.name = None
		
		data = pandas.concat([rate,offset], axis=1)
		
		for var in {'Rate (events s^-1)','Offset (s)'}:
			fig = grafica.plotly_utils.utils.line(
				title = f'Rate vs trigger threshold',
				data_frame = data.reset_index(drop=False).sort_values('run_name'),
				y = var,
				error_y = f'{var} error',
				x = 'run_name',
				markers = True,
				log_y = True if 'rate' in var.lower() else False,
			)
			fig.write_html(
				employee.path_to_directory_of_my_task/(var.split(' ')[0] + '.html'),
				include_plotlyjs = 'cdn',
			)

def read_auto_trigger_rate(bureaucrat:RunBureaucrat):
	if bureaucrat.was_task_run_successfully('auto_trigger_rate'):
		return pandas.read_pickle(bureaucrat.path_to_directory_of_task('auto_trigger_rate')/'results.pickle')
	elif True:#bureaucrat.was_task_run_successfully('auto_trigger_rate_sweeping_trigger_level'):
		data = []
		for subrun in bureaucrat.list_subruns_of_task('auto_trigger_rate_sweeping_trigger_level'):
			_ = read_auto_trigger_rate(subrun)
			_.index.name = 'type'
			_.columns.name = 'variable'
			_ = _.unstack()
			_.name = 'value'
			_ = _.to_frame()
			_['run_name'] = subrun.run_name
			_.reset_index(inplace=True,drop=False)
			_.set_index('run_name',inplace=True)
			data.append(_)
		data = pandas.concat(data)
		return data
	else:
		raise RuntimeError(f'Dont know how to read the auto trigger rate in run {repr(bureaucrat.run_name)} located in {repr(str(bureaucrat.path_to_run_directory))}')

if __name__ == '__main__':
	from configuration_files.current_run import Alberto
	from utils import create_a_timestamp
	import os
	
	NAME_TO_ACCESS_TO_THE_SETUP = f'auto trigger rate PID: {os.getpid()}'
	
	with Alberto.handle_task('auto_trigger', drop_old_data=False) as employee:
		John = employee.create_subrun('just_testing_average')
		auto_trigger_rate_sweeping_trigger_level(
			bureaucrat = John, 
			name_to_access_to_the_setup = NAME_TO_ACCESS_TO_THE_SETUP,
			slot_number = 1,
			bias_voltage = 300,
			trigger_levels = [2e-3,5e-3,8e-3,13e-3],
			n_bootstraps = 44,
			timeout_seconds = 1,
			n_measurements_per_trigger = 1111,
			silent = False,
		)
	
