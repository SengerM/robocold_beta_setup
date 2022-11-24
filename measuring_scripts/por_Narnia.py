from the_bureaucrat.bureaucrats import RunBureaucrat # https://github.com/SengerM/the_bureaucrat
from beta_scan import beta_scan
from TheSetup import connect_me_with_the_setup
import time
from pathlib import Path

def does_device_in_slot_number_looks_like_dead(slot_number:int)->str:
	"""Returns `"yes"` or `"no"`, or `"?"` if I don't know."""
	UNKNOWN = '?'
	DEVICE_IS_DEAD = 'yes'
	DEVICE_IS_ALIVE = 'no'
	
	the_setup = connect_me_with_the_setup()
	
	compliance = abs(the_setup.get_current_compliance(slot_number))
	current = abs(the_setup.measure_bias_current(slot_number))
	set_voltage = abs(the_setup.get_set_voltage(slot_number))
	voltage = abs(the_setup.measure_bias_voltage(slot_number))
	status = the_setup.get_bias_voltage_status(slot_number)
	
	if status == 'off' or set_voltage < 11: # There is an offset, if it is less than 11 there is no voltage applied... This is how the CAEN works...
		return UNKNOWN
	if abs(current-compliance) < 5e-6:
		if voltage < 11:
			return DEVICE_IS_DEAD
	if current < 1e-6 and voltage < 11 and status == 'on' and set_voltage > 11 and compliance > 1e-6:
		# Once, don't ask me why, the CAEN went into a weird state in which for some reason it was not able to apply voltage. It looked like the power supply itself was broken. Then I made a hard reset and it went back to life.
		return UNKNOWN
	return DEVICE_IS_ALIVE

def set_maximum_current_compliance(slot_number:int, name_to_access_to_the_setup):
	the_setup = connect_me_with_the_setup()
	increase_µA = 10
	while True:
		current_current_compliance = the_setup.get_current_compliance(slot_number)
		try:
			the_setup.set_current_compliance(slot_number, amperes=current_current_compliance+increase_µA*1e-6, who=name_to_access_to_the_setup)
		except Exception:
			increase_µA -= 1
		if increase_µA <= 0:
			break
	current_current_compliance = the_setup.get_current_compliance(slot_number)
	the_setup.set_current_compliance(slot_number, amperes=current_current_compliance-10e-6, who=name_to_access_to_the_setup)

def kill_device_por_Narnia(bureaucrat:RunBureaucrat, initial_bias_voltage:float, name_to_access_to_the_setup:str, slot_number:int, n_triggers_per_voltage:list, voltage_step:float, software_trigger=None, silent=False):
	the_setup = connect_me_with_the_setup()
	
	if not silent:
		print(f'Waiting to acquire control of the setup...')
	with \
		bureaucrat.handle_task('beta_scan_sweeping_bias_voltage') as employee, \
		the_setup.hold_signal_acquisition(who=name_to_access_to_the_setup), \
		the_setup.hold_control_of_bias_for_slot_number(slot_number, who=name_to_access_to_the_setup), \
		the_setup.hold_control_of_robocold(who=name_to_access_to_the_setup) \
	:
		try:
			if not silent:
				print(f'Control of the setup acquired!')
			if not silent:
				print(f'Setting maximum current compliance for slot number {slot_number}...')
			set_maximum_current_compliance(slot_number, name_to_access_to_the_setup)
			if not silent:
				print(f'Current compliance of slot number {slot_number} is now {the_setup.get_current_compliance(slot_number)*1e6:.2f} µA.')
			
			current_voltage = initial_bias_voltage
			keep_measuring = True
			while keep_measuring:
				if not silent:
					print(f'Seting bias voltage to {int(current_voltage)} V to slot number {slot_number}...')
				the_setup.set_bias_voltage(slot_number, volts=current_voltage, who=name_to_access_to_the_setup)
				
				if not silent:
					print(f'Waiting a few seconds for any transcient or whatever to finish...')
				time.sleep(10)
				
				is_device_dead = does_device_in_slot_number_looks_like_dead(slot_number)
				if is_device_dead == '?':
					keep_measuring = False
					raise RuntimeError(f'Cannot determine if the device in slot number {slot_number} is dead or alive. One possible reason is that the output is off. Summary: compliance = {the_setup.get_current_compliance(slot_number)*1e6:.2f} µA, current = {the_setup.measure_bias_current(slot_number)*1e6:.2f} µA, set voltage = {int(abs(the_setup.get_set_voltage(slot_number)))} V, voltage = {int(abs(the_setup.measure_bias_voltage(slot_number)))} V, output status = {repr(the_setup.get_bias_voltage_status(slot_number))}.')
				elif is_device_dead == 'yes':
					keep_measuring = False
					if not silent:
						print(f'Device in slot number {slot_number} seems to be dead. Finishing task {repr(employee.task_name)} in run {repr(employee.run_name)}')
					break
				
				if not silent:
					print(f'The device seems to be still alive ({the_setup.measure_bias_current(slot_number)*1e6:.2f} µA, {int(the_setup.measure_bias_voltage(slot_number))} V).')
				
				measured_I = the_setup.measure_bias_current(slot_number)
				set_I = the_setup.get_current_compliance(slot_number)
				if is_device_dead == 'no' and (set_I > 100e-6 and (abs(measured_I - set_I) < 10e-6 or measured_I > set_I)):
					if not silent:
						print(f'The device seems to be still alive BUT the current compliance has been reached. This means that we dont have enough power to kill it, so I will finish now.')
					keep_measuring = False
					break
				
				beta_scan(
					bureaucrat = employee.create_subrun(f'{bureaucrat.run_name}_{int(current_voltage)}V'), 
					name_to_access_to_the_setup = name_to_access_to_the_setup, 
					slot_number = slot_number, 
					n_triggers = n_triggers_per_voltage, 
					bias_voltage = current_voltage, 
					software_trigger = software_trigger, 
					silent = silent,
				)
				
				current_voltage += voltage_step
		finally:
			if not silent:
				print(f'Setting bias voltage and current to 0...')
			the_setup.set_bias_voltage(slot_number, volts=0, who=name_to_access_to_the_setup)
			the_setup.set_current_compliance(slot_number, amperes=1e-6, who=name_to_access_to_the_setup)
			time.sleep(5)
			if not silent:
				print(f'Slot number {slot_number}: {the_setup.measure_bias_current(slot_number)*1e6:.2f} µA, {int(the_setup.measure_bias_voltage(slot_number))} V')

if __name__=='__main__':
	import os
	from configuration_files.current_run import Alberto
	PATH_TO_ANALYSIS_SCRIPTS = Path(__file__).resolve().parent.parent/'analysis_scripts'
	import sys
	sys.path.append(str(PATH_TO_ANALYSIS_SCRIPTS))
	from plot_beta_scan import plot_everything_from_beta_scan
	from utils import create_a_timestamp
	from progressreporting.TelegramProgressReporter import TelegramReporter # https://github.com/SengerM/progressreporting
	import my_telegram_bots
	
	def software_trigger(signals_dict, minimum_DUT_amplitude:float):
		DUT_signal = signals_dict['DUT']
		PMT_signal = signals_dict['MCP-PMT']
		try:
			is_peak_in_correct_time_window = 1e-9 < float(DUT_signal.find_time_at_rising_edge(50)) - float(PMT_signal.find_time_at_rising_edge(50)) < 5.5e-9
		except Exception:
			is_peak_in_correct_time_window = False
		is_DUT_amplitude_above_threshold = DUT_signal.amplitude > minimum_DUT_amplitude
		return is_peak_in_correct_time_window and is_DUT_amplitude_above_threshold
	
	NAME_TO_ACCESS_TO_THE_SETUP = f'beta scan por Narnia PID: {os.getpid()}'
	
	SLOT_NUMBERS = [4]
	INITIAL_BIAS_VOLTAGES = [400]
	
	the_setup = connect_me_with_the_setup()
	
	with Alberto.handle_task('beta_scans', drop_old_data=False) as beta_scans_task_bureaucrat:
		reporter = TelegramReporter(
			telegram_token = my_telegram_bots.robobot.token,
			telegram_chat_id = my_telegram_bots.chat_ids['Robobot beta setup'],
		)
		try:
			for slot_number, initial_bias_voltage in zip(SLOT_NUMBERS,INITIAL_BIAS_VOLTAGES):
				reporter.send_message(f'Beggining "por Narnia" for device {the_setup.get_name_of_device_in_slot_number(slot_number)} in slot number {slot_number}...')
				kill_device_por_Narnia(
					bureaucrat = beta_scans_task_bureaucrat.create_subrun(f'{create_a_timestamp()} {the_setup.get_name_of_device_in_slot_number(slot_number)} por Narnia'.replace(' ','_')),
					initial_bias_voltage = initial_bias_voltage,
					name_to_access_to_the_setup = NAME_TO_ACCESS_TO_THE_SETUP, 
					slot_number = slot_number, 
					n_triggers_per_voltage = 5555,
					voltage_step = 11, 
					software_trigger = lambda x: software_trigger(x, 0), 
					silent = False
				)
				reporter.send_message(f'Successfully finished "por Narnia" for device {the_setup.get_name_of_device_in_slot_number(slot_number)} in slot number {slot_number}...')
		finally:
			reporter.send_message(f'Finished por Narnia!')
