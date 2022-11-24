from the_bureaucrat.bureaucrats import RunBureaucrat # https://github.com/SengerM/the_bureaucrat
from pathlib import Path
import pandas
from TheSetup import connect_me_with_the_setup, load_beta_scans_configuration
import numpy
from beta_scan import beta_scan_sweeping_bias_voltage
from auto_trigger_rate import auto_trigger_rate_sweeping_trigger_level_and_bias_voltage

def automatic_measurements(bureaucrat:RunBureaucrat, name_to_access_to_the_setup:str, beta_scans_configuration_df:pandas.DataFrame, silent:bool=False):
	the_setup = connect_me_with_the_setup()
	with bureaucrat.handle_task('automatic_beta_scans', drop_old_data=False) as employee:
		if not silent:
			print(f'Waiting to acquire control of Robocold...')
		with the_setup.hold_control_of_robocold(who=name_to_access_to_the_setup):
			if not silent:
				print('Control of Robocold acquired!')
			for slot_number in beta_scans_configuration_df.index.unique():
				if not silent:
					print(f'Reseting robocold...')
				the_setup.reset_robocold(who=name_to_access_to_the_setup)
				the_setup.set_oscilloscope_vdiv(
					oscilloscope_channel_number = the_setup.get_oscilloscope_configuration_df().loc['DUT','n_channel'], 
					vdiv = beta_scans_configuration_df.loc[slot_number,'Oscilloscope vertical scale (V/DIV)'].max(),
					who = name_to_access_to_the_setup,
				)
				if not silent:
					print(f'Starting beta scans sweeping bias voltage on slot {slot_number}...')
				John = employee.create_subrun(f'{create_a_timestamp()}_{the_setup.get_name_of_device_in_slot_number(slot_number)}')
				beta_scan_sweeping_bias_voltage(
					bureaucrat = John,
					name_to_access_to_the_setup = name_to_access_to_the_setup,
					slot_number = slot_number,
					n_triggers_per_voltage = beta_scans_configuration_df.loc[slot_number,'n_triggers'], 
					bias_voltages = beta_scans_configuration_df.loc[slot_number,'Bias voltage (V)'],
					software_triggers = beta_scans_configuration_df.loc[slot_number,'software_trigger'],
					silent = silent,
				)
				if not silent:
					print(f'Beta scan sweeping bias voltage on slot {slot_number} finished.')
				
				if not silent:
					print(f'Starting auto-trigger rate measurement for slot {slot_number}...')
				auto_trigger_rate_sweeping_trigger_level_and_bias_voltage(
					bureaucrat = John,
					name_to_access_to_the_setup = name_to_access_to_the_setup,
					slot_number = slot_number,
					bias_voltages = beta_scans_configuration_df.loc[slot_number,'Bias voltage (V)'],
					trigger_levels = numpy.array(sorted(set((numpy.logspace(numpy.log10(2e-3),numpy.log10(70e-3),111)*1e4).astype(int))))/1e4,
					n_bootstraps = 11,
					timeout_seconds = .1,
					n_measurements_per_trigger = 1111,
					silent = silent,
				)
				if not silent:
					print(f'Auto-trigger rate sweeping bias voltage on slot {slot_number} finished.')

if __name__=='__main__':
	import os
	from configuration_files.current_run import Alberto
	PATH_TO_ANALYSIS_SCRIPTS = Path(__file__).resolve().parent.parent/'analysis_scripts'
	import sys
	sys.path.append(str(PATH_TO_ANALYSIS_SCRIPTS))
	from plot_beta_scan import plot_everything_from_beta_scan
	from utils import create_a_timestamp
	
	def software_trigger(signals_dict, minimum_DUT_amplitude:float):
		DUT_signal = signals_dict['DUT']
		PMT_signal = signals_dict['MCP-PMT']
		try:
			is_peak_in_correct_time_window = 1e-9 < float(DUT_signal.find_time_at_rising_edge(50)) - float(PMT_signal.find_time_at_rising_edge(50)) < 5.5e-9
		except Exception:
			is_peak_in_correct_time_window = False
		is_DUT_amplitude_above_threshold = DUT_signal.amplitude > minimum_DUT_amplitude
		return is_peak_in_correct_time_window and is_DUT_amplitude_above_threshold
	
	NAME_TO_ACCESS_TO_THE_SETUP = f'beta scan PID: {os.getpid()}'
	beta_scans_configuration_df = load_beta_scans_configuration()
	beta_scans_configuration_df['software_trigger'] = lambda x: software_trigger(x, 0)
	
	with Alberto.handle_task('automatic_measurements', drop_old_data=False) as beta_scans_task_bureaucrat:
		automatic_measurements(
			bureaucrat = beta_scans_task_bureaucrat.create_subrun(create_a_timestamp() + '_' + input('Measurement name? ').replace(' ','_')),
			name_to_access_to_the_setup = NAME_TO_ACCESS_TO_THE_SETUP,
			beta_scans_configuration_df = beta_scans_configuration_df,
		)
