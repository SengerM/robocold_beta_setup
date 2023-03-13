from the_bureaucrat.bureaucrats import RunBureaucrat # https://github.com/SengerM/the_bureaucrat
from pathlib import Path
import pandas
from TheSetup import connect_me_with_the_setup, load_beta_scans_configuration
import numpy
from beta_scan import beta_scan_sweeping_bias_voltage
from auto_trigger_rate import auto_trigger_rate_sweeping_trigger_level_and_bias_voltage
from utils import create_a_timestamp
from grafica.plotly_utils.utils import set_my_template_as_default # https://github.com/SengerM/grafica
from progressreporting.TelegramProgressReporter import SafeTelegramReporter4Loops # https://github.com/SengerM/progressreporting
import my_telegram_bots # Here I keep the info from my bots, never make it public!

def automatic_measurements(bureaucrat:RunBureaucrat, name_to_access_to_the_setup:str, beta_scans_configuration_df:pandas.DataFrame, silent:bool=False, reporter:SafeTelegramReporter4Loops=None):
	the_setup = connect_me_with_the_setup()
	with bureaucrat.handle_task('automatic_measurements', drop_old_data=False) as employee:
		if not silent:
			print(f'Waiting to acquire control of Robocold...')
		with the_setup.hold_control_of_robocold(who=name_to_access_to_the_setup):
			if not silent:
				print('Control of Robocold acquired!')
			with reporter.report_loop(len(beta_scans_configuration_df.index.unique()), bureaucrat.run_name) if reporter is not None else nullcontext() as reporter:
				for slot_number in beta_scans_configuration_df.index.unique():
					with the_setup.hold_control_of_bias_for_slot_number(slot_number=slot_number, who=name_to_access_to_the_setup):
						John = employee.create_subrun(f'{create_a_timestamp()}_{the_setup.get_name_of_device_in_slot_number(slot_number)}_Chubut1')
						
						# Beta scan -----
						if not silent:
							print(f'Reseting robocold...')
						the_setup.reset_robocold(who=name_to_access_to_the_setup)
						if not silent:
							print(f'Starting beta scans sweeping bias voltage on slot {slot_number}...')
						beta_scan_sweeping_bias_voltage(
							bureaucrat = John,
							name_to_access_to_the_setup = name_to_access_to_the_setup,
							slot_number = slot_number,
							n_triggers_per_voltage = beta_scans_configuration_df.loc[slot_number,'n_triggers'], 
							bias_voltages = beta_scans_configuration_df.loc[slot_number,'Bias voltage (V)'],
							software_triggers = beta_scans_configuration_df.loc[slot_number,'software_trigger'],
							silent = silent,
							reporter = reporter.create_subloop_reporter(),
						)
						if not silent:
							print(f'Beta scan sweeping bias voltage on slot {slot_number} finished.')
						
						if not silent:
							print(f'Automatic measurements on slot {slot_number} finished :)')
						reporter.update(1)

if __name__=='__main__':
	import os
	from configuration_files.current_run import Alberto
	PATH_TO_ANALYSIS_SCRIPTS = Path(__file__).resolve().parent.parent/'analysis_scripts'
	import sys
	sys.path.append(str(PATH_TO_ANALYSIS_SCRIPTS))
	from plot_beta_scan import plot_everything_from_beta_scan
	
	set_my_template_as_default()
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
	
	the_setup = connect_me_with_the_setup()
	for slot_number in beta_scans_configuration_df.index.unique():
		the_setup.set_current_compliance(slot_number=slot_number, amperes=99e-6, who=NAME_TO_ACCESS_TO_THE_SETUP)
	
	automatic_measurements(
		bureaucrat = Alberto,
		name_to_access_to_the_setup = NAME_TO_ACCESS_TO_THE_SETUP,
		beta_scans_configuration_df = beta_scans_configuration_df,
		reporter = SafeTelegramReporter4Loops(
			bot_token = my_telegram_bots.robobot.token,
			chat_id = my_telegram_bots.chat_ids['Robobot beta setup'],
		)
	)
