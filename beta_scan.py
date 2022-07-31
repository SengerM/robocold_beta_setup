from bureaucrat.SmarterBureaucrat import SmarterBureaucrat # https://github.com/SengerM/bureaucrat
from pathlib import Path
import pandas
import datetime
import time
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from TheSetup import TheRobocoldBetaSetup
from huge_dataframe.SQLiteDataFrame import SQLiteDataFrameDumper, load_whole_dataframe # https://github.com/SengerM/huge_dataframe
import threading
import warnings
from signals.PeakSignal import PeakSignal, draw_in_plotly
from progressreporting.TelegramProgressReporter import TelegramReporter # https://github.com/SengerM/progressreporting

def parse_waveform(signal:PeakSignal):
	return {
		'Amplitude (V)': signal.amplitude,
		'Noise (V)': signal.noise,
		'Rise time (s)': signal.rise_time,
		'Collected charge (V s)': signal.peak_integral,
		'Time over noise (s)': signal.time_over_noise,
		'Peak start time (s)': signal.peak_start_time,
	}

def trigger_and_measure_dut_stuff(the_setup:TheRobocoldBetaSetup, slot_number:int):
	elapsed_seconds = 9999
	while elapsed_seconds > 5: # Because of multiple threads locking the different elements of the_setup, it can happen that this gets blocked for a long time. Thus, the measured data will no longer belong to a single point in time as we expect...:
		the_setup.wait_for_trigger()
		trigger_time = time.time()
		measured_stuff = {
			'Bias voltage (V)': the_setup.measure_bias_voltage(slot_number),
			'Bias current (A)': the_setup.measure_bias_current(slot_number),
			'Temperature (°C)': the_setup.temperature,
			'Humidity (%RH)': the_setup.humidity,
			'device_name': the_setup.get_name_of_device_in_slot_number(slot_number),
		}
		elapsed_seconds = trigger_time - time.time()
	return measured_stuff

def script_core(path_to_directory_in_which_to_store_data:Path, measurement_name:str, the_setup:TheRobocoldBetaSetup, slot_number:int, n_triggers:int, bias_voltage:float, software_trigger=None, silent=False, telegram_progress_reporter:TelegramReporter=None)->Path:
	"""Perform a beta scan.
	
	Parameters
	----------
	path_to_directory_in_which_to_store_data: Path
		Path to the directory where to store the data.
	measurement_name: str
		A name for the measurement.
	n_triggers: int
		Number of triggers to record.
	the_setup: TheRobocoldBetaSetup
		An instance of `TheRobocoldBetaSetup` to control the hardware.
	slot_number: int
		The number of slot in which to measure the IV curve.
	software_trigger: callable, optional
		A callable that receives a dictionary of waveforms of type `PeakSignal`
		and returns `True` or `False`, that will be called for each trigger.
		If `software_trigger(waveforms_dict)` returns `True`, the trigger
		will be considered as nice, otherwise it will be discarded and a
		new trigger will be taken. Example:
		```
		def software_trigger(signals_dict):
			DUT_signal = signals_dict['DUT']
			PMT_signal = signals_dict['reference_trigger']
			return abs(DUT_signal.peak_start_time - PMT_signal.peak_start_time) < 2e-9
		```
	bias_voltage: float
		The value for the voltage.
	silent: bool, default False
		If `True`, no progress messages are printed.
	telegram_progress_reporter: TelegramReporter, optional
		A reporter to update and/or send warnings.
	
	Returns
	-------
	path_to_measurement_base_directory: Path
		A path to the directory where the measurement's data was stored.
	"""
	
	John = SmarterBureaucrat(
		path_to_directory_in_which_to_store_data/Path(measurement_name),
		new_measurement = True,
		_locals = locals(),
	)
	
	with the_setup.hold_signal_acquisition():
		with the_setup.hold_control_of_bias_for_slot_number(slot_number):
			with the_setup.hold_control_of_robocold():
				with John.do_your_magic():
					with SQLiteDataFrameDumper(John.path_to_default_output_directory/Path('measured_stuff.sqlite'), dump_after_n_appends=1e3, dump_after_seconds=66) as measured_stuff_dumper:
						with SQLiteDataFrameDumper(John.path_to_default_output_directory/Path('waveforms.sqlite'), dump_after_n_appends=1e3, dump_after_seconds=66) as waveforms_dumper:
							with SQLiteDataFrameDumper(John.path_to_default_output_directory/Path('parsed_from_waveforms.sqlite'), dump_after_n_appends=1e3, dump_after_seconds=66) as parsed_from_waveforms_dumper:
								
								if not silent:
									print(f'Moving beta source to slot number {slot_number}...')
								the_setup.move_to_slot(slot_number)
								if not silent:
									print(f'Connecting oscilloscope to slot number {slot_number}...')
								the_setup.connect_slot_to_oscilloscope(slot_number)
								if not silent:
									print(f'Setting bias voltage {bias_voltage} V to slot number {slot_number}...')
								the_setup.set_bias_voltage(slot_number=slot_number, volts=bias_voltage)
								
								n_waveform = -1
								for n_trigger in range(n_triggers):
									# Acquire ---
									if not silent:
										print(f'Acquiring n_trigger={n_trigger}/{n_triggers-1}...')
									
									do_they_like_this_trigger = False
									while do_they_like_this_trigger == False:
										this_trigger_measured_stuff = trigger_and_measure_dut_stuff(the_setup, slot_number) # Hold here until there is a trigger.
										
										this_trigger_measured_stuff['When'] = datetime.datetime.now()
										this_trigger_measured_stuff['n_trigger'] = n_trigger
										this_trigger_measured_stuff_df = pandas.DataFrame(this_trigger_measured_stuff, index=[0]).set_index(['n_trigger'])
										
										this_trigger_waveforms_dict = {}
										for signal_name in the_setup.oscilloscope_configuration_df.index:
											waveform_data = the_setup.get_waveform(oscilloscope_channel_number = the_setup.oscilloscope_configuration_df.loc[signal_name,'n_channel'])
											this_trigger_waveforms_dict[signal_name] = PeakSignal(
												time = waveform_data['Time (s)'],
												samples = waveform_data['Amplitude (V)']
											)
										
										if software_trigger is None:
											do_they_like_this_trigger = True
										else:
											do_they_like_this_trigger = software_trigger(this_trigger_waveforms_dict)
									
									# Parse and save data ---
									measured_stuff_dumper.append(this_trigger_measured_stuff_df)
									for signal_name in the_setup.oscilloscope_configuration_df.index:
										n_waveform += 1
										
										waveform_df = pandas.DataFrame({'Time (s)': this_trigger_waveforms_dict[signal_name].time, 'Amplitude (V)': this_trigger_waveforms_dict[signal_name].samples})
										waveform_df['n_waveform'] = n_waveform
										waveform_df.set_index('n_waveform', inplace=True)
										waveforms_dumper.append(waveform_df)
										
										parsed_from_waveform = parse_waveform(this_trigger_waveforms_dict[signal_name])
										parsed_from_waveform['n_trigger'] = n_trigger
										parsed_from_waveform['signal_name'] = signal_name
										parsed_from_waveform['n_waveform'] = n_waveform
										parsed_from_waveform_df = pandas.DataFrame(
											parsed_from_waveform,
											index = [0],
										).set_index(['n_trigger','signal_name'])
										parsed_from_waveforms_dumper.append(parsed_from_waveform_df)
									
									if telegram_progress_reporter is not None:
										telegram_progress_reporter.update(1) 
									
	return John.path_to_measurement_base_directory


if __name__=='__main__':
	import numpy
	import my_telegram_bots
	from plot_everything_from_beta_scan import script_core as plot_everything_from_beta_scan
	
	N_TRIGGERS = 111
	MEASUREMENT_NAME = input('Measurement name? ').replace(' ','_')
	
	the_setup = TheRobocoldBetaSetup(
		path_to_slots_configuration_file = Path('slots_configuration.csv'),
		path_to_oscilloscope_configuration_file = Path('oscilloscope_configuration.csv'),
	)
	
	reporter = TelegramReporter(
		telegram_token = my_telegram_bots.robobot.token,
		telegram_chat_id = my_telegram_bots.chat_ids['Robobot beta setup'],
	)
	
	def software_trigger(signals_dict):
		DUT_signal = signals_dict['DUT']
		PMT_signal = signals_dict['reference_trigger']
		return 1e-9 < DUT_signal.peak_start_time - PMT_signal.peak_start_time < 5e-9
	
	with reporter.report_for_loop(N_TRIGGERS, MEASUREMENT_NAME) as reporter:
		p = script_core(
			path_to_directory_in_which_to_store_data = Path.home()/Path('measurements_data'), 
			measurement_name = MEASUREMENT_NAME, 
			the_setup = the_setup, 
			slot_number = 2, 
			n_triggers = N_TRIGGERS, 
			bias_voltage = 500,
			silent = False, 
			telegram_progress_reporter = reporter,
			software_trigger = software_trigger,
		)
		plot_everything_from_beta_scan(p)
