from the_bureaucrat.bureaucrats import RunBureaucrat # https://github.com/SengerM/the_bureaucrat
from pathlib import Path
import pandas
import datetime
import time
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from TheSetup import connect_me_with_the_setup, load_beta_scans_configuration
from huge_dataframe.SQLiteDataFrame import SQLiteDataFrameDumper, load_whole_dataframe # https://github.com/SengerM/huge_dataframe
import threading
import warnings
from signals.PeakSignal import PeakSignal, draw_in_plotly # https://github.com/SengerM/signals
from progressreporting.TelegramProgressReporter import SafeTelegramReporter4Loops # https://github.com/SengerM/progressreporting
import numpy
from contextlib import nullcontext

def parse_waveform(signal:PeakSignal):
	parsed = {
		'Amplitude (V)': signal.amplitude,
		'Noise (V)': signal.noise,
		'Rise time (s)': signal.rise_time,
		'Collected charge (V s)': signal.peak_integral,
		'Time over noise (s)': signal.time_over_noise,
		'Peak start time (s)': signal.peak_start_time,
		'Whole signal integral (V s)': signal.integral_from_baseline,
		'SNR': signal.SNR
	}
	for threshold_percentage in [10,20,30,40,50,60,70,80,90]:
		try:
			time_over_threshold = signal.find_time_over_threshold(threshold_percentage)
		except Exception:
			time_over_threshold = float('NaN')
		parsed[f'Time over {threshold_percentage}% (s)'] = time_over_threshold
	for pp in [10,20,30,40,50,60,70,80,90]:
		try:
			time_at_this_pp = float(signal.find_time_at_rising_edge(pp))
		except Exception:
			time_at_this_pp = float('NaN')
		parsed[f't_{pp} (s)'] = time_at_this_pp
	return parsed

def trigger_and_measure_dut_stuff(name_to_access_to_the_setup:str, slot_number:int):
	the_setup = connect_me_with_the_setup()
	elapsed_seconds = 9999
	while elapsed_seconds > 5: # Because of multiple threads locking the different elements of the_setup, it can happen that this gets blocked for a long time. Thus, the measured data will no longer belong to a single point in time as we expect...:
		the_setup.wait_for_trigger(who=name_to_access_to_the_setup)
		trigger_time = time.time()
		measured_stuff = {
			'Bias voltage (V)': the_setup.measure_bias_voltage(slot_number),
			'Bias current (A)': the_setup.measure_bias_current(slot_number),
			'Temperature (°C)': the_setup.measure_temperature(),
			'Humidity (%RH)': the_setup.measure_humidity(),
			'device_name': the_setup.get_name_of_device_in_slot_number(slot_number),
		}
		elapsed_seconds = trigger_time - time.time()
	return measured_stuff

def plot_waveform(signal):
	fig = draw_in_plotly(signal)
	fig.update_layout(
		xaxis_title = "Time (s)",
		yaxis_title = "Amplitude (V)",
	)
	MARKERS = { # https://plotly.com/python/marker-style/#custom-marker-symbols
		10: 'circle',
		20: 'square',
		30: 'diamond',
		40: 'cross',
		50: 'x',
		60: 'star',
		70: 'hexagram',
		80: 'star-triangle-up',
		90: 'star-triangle-down',
	}
	for pp in [10,20,30,40,50,60,70,80,90]:
		try:
			fig.add_trace(
				go.Scatter(
					x = [signal.find_time_at_rising_edge(pp)],
					y = [signal(signal.find_time_at_rising_edge(pp))],
					mode = 'markers',
					name = f'Time at {pp} %',
					marker=dict(
						color = 'rgba(0,0,0,.5)',
						size = 11,
						symbol = MARKERS[pp]+'-open-dot',
						line = dict(
							color = 'rgba(0,0,0,.5)',
							width = 2,
						)
					),
				)
			)
		except Exception as e:
			pass
	return fig

def beta_scan(bureaucrat:RunBureaucrat, name_to_access_to_the_setup:str, slot_number:int, n_triggers:int, bias_voltage:float, software_trigger=None, silent=False, reporter:SafeTelegramReporter4Loops=None):
	"""Perform a beta scan.
	
	Parameters
	----------
	bureaucrat: RunBureaucrat
		The bureaucrat that will manage this measurement.
	n_triggers: int
		Number of triggers to record.
	name_to_access_to_the_setup: str
		Name to use when accessing to the setup.
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
	reporter: SafeTelegramReporter4Loops, optional
		An instance of `SafeTelegramReporter4Loops` to report the progress.
	
	Returns
	-------
	path_to_measurement_base_directory: Path
		A path to the directory where the measurement's data was stored.
	"""
	
	John = bureaucrat
	John.create_run(if_exists='skip')
	
	the_setup = connect_me_with_the_setup()
	
	if not silent:
		print('Waiting for acquiring control of the hardware...')
	with the_setup.hold_signal_acquisition(who=name_to_access_to_the_setup), the_setup.hold_control_of_bias_for_slot_number(slot_number, who=name_to_access_to_the_setup), the_setup.hold_control_of_robocold(who=name_to_access_to_the_setup):
		if not silent:
			print('Control of hardware acquired.')
		with John.handle_task('beta_scan', allowed_exceptions={KeyboardInterrupt}) as beta_scan_task_bureaucrat:
			the_setup.set_trigger_for_beta_scans(who=name_to_access_to_the_setup)
			if not silent:
				print(f'Moving beta source to slot {slot_number}...')
			the_setup.move_to_slot(slot_number=slot_number, who=name_to_access_to_the_setup)
			if not silent:
				print(f'Connecting slot {slot_number} to the oscilloscope...')
			the_setup.connect_slot_to_oscilloscope(slot_number=slot_number, who=name_to_access_to_the_setup)
			with open(beta_scan_task_bureaucrat.path_to_directory_of_my_task/'setup_description.txt','w') as ofile:
				print(the_setup.get_description(), file=ofile)
			the_setup.get_slots_configuration_df().to_csv(beta_scan_task_bureaucrat.path_to_directory_of_my_task/'slots_configuration.csv')
			with SQLiteDataFrameDumper(beta_scan_task_bureaucrat.path_to_directory_of_my_task/Path('measured_stuff.sqlite'), dump_after_n_appends=1e3, dump_after_seconds=66) as measured_stuff_dumper, \
				SQLiteDataFrameDumper(beta_scan_task_bureaucrat.path_to_directory_of_my_task/Path('waveforms.sqlite'), dump_after_n_appends=1e3, dump_after_seconds=66) as waveforms_dumper, \
				SQLiteDataFrameDumper(beta_scan_task_bureaucrat.path_to_directory_of_my_task/Path('parsed_from_waveforms.sqlite'), dump_after_n_appends=1e3, dump_after_seconds=66) as parsed_from_waveforms_dumper \
			:
				if not silent:
					print(f'Setting bias voltage {bias_voltage} V to slot number {slot_number}...')
				the_setup.set_bias_voltage(slot_number=slot_number, volts=bias_voltage, who=name_to_access_to_the_setup)
				
				with reporter.report_loop(n_triggers, John.run_name) if reporter is not None else nullcontext() as reporter:
					n_waveform = -1
					for n_trigger in range(n_triggers):
						# Acquire ---
						if not silent:
							print(f'Acquiring n_trigger={n_trigger}/{n_triggers-1}...')
						
						do_they_like_this_trigger = False
						while do_they_like_this_trigger == False:
							this_trigger_measured_stuff = trigger_and_measure_dut_stuff(slot_number=slot_number, name_to_access_to_the_setup=name_to_access_to_the_setup) # Hold here until there is a trigger.
							
							this_trigger_measured_stuff['When'] = datetime.datetime.now()
							this_trigger_measured_stuff['n_trigger'] = n_trigger
							this_trigger_measured_stuff_df = pandas.DataFrame(this_trigger_measured_stuff, index=[0]).set_index(['n_trigger'])
							
							this_trigger_waveforms_dict = {}
							for signal_name in the_setup.get_oscilloscope_configuration_df().index:
								waveform_data = the_setup.get_waveform(oscilloscope_channel_number = the_setup.get_oscilloscope_configuration_df().loc[signal_name,'n_channel'])
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
						for signal_name in the_setup.get_oscilloscope_configuration_df().index:
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
						
						# Plot some of the signals ---
						if numpy.random.rand()<20/n_triggers or n_trigger < 2:
							for signal_name in the_setup.get_oscilloscope_configuration_df().index:
								fig = plot_waveform(this_trigger_waveforms_dict[signal_name])
								fig.update_layout(
									title = f'n_trigger {n_trigger}, signal_name {signal_name}<br><sup>Run: {John.run_name}</sup>',
								)
								path_to_save_plots = beta_scan_task_bureaucrat.path_to_directory_of_my_task/Path('plots of some of the waveforms')
								path_to_save_plots.mkdir(exist_ok=True)
								fig.write_html(
									str(path_to_save_plots/Path(f'n_trigger {n_trigger} signal_name {signal_name}.html')),
									include_plotlyjs = 'cdn',
								)
						reporter.update(1) if reporter is not None else None
	
	if not silent:
		print('Beta scan finished.')

def beta_scan_sweeping_bias_voltage(bureaucrat:RunBureaucrat, name_to_access_to_the_setup:str, slot_number:int, n_triggers_per_voltage:list, bias_voltages:list, software_triggers:list=None, silent=False, reporter:SafeTelegramReporter4Loops=None):
	"""Perform multiple beta scans at different bias voltages each.
	
	Parameters
	----------
	bureaucrat: RunBureaucrat
		The bureaucrat that will handle this measurement.
	n_triggers_per_voltage: list of int
		Number of triggers to record on each individual beta scan.
	name_to_access_to_the_setup: str
		Name to use when accessing to the setup.
	slot_number: int
		The number of slot in which to measure the IV curve.
	software_trigger: list of callable, optional
		See documentation on `beta_scan`, in this case it is just a list
		of such objects one for each voltage.
	bias_voltage: list of float
		The bias voltages at which to measure.
	silent: bool, default False
		If `True`, no progress messages are printed.
	reporter: SafeTelegramReporter4Loops, optional
		An instance of `SafeTelegramReporter4Loops` to report the progress.
	
	Returns
	-------
	path_to_measurement_base_directory: Path
		A path to the directory where the measurement's data was stored.
	"""
	John = bureaucrat
	John.create_run(if_exists='skip')
	
	the_setup = connect_me_with_the_setup()
	
	if software_triggers is not None and len(bias_voltages) != len(software_triggers):
		raise ValueError(f'The length of `software_triggers` must be the same as the length of `bias_voltages` as one trigger is for each bias voltage.')

	if not silent:
		print('Waiting for acquiring control of the hardware...')
	with the_setup.hold_signal_acquisition(who=name_to_access_to_the_setup), the_setup.hold_control_of_bias_for_slot_number(slot_number, who=name_to_access_to_the_setup), the_setup.hold_control_of_robocold(who=name_to_access_to_the_setup):
		if not silent:
			print('Control of hardware acquired.')
		with John.handle_task('beta_scan_sweeping_bias_voltage', allowed_exceptions={KeyboardInterrupt}) as beta_scan_sweeping_bias_voltage_task_bureaucrat:
			with open(beta_scan_sweeping_bias_voltage_task_bureaucrat.path_to_directory_of_my_task/'setup_description.txt','w') as ofile:
				print(the_setup.get_description(), file=ofile)
			with reporter.report_loop(len(bias_voltages), John.run_name) if reporter is not None else nullcontext() as reporter:
				if software_triggers is None:
					software_triggers = [lambda x: True for v in bias_voltages]
				for bias_voltage,software_trigger,n_triggers in zip(bias_voltages,software_triggers,n_triggers_per_voltage):
					p = beta_scan(
						beta_scan_sweeping_bias_voltage_task_bureaucrat.create_subrun(f'{John.run_name}_{int(bias_voltage)}V'),
						name_to_access_to_the_setup = name_to_access_to_the_setup,
						slot_number = slot_number,
						n_triggers = n_triggers,
						bias_voltage = bias_voltage,
						software_trigger = software_trigger,
						silent = silent,
						reporter = reporter.create_subloop_reporter() if reporter is not None else None,
					)
					reporter.update(1) if reporter is not None else None
