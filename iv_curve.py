from bureaucrat.SmarterBureaucrat import SmarterBureaucrat # https://github.com/SengerM/bureaucrat
from pathlib import Path
import pandas
import datetime
import time
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from TheSetup import connect_me_with_the_setup
from huge_dataframe.SQLiteDataFrame import SQLiteDataFrameDumper, load_whole_dataframe # https://github.com/SengerM/huge_dataframe
import threading
import warnings

def measure_iv_curve(path_to_directory_in_which_to_store_data:Path, measurement_name:str, voltages:list, slot_number:int, n_measurements_per_voltage:int, name_to_access_to_the_setup:str, current_compliance:float, silent=False)->Path:
	"""Measure an IV curve.
	Parameters
	----------
	path_to_directory_in_which_to_store_data: Path
		Path to the directory where to store the data.
	measurement_name: str
		A name for the measurement.
	n_measurements_per_voltage: int
		Number of measurements to perform at each voltage.
	voltages: list of float
		A list with the voltage values.
	slot_number: int
		The number of slot in which to measure the IV curve.
	name_to_access_to_the_setup: str
		The name to use when accessing to the setup.
	current_compliance: float
		The value for the current limit.
	silent: bool, default False
		If `True`, no progress messages are printed.
	
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
	
	the_setup = connect_me_with_the_setup()
	
	with the_setup.hold_control_of_bias_for_slot_number(slot_number=slot_number, who=name_to_access_to_the_setup):
		with John.do_your_magic():
			with SQLiteDataFrameDumper(John.path_to_default_output_directory/Path('measured_data.sqlite'), dump_after_n_appends=1e3, dump_after_seconds=10) as measured_data_dumper:
				the_setup.set_current_compliance(slot_number=slot_number, amperes=current_compliance, who=name_to_access_to_the_setup)
				for n_voltage,voltage in enumerate(voltages):
					if not silent:
						print(f'Measuring n_voltage={n_voltage}/{len(voltages)-1}...')
					try:
						the_setup.set_bias_voltage(slot_number, voltage, who=name_to_access_to_the_setup)
					except Exception as e:
						if '#BD:00,VAL:ERR' in str(e):
							warnings.warn(f'Cannot measure slot {slot_number} at voltage {voltage}, reason: `{e}`, will skip this point.')
							continue
					for n_measurement in range(n_measurements_per_voltage):
						elapsed_seconds = 9999999999
						while elapsed_seconds > 5: # Because of multiple threads locking the different elements of the_setup, it can happen that this gets blocked for a long time. Thus, the measured data will no longer belong to a single point in time as we expect...:
							measurement_started = time.time()
							measured_data_df = pandas.DataFrame(
								{
									'n_voltage': n_voltage,
									'n_measurement': n_measurement,
									'When': datetime.datetime.now(),
									'Set voltage (V)': voltage,
									'Bias voltage (V)': the_setup.measure_bias_voltage(slot_number),
									'Bias current (A)': the_setup.measure_bias_current(slot_number),
									'Temperature (°C)': the_setup.measure_temperature(),
									'Humidity (%RH)': the_setup.measure_humidity(),
									'device_name': the_setup.get_name_of_device_in_slot_number(slot_number),
								},
								index = [0],
							)
							elapsed_seconds = measurement_started - time.time()
						measured_data_df.set_index(['n_voltage','n_measurement'], inplace=True)
						measured_data_dumper.append(measured_data_df)
						time.sleep(.5)
	if not silent:
		print(f'Finished measuring IV curve...')
	
	# Finished, now do a plot ---
	measured_data_df = load_whole_dataframe(John.path_to_default_output_directory/Path('measured_data.sqlite'))
	measured_data_df.reset_index(inplace=True)
	measured_data_df['Bias voltage (V)'] *= -1
	fig = px.line(
		measured_data_df.groupby(by='n_voltage').mean(), 
		x = "Bias voltage (V)", 
		y = "Bias current (A)",
		title = f'IV curve<br><sup>Measurement: {John.measurement_name}</sup>',
		markers = True,
	)
	fig.write_html(str(John.path_to_default_output_directory/Path('iv_curve_lin_scale.html')), include_plotlyjs='cdn')
	
	fig.update_yaxes(type="log")
	fig.write_html(str(John.path_to_default_output_directory/Path('iv_curve_log_scale.html')), include_plotlyjs='cdn')
	
	fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.02)
	fig.update_layout(title =  f'Current/Voltage vs time with beta source<br><sup>Measurement: {John.measurement_name}</sup>')
	for row_minus_one, variable in enumerate(['Bias current (A)', 'Bias voltage (V)']):
		fig.add_trace(
			go.Scatter(
				x = measured_data_df['When'],
				y = measured_data_df[variable],
				name = variable,
				mode = 'lines+markers',
			),
			row = row_minus_one + 1,
			col = 1
		)
		fig.update_yaxes(title_text=variable, row=row_minus_one+1, col=1)
		
		if variable == 'Bias voltage (V)':
			fig.add_trace(
				go.Scatter(
					x = measured_data_df['When'],
					y = measured_data_df['Set voltage (V)'],
					name = 'Set voltage (V)',
					mode = 'lines+markers',
				),
				row = row_minus_one + 1,
				col = 1
			)
			fig.update_yaxes(title_text='Voltage (V)', row=row_minus_one+1, col=1)
	fig.update_xaxes(title_text='When', row=row_minus_one+1, col=1)
	fig.write_html(str(John.path_to_default_output_directory/Path(f'iv_vs_time.html')), include_plotlyjs='cdn')

	return John.path_to_measurement_base_directory

def measure_iv_curve_multiple_slots(path_to_directory_in_which_to_store_data:Path, measurement_name:str, voltages:dict, current_compliances:dict, n_measurements_per_voltage:int, name_to_access_to_the_setup:str, silent:bool=False)->Path:
	"""Measure the IV curve of multiple slots.
	
	Parameters
	----------
	path_to_directory_in_which_to_store_data: Path
		Path to the directory where to store the data.
	measurement_name: str
		A name for the measurement.
	name_to_access_to_the_setup: str
		The name to use when accessing to the setup.
	n_measurements_per_voltage: int
		Number of measurements to perform at each voltage.
	voltages: dict of lists of floats
		A dictionary of the form `{int: list of float, int: list of float, ...}`
		specifying for each slot a list of voltages.
	current_compliances: dict of floats
		A dictionary of the form `{int: float, int: float, ...}`
		specifying the current compliance for each slot.
	silent: bool, default False
		If `True`, no progress messages are printed.
	
	Returns
	-------
	path_to_measurement_base_directory: Path
		A path to the directory where the measurement's data was stored.
	"""
	
	class MeasureIVCurveThread(threading.Thread):
		def __init__(self, name:str, slot_number:int, voltages_to_measure:list, n_measurements_per_voltage:int, directory_to_store_data:Path, name_to_access_to_the_setup:str, current_compliance:float, silent:bool):
			threading.Thread.__init__(self)
			self.name = name
			self.slot_number = slot_number
			self.name_to_access_to_the_setup = name_to_access_to_the_setup
			self.voltages_to_measure = voltages_to_measure
			self.n_measurements_per_voltage = n_measurements_per_voltage
			self.directory_to_store_data = directory_to_store_data
			self.current_compliance = current_compliance
			self.silent = silent
		def run(self):
			measure_iv_curve(
				path_to_directory_in_which_to_store_data = self.directory_to_store_data,
				measurement_name = f'IV_curve_{the_setup.get_name_of_device_in_slot_number(self.slot_number)}',
				name_to_access_to_the_setup = name_to_access_to_the_setup,
				voltages = self.voltages_to_measure, 
				slot_number = self.slot_number, 
				n_measurements_per_voltage = self.n_measurements_per_voltage, 
				current_compliance = self.current_compliance,
				silent = self.silent,
			)
	
	Richard = SmarterBureaucrat(
		path_to_directory_in_which_to_store_data/Path(measurement_name),
		new_measurement = True,
		_locals = locals(),
	)
	
	if any([not isinstance(_, dict) for _ in [voltages, current_compliances]]):
		raise TypeError(f'`voltages` and `current_compliances` must be dictionaries, but at least one of them is not...')
	if set(voltages) != set(current_compliances):
		raise ValueError(f'The keys of `voltages` and `current_compliances` do not coincide. They should specify the same slot numbers to measure.')
	
	the_setup = connect_me_with_the_setup()
	
	threads = []
	for slot_number in set(voltages):
		thread = MeasureIVCurveThread(
			name = f'IV measuring thread for slot {slot_number}',
			slot_number = slot_number,
			name_to_access_to_the_setup = name_to_access_to_the_setup,
			voltages_to_measure = voltages[slot_number],
			n_measurements_per_voltage = n_measurements_per_voltage,
			directory_to_store_data = Richard.path_to_submeasurements_directory,
			current_compliance = current_compliances[slot_number],
			silent = silent,
		)
		threads.append(thread)
	
	with Richard.do_your_magic():
		with open(Richard.path_to_default_output_directory/Path('setup_description.txt'), 'w') as ofile:
			print(the_setup.get_description(), file=ofile)
		
		if not silent:
			print(f'Moving the beta source outside all detectors...')
		the_setup.place_source_such_that_it_does_not_irradiate_any_DUT(who=name_to_access_to_the_setup)
		
		for thread in threads:
			if not silent:
				print(f'Starting measurement for slot_number = {thread.slot_number}')
			thread.start()
		
		while any([thread.is_alive() for thread in threads]):
			time.sleep(1)
		
		if not silent:
			print(f'Finished measuring all.')
		
		# Do a plot...
		measured_data_list = []
		for path_to_submeasurement in Richard.path_to_submeasurements_directory.iterdir():
			Felipe = SmarterBureaucrat(
				path_to_submeasurement,
				_locals = locals(),
			)
			measured_data_list.append(
				load_whole_dataframe(Felipe.path_to_output_directory_of_script_named('iv_curve.py')/Path('measured_data.sqlite')).reset_index(),
			)
		measured_data_df = pandas.concat(measured_data_list, ignore_index=True)
		measured_data_df['Bias voltage (V)'] *= -1
		grouped_thing = measured_data_df.groupby(['device_name','n_voltage'])
		averaged_data_df = grouped_thing.mean()
		standard_deviation_data_df = grouped_thing.std()
		for col in {'Bias voltage (V)','Bias current (A)'}:
			averaged_data_df[f'{col} error'] = standard_deviation_data_df[col]
		averaged_data_df.reset_index(inplace=True)
		fig = px.line(
			averaged_data_df.sort_values(['device_name','n_voltage']).reset_index(),
			x = 'Bias voltage (V)',
			y = 'Bias current (A)',
			error_y = 'Bias current (A) error',
			error_x = 'Bias voltage (V) error',
			color = 'device_name',
			markers = True,
			title = f'IV curves<br><sup>Measurement: {Richard.measurement_name}</sup>',
		)
		fig.write_html(
			str(Richard.path_to_default_output_directory/Path('iv_curves_measured.html')),
			include_plotlyjs = 'cdn',
		)
	return Richard.path_to_measurement_base_directory

if __name__=='__main__':
	import numpy
	import os
	
	SLOTS = [1,2,3,4,5,6,7]
	VOLTAGE_VALUES = list(numpy.linspace(0,777,99))
	VOLTAGE_VALUES += VOLTAGE_VALUES[::-1]
	VOLTAGES_FOR_EACH_SLOT = {slot: VOLTAGE_VALUES for slot in SLOTS}
	CURRENT_COMPLIANCES = {slot: 10e-6 for slot in SLOTS}
	NAME_TO_ACCESS_TO_THE_SETUP = f'IV curves measurement script PID: {os.getpid()}'
	
	the_setup = connect_me_with_the_setup()
	
	measure_iv_curve_multiple_slots(
		path_to_directory_in_which_to_store_data = Path.home()/Path('measurements_data'),
		measurement_name = input('Measurement name? ').replace(' ','_'),
		name_to_access_to_the_setup = NAME_TO_ACCESS_TO_THE_SETUP,
		voltages = VOLTAGES_FOR_EACH_SLOT,
		current_compliances = CURRENT_COMPLIANCES,
		n_measurements_per_voltage = 11,
		silent = False,
	)
