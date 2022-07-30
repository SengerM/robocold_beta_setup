from bureaucrat.SmarterBureaucrat import SmarterBureaucrat # https://github.com/SengerM/bureaucrat
from pathlib import Path
import pandas
import datetime
import time
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from TheSetup import TheRobocoldBetaSetup
import datetime
from huge_dataframe.SQLiteDataFrame import SQLiteDataFrameDumper, load_whole_dataframe # https://github.com/SengerM/huge_dataframe

def script_core(directory, the_setup:TheRobocoldBetaSetup, voltages:list, slot_number:int, n_measurements_per_voltage:int, silent=False):
	"""Measure an IV curve.
	Parameters
	----------
	directory: Path
		Path to the directory where to store the data.
	n_measurements_per_voltage: int
		Number of measurements to perform at each voltage.
	the_setup: TheRobocoldBetaSetup
		An instance of `TheRobocoldBetaSetup` to control the hardware.
	voltages: list of float
		A list with the voltage values.
	slot_number: int
		The number of slot in which to measure the IV curve.
	silent: bool, default False
		If `True`, no progress messages are printed.
	"""
	
	John = SmarterBureaucrat(
		directory,
		new_measurement = True,
		_locals = locals(),
	)
	
	with the_setup.hold_control_of_bias_for_slot_number(slot_number):
		with John.do_your_magic():
			with SQLiteDataFrameDumper(John.path_to_default_output_directory/Path('measured_data.sqlite'), dump_after_n_appends=1e3, dump_after_seconds=10) as measured_data_dumper:
				for n_voltage,voltage in enumerate(voltages):
					if not silent:
						print(f'Measuring n_voltage={n_voltage}/{len(voltages)-1}...')
					the_setup.set_bias_voltage(slot_number, voltage)
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
									'Temperature (°C)': the_setup.temperature,
									'Humidity (%RH)': the_setup.humidity,
									'device_name': the_setup.get_name_of_device_in_slot_number(slot_number),
								},
								index = [0],
							)
							elapsed_seconds = measurement_started - time.time()
						measured_data_df.set_index(['n_voltage','n_measurement'], inplace=True)
						measured_data_dumper.append(measured_data_df)
						time.sleep(.5)
	
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

if __name__ == '__main__':
	import time
	import threading
	import numpy
	
	measure_iv_curve = script_core
	
	class MeasureIVCurve(threading.Thread):
		def __init__(self, name:str, slot_number:int, the_setup:TheRobocoldBetaSetup, voltages_to_measure:list, n_measurements_per_voltage:int, directory_to_store_data:Path):
			threading.Thread.__init__(self)
			self.name = name
			self.slot_number = slot_number
			self.the_setup = the_setup
			self.voltages_to_measure = voltages_to_measure
			self.n_measurements_per_voltage = n_measurements_per_voltage
			self.directory_to_store_data = directory_to_store_data
		def run(self):
			print(f'Starting thread {self.name} to measure device {the_setup.get_name_of_device_in_slot_number(self.slot_number)}...')
			measure_iv_curve(
				directory = self.directory_to_store_data/Path(f'IV_curve_{the_setup.get_name_of_device_in_slot_number(self.slot_number)}'),
				the_setup = self.the_setup, 
				voltages = self.voltages_to_measure, 
				slot_number = self.slot_number, 
				n_measurements_per_voltage = self.n_measurements_per_voltage, 
				silent = True,
			)
	
	VOLTAGES = {
		1: 500,
		2: 500,
		3: 500,
		4: 500,
		5: 500,
		6: 500,
	}
	
	Richard = SmarterBureaucrat(
		Path.home()/Path('measurements_data')/Path('IV_curves'),
		new_measurement = True,
		_locals = locals(),
	)
	
	the_setup = TheRobocoldBetaSetup(path_to_configuration_file = Path('configuration.csv'))
	
	print(the_setup.description)
	print(the_setup.configuration_df)
	
	with Richard.do_your_magic():
		the_setup.configuration_df.to_csv(Richard.path_to_default_output_directory/Path('setup_configuration.csv'))
		with open(Richard.path_to_default_output_directory/Path('setup_description.txt'), 'w') as ofile:
			print(the_setup.description, file=ofile)
		
		threads = []
		for slot_number in VOLTAGES.keys():
			the_setup.set_current_compliance(slot_number=slot_number, amperes=10e-6)
			thread = MeasureIVCurve(
				name = f'IV measuring thread for slot {slot_number}',
				slot_number = slot_number,
				the_setup = the_setup,
				voltages_to_measure = numpy.linspace(0,VOLTAGES[slot_number],111),
				n_measurements_per_voltage = 11,
				directory_to_store_data = Richard.path_to_submeasurements_directory,
			)
			threads.append(thread)
		
		print(f'Moving the beta source outside all detectors...')
		the_setup.place_source_such_that_it_does_not_irradiate_any_DUT()
		
		for thread in threads:
			thread.start()
		
		while any([thread.is_alive() for thread in threads]):
			time.sleep(1)
		
		measured_data_list = []
		for path_to_submeasurement in Richard.path_to_submeasurements_directory.iterdir():
			measured_data_list.append(
				load_whole_dataframe(path_to_submeasurement/Path('measure_iv_curve/measured_data.sqlite')).reset_index(),
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
			averaged_data_df.sort_values('n_voltage'),
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
