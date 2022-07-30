from bureaucrat.SmarterBureaucrat import SmarterBureaucrat # https://github.com/SengerM/bureaucrat
from pathlib import Path
import pandas
import numpy as np
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
