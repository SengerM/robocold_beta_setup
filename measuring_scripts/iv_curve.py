from the_bureaucrat.bureaucrats import RunBureaucrat # https://github.com/SengerM/the_bureaucrat
from pathlib import Path
import pandas
import datetime
import time
from TheSetup import connect_me_with_the_setup
from huge_dataframe.SQLiteDataFrame import SQLiteDataFrameDumper # https://github.com/SengerM/huge_dataframe
import threading
import warnings
import sys 
sys.path.append(str(Path.home()/'scripts_and_codes/repos/robocold_beta_setup/analysis_scripts'))
from plot_iv_curves import plot_IV_curves_all_together
from progressreporting.TelegramProgressReporter import TelegramReporter # https://github.com/SengerM/progressreporting
import my_telegram_bots

def measure_iv_curve(bureaucrat:RunBureaucrat, voltages:list, slot_number:int, n_measurements_per_voltage:int, name_to_access_to_the_setup:str, current_compliance:float, silent=False, reporter:TelegramReporter=None):
	"""Measure an IV curve.
	Parameters
	----------
	bureaucrat: RunBureaucrat
		The `RunBureaucrat` object that will manage this measurement.
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
	"""
	
	John = bureaucrat
	John.create_run()
	
	the_setup = connect_me_with_the_setup()
	
	report_progress = reporter is not None
	
	if not silent:
		print('Waiting for acquiring control of the hardware.')
	with the_setup.hold_control_of_bias_for_slot_number(slot_number=slot_number, who=name_to_access_to_the_setup):
		if not silent:
			print('Control of hardware acquired.')
		with John.handle_task('measure_iv_curve') as measure_iv_curve_task_handler:
			with open(measure_iv_curve_task_handler.path_to_directory_of_my_task/'setup_description.txt','w') as ofile:
				print(the_setup.get_description(), file=ofile)
			with SQLiteDataFrameDumper(measure_iv_curve_task_handler.path_to_directory_of_my_task/Path('measured_data.sqlite'), dump_after_n_appends=1e3, dump_after_seconds=10) as measured_data_dumper:
				the_setup.set_current_compliance(slot_number=slot_number, amperes=current_compliance, who=name_to_access_to_the_setup)
				for n_voltage,voltage in enumerate(voltages):
					if not silent:
						print(f'Measuring n_voltage={n_voltage}/{len(voltages)-1} on slot {slot_number}...')
					try:
						the_setup.set_bias_voltage(slot_number, voltage, who=name_to_access_to_the_setup)
					except Exception as e:
						if '#BD:00,VAL:ERR' in str(e):
							warnings.warn(f'Cannot measure slot {slot_number} at voltage {voltage}, reason: `{e}`, will skip this point.')
							continue
						else:
							raise e
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
						if report_progress:
							reporter.update(1)
	if not silent:
		print(f'Finished measuring IV curve on slot {slot_number}...')

def measure_iv_curves_on_multiple_slots(bureaucrat:RunBureaucrat, voltages:dict, current_compliances:dict, n_measurements_per_voltage:int, name_to_access_to_the_setup:str, silent:bool=False)->Path:
	"""Measure the IV curve of multiple slots.
	
	Parameters
	----------
	bureaucrat: RunBureaucrat
		The bureaucrat that will manage this measurement.
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
	"""
	
	class MeasureIVCurveThread(threading.Thread):
		def __init__(self, bureaucrat:RunBureaucrat, slot_number:int, voltages_to_measure:list, n_measurements_per_voltage:int, name_to_access_to_the_setup:str, current_compliance:float, silent:bool, reporter:TelegramReporter):
			threading.Thread.__init__(self)
			self.bureaucrat = bureaucrat
			self.slot_number = slot_number
			self.name_to_access_to_the_setup = name_to_access_to_the_setup
			self.voltages_to_measure = voltages_to_measure
			self.n_measurements_per_voltage = n_measurements_per_voltage
			self.current_compliance = current_compliance
			self.silent = silent
			self.reporter = reporter
		def run(self):
			measure_iv_curve(
				bureaucrat = self.bureaucrat,
				name_to_access_to_the_setup = name_to_access_to_the_setup,
				voltages = self.voltages_to_measure, 
				slot_number = self.slot_number, 
				n_measurements_per_voltage = self.n_measurements_per_voltage, 
				current_compliance = self.current_compliance,
				silent = self.silent,
				reporter = self.reporter,
			)
	
	Richard = bureaucrat
	Richard.create_run()
	
	if any([not isinstance(_, dict) for _ in [voltages, current_compliances]]):
		raise TypeError(f'`voltages` and `current_compliances` must be dictionaries, but at least one of them is not...')
	if set(voltages) != set(current_compliances):
		raise ValueError(f'The keys of `voltages` and `current_compliances` do not coincide. They should specify the same slot numbers to measure.')
	
	the_setup = connect_me_with_the_setup()
	
	reporter = TelegramReporter(
		telegram_token = my_telegram_bots.robobot.token, 
		telegram_chat_id = my_telegram_bots.chat_ids['Robobot beta setup'],
	)
	
	with Richard.handle_task('measure_iv_curves_on_multiple_slots') as measure_iv_curves_on_multiple_slots_task_handler, \
		reporter.report_for_loop(sum([len(v) for _,v in voltages.items()])*n_measurements_per_voltage, bureaucrat.run_name) as reporter \
	:
		threads = []
		for slot_number in set(voltages):
			thread = MeasureIVCurveThread(
				bureaucrat = measure_iv_curves_on_multiple_slots_task_handler.create_subrun(subrun_name=f'IV_curve_{the_setup.get_name_of_device_in_slot_number(slot_number)}'),
				slot_number = slot_number,
				name_to_access_to_the_setup = name_to_access_to_the_setup,
				voltages_to_measure = voltages[slot_number],
				n_measurements_per_voltage = n_measurements_per_voltage,
				current_compliance = current_compliances[slot_number],
				silent = silent,
				reporter = reporter,
			)
			threads.append(thread)
	
		with open(measure_iv_curves_on_multiple_slots_task_handler.path_to_directory_of_my_task/Path('setup_description.txt'), 'w') as ofile:
			print(the_setup.get_description(), file=ofile)
		
		if not silent:
			print(f'Waiting to acquire the control of Robocold...')
		with the_setup.hold_control_of_robocold(who=name_to_access_to_the_setup):
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
			print(f'Finished measuring all IV curves.')

if __name__=='__main__':
	import numpy
	import os
	from configuration_files.current_run import Alberto
	from utils import create_a_timestamp
	
	SLOTS = [1,2,3,4,5,6,7,8]
	VOLTAGE_VALUES = list(numpy.linspace(0,777,99))
	VOLTAGE_VALUES += VOLTAGE_VALUES[::-1]
	VOLTAGES_FOR_EACH_SLOT = {slot: VOLTAGE_VALUES for slot in SLOTS}
	CURRENT_COMPLIANCES = pandas.read_csv('configuration_files/standby_configuration.csv').set_index('slot_number')['Current compliance (A)'].to_dict()
	NAME_TO_ACCESS_TO_THE_SETUP = f'IV curves measurement script PID: {os.getpid()}'
	
	with Alberto.handle_task('iv_curves', drop_old_data=False) as iv_curves_task_bureaucrat:
		Mariano = iv_curves_task_bureaucrat.create_subrun(create_a_timestamp() + '_' + input('Measurement name? ').replace(' ','_'))
		measure_iv_curves_on_multiple_slots(
			bureaucrat = Mariano,
			name_to_access_to_the_setup = NAME_TO_ACCESS_TO_THE_SETUP,
			voltages = VOLTAGES_FOR_EACH_SLOT,
			current_compliances = CURRENT_COMPLIANCES,
			n_measurements_per_voltage = 2,
			silent = False,
		)
		print(f'Doing plots...')
		plot_IV_curves_all_together(Mariano)
