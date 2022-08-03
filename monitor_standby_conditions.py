from pathlib import Path
import pandas
import datetime
import time
from TheSetup import connect_me_with_the_setup
from huge_dataframe.SQLiteDataFrame import SQLiteDataFrameDumper # https://github.com/SengerM/huge_dataframe
import threading
import warnings
from progressreporting.TelegramProgressReporter import TelegramReporter # https://github.com/SengerM/progressreporting
import my_telegram_bots
import traceback
import numpy
import os

THREADS_SLEEPING_SECONDS = 1

# ----------------------------------------------------------------------
PATH_TO_DIRECTORY_WHERE_TO_STORE_DATA = Path.home()/Path('monitor_standby_conditions')
NAME_TO_ACCESS_TO_THE_SETUP = f'monitor standby conditions {os.getpid()}'
# ----------------------------------------------------------------------

def measure_data(slot_number:int, average_at_least_during_seconds:float, average_at_least_n_samples:int)->dict:
	the_setup = connect_me_with_the_setup()
	bias_voltages = []
	bias_currents = []
	start_measuring = time.time()
	while time.time()-start_measuring < average_at_least_during_seconds or len(bias_voltages) < average_at_least_n_samples:
		bias_voltages.append(the_setup.measure_bias_voltage(slot_number))
		bias_currents.append(the_setup.measure_bias_current(slot_number))
		time.sleep(.3) # To avoid overloading communications with hardware.
	measured_data = {
		'device_name': the_setup.get_name_of_device_in_slot_number(slot_number),
		'Temperature (°C)': the_setup.measure_temperature(),
		'Humidity (%RH)': the_setup.measure_humidity(),
		'Bias voltage (V)': numpy.mean(bias_voltages),
		'Bias current (A)': numpy.mean(bias_currents),
		'Bias voltage (V) std': numpy.std(bias_voltages),
		'Bias current (A) std': numpy.std(bias_currents),
		'When': datetime.datetime.now(),
	}
		
	return measured_data

if __name__=='__main__':
	PATH_TO_DIRECTORY_WHERE_TO_STORE_DATA.mkdir(exist_ok=True)

	the_setup = connect_me_with_the_setup()

	keep_threads_alive = True

	data_to_dump_Lock = threading.RLock()
	data_to_dump = []

	def monitor_one_slot(slot_number:int):
		while keep_threads_alive:
			try:
				standby_configuration = pandas.read_csv(Path('configuration_files/standby_configuration.csv'), index_col='slot_number', dtype={'slot_number': int, 'Bias voltage (V)': float, 'Current compliance (A)': float, 'Measure once every (s)': float}).loc[slot_number]
			except FileNotFoundError as e:
				warnings.warn(f'Cannot read standby configuration file, reason: `{e}`. Will ignore this and try again.')
				time.sleep(THREADS_SLEEPING_SECONDS)
				continue
			if 'last_time_I_measured' not in locals() or (datetime.datetime.now()-last_time_I_measured).seconds > standby_configuration['Measure once every (s)']:
				if the_setup.is_bias_slot_number_being_hold_by_someone(slot_number):
					measured_data = measured_data = measure_data(
						slot_number = slot_number, 
						average_at_least_during_seconds = 11,
						average_at_least_n_samples = 11,
					)
				else:
					with the_setup.hold_control_of_bias_for_slot_number(slot_number = slot_number, who = NAME_TO_ACCESS_TO_THE_SETUP):
						the_setup.set_current_compliance(slot_number=slot_number, amperes=standby_configuration['Current compliance (A)'], who=NAME_TO_ACCESS_TO_THE_SETUP)
						the_setup.set_bias_voltage(slot_number=slot_number, volts=standby_configuration['Bias voltage (V)'], who=NAME_TO_ACCESS_TO_THE_SETUP)
						measured_data = measure_data(
							slot_number = slot_number, 
							average_at_least_during_seconds = 11,
							average_at_least_n_samples = 11,
						)
				last_time_I_measured = measured_data['When']
				measured_data_df = pandas.DataFrame(
					measured_data,
					index = [0],
				)
				measured_data_df.set_index('device_name', inplace=True)
				with data_to_dump_Lock:
					data_to_dump.append(measured_data_df)
			time.sleep(THREADS_SLEEPING_SECONDS)

	reporter = TelegramReporter(
		telegram_token = my_telegram_bots.robobot.token,
		telegram_chat_id = my_telegram_bots.chat_ids['Long term tests setup'],
	)

	with SQLiteDataFrameDumper(PATH_TO_DIRECTORY_WHERE_TO_STORE_DATA/Path('measured_data.sqlite'), dump_after_n_appends=1e3, dump_after_seconds=10, delete_database_if_already_exists=False) as measured_data_dumper:
		threads = []
		for slot_number in the_setup.get_slots_configuration_df().index:
			thread = threading.Thread(target=monitor_one_slot, args=(slot_number,))
			threads.append(thread)
		
		for thread in threads:
			thread.start()
		
		try:
			while True:
				if len(data_to_dump) > 0:
					with data_to_dump_Lock:
						data_df = pandas.concat(data_to_dump).sort_values('When')
						measured_data_dumper.append(data_df)
						data_to_dump = []
				time.sleep(THREADS_SLEEPING_SECONDS)
		except Exception as e:
			print(traceback.format_exc())
			print(e)
			reporter.send_message(f'🔥 `monitor_standby_conditions.py` has just crashed. Reason: {e}')
		finally:
			keep_threads_alive = False
			while any([thread.is_alive() for thread in threads]):
				print(f'{sum([thread.is_alive() for thread in threads])} threads are still alive...')
				time.sleep(THREADS_SLEEPING_SECONDS)
	
