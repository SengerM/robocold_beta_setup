from pathlib import Path
import pandas
import datetime
import time
from TheSetup import connect_me_with_the_setup
from huge_dataframe.SQLiteDataFrame import SQLiteDataFrameDumper # https://github.com/SengerM/huge_dataframe

# ----------------------------------------------------------------------
PATH_TO_DIRECTORY_WHERE_TO_STORE_DATA = Path.home()/Path('monitor_standby_conditions')
NAME_TO_ACCESS_TO_THE_SETUP = 'monitor standby conditions'
SILENT = False
STANDBY_VOLTAGES = {
	1: 350,
	2: 544,
	3: 277,
	4: 99,
	5: 500,
	6: 200,
	7: 190,
}
# ----------------------------------------------------------------------

def measure_data(slot_number: int)->dict:
	elapsed_seconds = 99999
	while elapsed_seconds>5:
		start_measuring = time.time()
		measured_data = {
			'device_name': the_setup.get_name_of_device_in_slot_number(slot_number),
			'Temperature (°C)': the_setup.measure_temperature(),
			'Humidity (%RH)': the_setup.measure_humidity(),
			'Bias voltage (V)': the_setup.measure_bias_voltage(slot_number),
			'Bias current (A)': the_setup.measure_bias_current(slot_number),
			'When': datetime.datetime.now(),
		}
		elapsed_seconds = start_measuring - time.time()
	return measured_data

PATH_TO_DIRECTORY_WHERE_TO_STORE_DATA.mkdir(exist_ok=True)

the_setup = connect_me_with_the_setup()

with SQLiteDataFrameDumper(PATH_TO_DIRECTORY_WHERE_TO_STORE_DATA/Path('measured_data.sqlite'), dump_after_n_appends=1e3, dump_after_seconds=10, delete_database_if_already_exists=False) as measured_data_dumper:
	while True:
		for slot_number in the_setup.get_slots_configuration_df().index:
			if not SILENT:
				print(f'Preparing to measure slot number {slot_number}...')
			
			if the_setup.is_bias_slot_number_being_hold_by_someone(slot_number):
				if not SILENT:
					print(f'Slot number {slot_number} is being hold by someone else, will anyway measure data under his conditions...')
				measured_data = measure_data(slot_number)
			else:
				with the_setup.hold_control_of_bias_for_slot_number(slot_number = slot_number, who = NAME_TO_ACCESS_TO_THE_SETUP):
					if not SILENT:
						print(f'Setting bias voltage to slot number {slot_number}...')
					the_setup.set_bias_voltage(slot_number=slot_number, volts=STANDBY_VOLTAGES[slot_number], who=NAME_TO_ACCESS_TO_THE_SETUP)
					time.sleep(5) # Let a bit of time to stabilize.
					measured_data = measure_data(slot_number)
			
			measured_data_df = pandas.DataFrame(
				measured_data,
				index = [0],
			)
			measured_data_df.set_index('device_name', inplace=True)
			measured_data_dumper.append(measured_data_df)
			if not SILENT:
				print(f'Slot number {slot_number} was measured.')
			time.sleep(1)
		time.sleep(60)
