import TeledyneLeCroyPy # https://github.com/SengerM/TeledyneLeCroyPy
from time import sleep
import threading
import EasySensirion # https://github.com/SengerM/EasySensirion
from CAENpy.CAENDesktopHighVoltagePowerSupply import CAENDesktopHighVoltagePowerSupply, OneCAENChannel # https://github.com/SengerM/CAENpy
from robocoldpy import Robocold, find_Robocold_port # https://github.com/SengerM/Robocold
from The_Castle_RF_multiplexer import TheCastle, find_The_Castle_port # https://github.com/SengerM/The_Castle_RF_multiplexer
from pathlib import Path
import pandas

class TheRobocoldBetaSetup:
	"""This class wraps all the hardware so if there are changes it is easy to adapt."""
	def __init__(self, path_to_configuration_file:Path):
		"""
		"""
		if not isinstance(path_to_configuration_file, Path):
			raise TypeError(f'`configuration_file` must be of type {type(Path())}, received object of type {type(path_to_configuration_file)}.')
		self.path_to_configuration_file = path_to_configuration_file
		
		self.oscilloscope = TeledyneLeCroyPy.LeCroyWaveRunner('USB0::0x05ff::0x1023::4751N40408::INSTR')
		self.sensirion = EasySensirion.SensirionSensor()
		self.caens = {
			'13398': CAENDesktopHighVoltagePowerSupply(ip='130.60.165.119'), # DT1470ET, the new one.
			'139': CAENDesktopHighVoltagePowerSupply(ip='130.60.165.121'), # DT1419ET, the old one.
		}
		self.robocold = Robocold(find_Robocold_port().device)
		self.rf_multiplexer = TheCastle(find_The_Castle_port().device)
		
		# Threading locks ---
		self._oscilloscope_Lock = threading.RLock()
		self._sensirion_Lock = threading.RLock()
		self._caen_Lock = threading.RLock()
		self._robocold_Lock = threading.RLock()
		self._rf_multiplexer_Lock = threading.RLock()
	
	@property
	def description(self) -> str:
		"""Returns a string with a "human description" of the setup, i.e.
		which instruments are connected, etc. This is useful to print in
		a file when you measure something, then later on you know which 
		instruments were you using."""
		string =  'Instruments\n'
		string += '-----------\n'
		for instrument in [self.oscilloscope, self.sensirion, self.caens['13398'], self.caens['139'], self.robocold, self.rf_multiplexer]:
			string += f'{instrument.idn}\n'
		return string
	
	@property
	def configuration_df(self):
		"""Returns a data frame with the configuration as specified in
		the file `configuration`."""
		if not hasattr(self, '_configuration_df'):
			self._configuration_df = pandas.read_csv(
				self.path_to_configuration_file,
				dtype = {
					'slot_number': int,
					'device_name': str,
					'position_long_stage': int,
					'position_short_stage': int,
					'The_Castle_channel_number': int,
					'caen_serial_number': str,
					'caen_channel_number': int,
				},
				index_col = 'slot_number',
			)
		return self._configuration_df.copy()
	
	# Bias voltage power supply ----------------------------------------
	
	def measure_bias_voltage(self, slot_number:int)->float:
		"""Returns the measured bias voltage in the given slot."""
		caen_channel = self._caen_channel_given_slot_number(slot_number)
		with self._caen_Lock:
			return caen_channel.V_mon
	
	def set_bias_voltage(self, slot_number:int, volts:float, freeze_until_not_ramping_anymore:bool=True):
		"""Set the bias voltage for the given slot.
		
		Parameters
		----------
		slot_number: int
			The number of the slot to which to set the bias voltage.
		volts: float
			The voltage to set.
		freeze_until_not_ramping_anymore: bool, default True
			If `True`, the method will hold the execution frozen until the
			CAEN says it has stopped ramping the voltage. If `False`, returns
			immediately after setting the voltage.
		"""
		if not isinstance(volts, (int, float)):
			raise TypeError(f'`volts` must be a float number, received object of type {type(volts)}.')
		if not isinstance(freeze_until_not_ramping_anymore, bool):
			raise TypeError(f'`freeze_until_not_ramping_anymore` must be boolean.')
		caen_channel = self._caen_channel_given_slot_number(slot_number)
		with self._caen_Lock:
			if freeze_until_not_ramping_anymore:
				caen_channel.ramp_voltage(voltage=volts)
			else:
				caen_channel.V_set = volts
	
	def measure_bias_current(self, slot_number:int)->float:
		"""Measures the bias current for the given slot."""
		caen_channel = self._caen_channel_given_slot_number(slot_number)
		with self._caen_Lock:
			return caen_channel.I_mon
	
	def set_current_compliance(self, slot_number:int, amperes:float):
		"""Set the current compliance for the given slot."""
		if not isinstance(amperes, (int, float)):
			raise TypeError(f'`amperes` must be a float number, received object of type {type(amperes)}.')
		caen_channel = self._caen_channel_given_slot_number(slot_number)
		with self._caen_Lock:
			caen_channel.set(
				PAR = 'ISET',
				VAL = 1e6*amperes,
			)
	
	def get_current_compliance(self, slot_number:int)->float:
		"""Returns the current compliance for the given slot number."""
		caen_channel = self._caen_channel_given_slot_number(slot_number)
		with self._caen_Lock:
			return caen_channel.get('ISET')
	
	def set_bias_voltage_status(self, slot_number:int, status:str):
		"""Turn on or off the bias voltage for the given slot.
		
		Parameters
		----------
		slot_number: int
			The number of the slot on which to operate.
		status: str
			Either `'on'` or `'off'`.
		"""
		self._check_device_name(device_name)
		if status not in {'on','off'}:
			raise ValueError(f'`status` must be either "on" or "off", received {status}.')
		caen_channel = self._caen_channel_given_slot_number(slot_number)
		with self._caen_Lock:
			caen_channel.output = status
	
	# Signal acquiring -------------------------------------------------
	
	def connect_slot_to_oscilloscope(self, slot_number:int):
		"""Commute the RF multiplexer such that the device in the `slot_number`
		gets connected to the oscilloscope."""
		with self._rf_multiplexer_Lock:
			self.rf_multiplexer.connect_channel(int(self.configuration_df.loc[slot_number,'The_Castle_channel_number']))
	
	def set_oscilloscope_vdiv(self, oscilloscope_channel_number:int, vdiv:float):
		"""Set the vertical scale of the given channel in the oscilloscope."""
		with self._oscilloscope_Lock:
			self.oscilloscope.set_vdiv(channel=channel, vdiv=vdiv)
			
	def set_oscilloscope_trigger_threshold(self, level:float):
		"""Set the threshold level of the trigger."""
		with self._oscilloscope_Lock:
			source = self.oscilloscope.get_trig_source()
			self.oscilloscope.set_trig_level(trig_source=source, level=level)
	
	def wait_for_trigger(self):
		"""Blocks execution until there is a trigger in the oscilloscope."""
		with self._oscilloscope_Lock:
			self.oscilloscope.wait_for_single_trigger()
	
	def get_waveform(self, oscilloscope_channel_number:int)->dict:
		"""Gets a waveform from the oscilloscope.
		
		Parameters
		----------
		oscilloscope_channel_number: int
			The number of channel you want to read from the oscilloscope.
		
		Returns
		-------
		waveform: dict
			A dictionary of the form `{'Time (s)': np.array, 'Amplitude (V)': np.array}`.
		"""
		self._check_device_name(device_name)
		with self._oscilloscope_Lock:
			return self.oscilloscope.get_waveform(channel=self.devices_connections[device_name]['oscilloscope channel']) 
	
	# Temperature and humidity sensor ----------------------------------
	
	@property
	def temperature(self):
		"""Returns a reading of the temperature as a float number in Celsius."""
		with self._sensirion_Lock:
			return self.sensirion.temperature
	
	@property
	def humidity(self):
		"""Returns a reading of the humidity as a float number in %RH."""
		with self._sensirion_Lock:
			return self.sensirion.humidity
	
	# Robocold ---------------------------------------------------------
	
	def reset_robocold(self):
		"""Reset the position of both stages."""
		with self._robocold_Lock:
			self.robocold.reset()
	
	def move_to_slot(self, slot_number:int):
		"""Move stages in order to align the beta source and reference
		detector trigger to the given slot."""
		slot_position = [int(self.configuration_df.loc[slot_number,p]) for p in ['position_long_stage','position_short_stage']]
		with self._robocold_Lock:
			self.robocold.move_to(tuple(slot_position))
	
	def place_source_such_that_it_does_not_irradiate_any_DUT(self):
		"""Moves the stages such that not any DUT is irradiated by the 
		source."""
		self.reset_robocold() # Not the best implementation, but should work for the time being.
	
	# Others -----------------------------------------------------------
	
	def get_device_name_in_slot(self, slot_number:int)->str:
		"""Get the name of the device in the given slot."""
		return self.configuration_df.loc[slot_number,'device_name']
	
	def _caen_channel_given_slot_number(self, slot_number:int):
		caen_serial_number = self.configuration_df.loc[slot_number,'caen_serial_number']
		caen_channel_number = int(self.configuration_df.loc[slot_number,'caen_channel_number'])
		return OneCAENChannel(self.caens[caen_serial_number], caen_channel_number)
	
if __name__ == '__main__':
	import time
	from random import shuffle
	
	the_setup = TheRobocoldBetaSetup(
		path_to_configuration_file = Path('configuration.csv')
	)
	
	print(the_setup.description)
	print(the_setup.configuration_df)
	
	SLOTS_TO_MEASURE = [1,2,3,4]
	
	while True:
		shuffle(SLOTS_TO_MEASURE)
		for slot_number in SLOTS_TO_MEASURE:
			print(f'Moving to slot {slot_number}...')
			the_setup.move_to_slot(slot_number)
			print('Connecting the slot to the oscilloscope...')
			the_setup.connect_slot_to_oscilloscope(slot_number)
			print(f'In this slot we find device {the_setup.get_device_name_in_slot(slot_number)}')
			print(f'\tMeasured bias voltage = {the_setup.measure_bias_voltage(slot_number)} V')
			print(f'\tMeasured bias current = {the_setup.measure_bias_current(slot_number)} A')
			time.sleep(10)
