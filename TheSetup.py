import TeledyneLeCroyPy # https://github.com/SengerM/TeledyneLeCroyPy
from time import sleep
import EasySensirion # https://github.com/SengerM/EasySensirion
from CAENpy.CAENDesktopHighVoltagePowerSupply import CAENDesktopHighVoltagePowerSupply, OneCAENChannel # https://github.com/SengerM/CAENpy
from robocoldpy import Robocold, find_Robocold_port # https://github.com/SengerM/Robocold
from The_Castle_RF_multiplexer import TheCastle, find_The_Castle_port # https://github.com/SengerM/The_Castle_RF_multiplexer
from pathlib import Path
import pandas
from VotschTechnikClimateChamber.ClimateChamber import ClimateChamber # https://github.com/SengerM/VotschTechnik-climate-chamber-Python
from CrossProcessLock import CrossProcessNamedLock
from threading import RLock
from multiprocessing.managers import BaseManager

class TheRobocoldBetaSetup:
	"""This class wraps all the hardware so if there are changes it is 
	easy to adapt. It should be thread safe.
	"""
	def __init__(self, path_to_slots_configuration_file:Path=None, path_to_oscilloscope_configuration_file:Path=None):
		if path_to_slots_configuration_file is None:
			path_to_slots_configuration_file = Path('slots_configuration.csv')
		if path_to_oscilloscope_configuration_file is None:
			path_to_oscilloscope_configuration_file = Path('oscilloscope_configuration.csv')
		for name in {'path_to_oscilloscope_configuration_file','path_to_slots_configuration_file'}:
			if not isinstance(locals()[name], Path):
				raise TypeError(f'`{name}` must be of type {type(Path())}, received object of type {type(locals()[name])}.')
		self.path_to_slots_configuration_file = path_to_slots_configuration_file
		self.path_to_oscilloscope_configuration_file = path_to_oscilloscope_configuration_file
		
		self.slots_configuration_df # This will trigger the load of the file, so if it fails it does now.
		self.oscilloscope_configuration_df # This will trigger the load of the file, so if it fails it does now.
		
		# Hardware elements ---
		self._oscilloscope = TeledyneLeCroyPy.LeCroyWaveRunner('USB0::0x05ff::0x1023::4751N40408::INSTR')
		self._sensirion = EasySensirion.SensirionSensor()
		self._caens = {
			'13398': CAENDesktopHighVoltagePowerSupply(ip='130.60.165.119'), # DT1470ET, the new one.
			'139': CAENDesktopHighVoltagePowerSupply(ip='130.60.165.121'), # DT1419ET, the old one.
		}
		self._robocold = Robocold(find_Robocold_port().device)
		self._rf_multiplexer = TheCastle(find_The_Castle_port().device)
		self._climate_chamber = ClimateChamber(ip = '130.60.165.218' , temperature_min = -25, temperature_max = 30)
		
		# Locks for hardware ---
		# These locks ensure that each hardware is accessed only once at
		# a time, but the user does not know anything about them.
		self._sensirion_Lock = RLock()
		self._caen_Lock = RLock()
		self._rf_multiplexer_Lock = RLock()
		self._robocold_Lock = RLock()
		self._climate_chamber_Lock = RLock()
		self._oscilloscope_Lock = RLock()
		# Locks for the user to hold ---
		# These locks are so the user can hold the control of a part of
		# the setup for an extended period of time. I had to write my own
		# lock because the ones existing in Python are not compatible
		# with multiple processes.
		self._bias_for_slot_Lock = {slot_number: CrossProcessNamedLock(Path.home()) for slot_number in self.slots_configuration_df.index}
		self._signal_acquisition_Lock = CrossProcessNamedLock(Path.home())
		self._hold_robocold_Lock = CrossProcessNamedLock(Path.home())
	
	@property
	def description(self) -> str:
		"""Returns a string with a "human description" of the setup, i.e.
		which instruments are connected, etc. This is useful to print in
		a file when you measure something, then later on you know which 
		instruments were you using."""
		instruments = [
			self._oscilloscope, 
			self._sensirion, 
			self._caens['13398'], 
			self._caens['139'], 
			self._robocold, 
			self._rf_multiplexer, 
			self._climate_chamber,
		]
		string =  'Instruments\n'
		string += '-----------\n\n'
		for instrument in instruments:
			string += f'{instrument.idn}\n'
		string += '\nSlots configuration\n'
		string += '-------------------\n\n'
		string += self.slots_configuration_df.to_string(max_rows=999,max_cols=999)
		return string
	
	@property
	def slots_configuration_df(self):
		"""Returns a data frame with the configuration as specified in
		the slots configuration file."""
		if not hasattr(self, '_slots_configuration_df'):
			self._slots_configuration_df = pandas.read_csv(
				self.path_to_slots_configuration_file,
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
		return self._slots_configuration_df.copy()
	
	@property
	def oscilloscope_configuration_df(self):
		"""Returns a data frame with the configuration specified in the
		oscilloscope configuration file."""
		if not hasattr(self, '_oscilloscope_configuration_df'):
			self._oscilloscope_configuration_df = pandas.read_csv(
				self.path_to_oscilloscope_configuration_file,
				dtype = {
					'n_channel': int,
					'signal_name': str,
				},
				index_col = 'signal_name',
			)
		return self._oscilloscope_configuration_df.copy()
	
	# Bias voltage power supply ----------------------------------------
	
	def hold_control_of_bias_for_slot_number(self, slot_number:int, who:str):
		"""When this is called in a `with` statement, it will guarantee
		the exclusive control of the bias conditions for the slot. Note 
		that others will be able to measure, but not change the voltage/current.
		
		Parameters
		----------
		slot_number: int
			Number of the slot for which you want to hold the control of
			the bias.
		who: str
			A string identifying you. This can be whatever you want, but
			you have to use always the same. A good choice is `str(os.getpid())`
			because it will give all your imported modules the same name.
			This is a workaround because, surprisingly,  the Locks in python
			are not multiprocess friendly.
		
		Example
		-------
		```
		with the_setup.hold_control_of_bias_for_slot_number(slot_number, my_name):
			# Nobody else from other thread can change the bias conditions for this slot.
			the_setup.set_bias_voltage(slot_number, volts, my_name) # This will not change unless you change it here.
		```
		"""
		return self._bias_for_slot_Lock[slot_number](who)
	
	def measure_bias_voltage(self, slot_number:int)->float:
		"""Returns the measured bias voltage in the given slot.
		
		Parameters
		----------
		slot_number: int
			Number of the slot for which you want to hold the control of
			the bias.
		"""
		caen_channel = self._caen_channel_given_slot_number(slot_number)
		with self._caen_Lock:
			return caen_channel.V_mon
	
	def set_bias_voltage(self, slot_number:int, volts:float, who:str, block_until_not_ramping_anymore:bool=True):
		"""Set the bias voltage for the given slot.
		
		Parameters
		----------
		slot_number: int
			The number of the slot to which to set the bias voltage.
		volts: float
			The voltage to set.
		who: str
			A string identifying you. This can be whatever you want, but
			you have to use always the same. A good choice is `str(os.getpid())`
			because it will give all your imported modules the same name.
			This is a workaround because, surprisingly,  the Locks in python
			are not multiprocess friendly.
		freeze_until_not_ramping_anymore: bool, default True
			If `True`, the method will hold the execution frozen until the
			CAEN says it has stopped ramping the voltage. If `False`, returns
			immediately after setting the voltage. This function is "thread
			friendly" in the sense that it will not block the whole access
			to the CAEN power supplies while it waits for the ramping to
			stop. Yet it is thread safe.
		"""
		if not isinstance(volts, (int, float)):
			raise TypeError(f'`volts` must be a float number, received object of type {type(volts)}.')
		if not isinstance(block_until_not_ramping_anymore, bool):
			raise TypeError(f'`block_until_not_ramping_anymore` must be boolean.')
		with self._bias_for_slot_Lock[slot_number](who):
			caen_channel = self._caen_channel_given_slot_number(slot_number)
			with self._caen_Lock:
				caen_channel.V_set = volts
			if block_until_not_ramping_anymore:
				sleep(1) # It takes a while for the CAEN to realize that it has to change the voltage...
				while True:
					if self.is_ramping_bias_voltage(slot_number) == False:
						break
					sleep(1)
				sleep(3) # Empirically, after CAEN says it is not ramping anymore, you have to wait 3 seconds to be sure it actually stopped ramping...
	
	def is_ramping_bias_voltage(self, slot_number:int)->bool:
		caen_channel = self._caen_channel_given_slot_number(slot_number)
		with self._caen_Lock:
			return caen_channel.is_ramping
	
	def measure_bias_current(self, slot_number:int)->float:
		"""Measures the bias current for the given slot."""
		caen_channel = self._caen_channel_given_slot_number(slot_number)
		with self._caen_Lock:
			return caen_channel.I_mon
	
	def set_current_compliance(self, slot_number:int, amperes:float, who:str):
		"""Set the current compliance for the given slot."""
		if not isinstance(amperes, (int, float)):
			raise TypeError(f'`amperes` must be a float number, received object of type {type(amperes)}.')
		caen_channel = self._caen_channel_given_slot_number(slot_number)
		with self._bias_for_slot_Lock[slot_number](who), self._caen_Lock:
			caen_channel.set(
				PAR = 'ISET',
				VAL = 1e6*amperes,
			)
	
	def get_current_compliance(self, slot_number:int)->float:
		"""Returns the current compliance for the given slot number."""
		caen_channel = self._caen_channel_given_slot_number(slot_number)
		with self._caen_Lock:
			return caen_channel.get('ISET')
	
	def set_bias_voltage_status(self, slot_number:int, status:str, who:str):
		"""Turn on or off the bias voltage for the given slot.
		
		Parameters
		----------
		slot_number: int
			The number of the slot on which to operate.
		status: str
			Either `'on'` or `'off'`.
		who: str
			A string identifying you. This can be whatever you want, but
			you have to use always the same. A good choice is `str(os.getpid())`
			because it will give all your imported modules the same name.
			This is a workaround because, surprisingly,  the Locks in python
			are not multiprocess friendly.
		"""
		if status not in {'on','off'}:
			raise ValueError(f'`status` must be either "on" or "off", received {status}.')
		caen_channel = self._caen_channel_given_slot_number(slot_number)
		with self._bias_for_slot_Lock[slot_number](who), self._caen_Lock:
			caen_channel.output = status
	
	# Signal acquiring -------------------------------------------------
	
	def hold_signal_acquisition(self, who:str):
		"""When this is called in a `with` statement, it will guarantee
		the exclusive control of the signal acquisition system, i.e. the
		oscilloscope and the RF multiplexer (The Castle).
		
		who: str
			A string identifying you. This can be whatever you want, but
			you have to use always the same. A good choice is `str(os.getpid())`
			because it will give all your imported modules the same name.
			This is a workaround because, surprisingly,  the Locks in python
			are not multiprocess friendly.
		
		Example
		-------
		```
		with the_setup.hold_signal_acquisition(my_name):
			# Nobody else from other thread can change anything from the oscilloscope or The Castle.
		```
		"""
		return self._signal_acquisition_Lock(who)
	
	def connect_slot_to_oscilloscope(self, slot_number:int, who:str):
		"""Commute the RF multiplexer such that the device in the `slot_number`
		gets connected to the oscilloscope."""
		with self._signal_acquisition_Lock(who), self._oscilloscope_Lock:
			self._rf_multiplexer.connect_channel(int(self.slots_configuration_df.loc[slot_number,'The_Castle_channel_number']))
	
	def set_oscilloscope_vdiv(self, oscilloscope_channel_number:int, vdiv:float, who:str):
		"""Set the vertical scale of the given channel in the oscilloscope."""
		with self._signal_acquisition_Lock(who), self._oscilloscope_Lock:
			self._oscilloscope.set_vdiv(channel=channel, vdiv=vdiv)
			
	def set_oscilloscope_trigger_threshold(self, level:float, who:str):
		"""Set the threshold level of the trigger."""
		with self._signal_acquisition_Lock(who), self._oscilloscope_Lock:
			source = self._oscilloscope.get_trig_source()
			self._oscilloscope.set_trig_level(trig_source=source, level=level)
	
	def wait_for_trigger(self, who:str):
		"""Blocks execution until there is a trigger in the oscilloscope."""
		with self._signal_acquisition_Lock(who), self._oscilloscope_Lock:
			self._oscilloscope.wait_for_single_trigger()
	
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
		with self._signal_acquisition_Lock:
			return self._oscilloscope.get_waveform(channel=oscilloscope_channel_number) 
	
	# Temperature and humidity sensor ----------------------------------
	
	def measure_temperature(self):
		"""Returns a reading of the temperature as a float number in Celsius."""
		with self._sensirion_Lock:
			return self._sensirion.temperature
	
	def measure_humidity(self):
		"""Returns a reading of the humidity as a float number in %RH."""
		with self._sensirion_Lock:
			return self._sensirion.humidity
	
	# Robocold ---------------------------------------------------------
	
	def hold_control_of_robocold(self, who:str):
		"""When this is called in a `with` statement, it will guarantee
		the exclusive control of Robocold while within the statement.
		
		who: str
			A string identifying you. This can be whatever you want, but
			you have to use always the same. A good choice is `str(os.getpid())`
			because it will give all your imported modules the same name.
			This is a workaround because, surprisingly,  the Locks in python
			are not multiprocess friendly.
		
		Example
		-------
		```
		with the_setup.hold_control_of_robocold(my_name):
			the_setup.move_to_slot(blah, my_name)
			# Nobody else from other thread can control Robocold now, it
			will stay in slot `blah` and do whatever you want.
		```
		"""
		return self._hold_robocold_Lock(who)
	
	def reset_robocold(self, who:str):
		"""Reset the position of both stages."""
		with self._hold_robocold_Lock(who), self._robocold_Lock:
			self._robocold.reset()
	
	def move_to_slot(self, slot_number:int, who:str):
		"""Move stages in order to align the beta source and reference
		detector trigger to the given slot."""
		slot_position = [int(self.slots_configuration_df.loc[slot_number,p]) for p in ['position_long_stage','position_short_stage']]
		with self._hold_robocold_Lock(who), self._robocold_Lock:
			self._robocold.move_to(tuple(slot_position))
	
	def place_source_such_that_it_does_not_irradiate_any_DUT(self, who:str):
		"""Moves the stages such that not any DUT is irradiated by the 
		source."""
		with self._hold_robocold_Lock(who), self._robocold_Lock:
			self.reset_robocold(who) # Not the best implementation, but should work for the time being.
	
	def get_robocold_position(self):
		"""Return the position of Robocold as a tuple of two int."""
		with self._robocold_Lock:
			return self._robocold.position
	
	def set_robocold_position(self, position:tuple, who:str):
		"""Set the position to Robocold. The execution of the program is
		halted until Robocold stops moving."""
		if not isinstance(position, tuple) or len(position)!=2 or any([not isinstance(p, int) for p in position]):
			raise TypeError('`position` must be a `tuple` with two `int`.')
		with self._hold_robocold_Lock(who), self._robocold_Lock:
			self._robocold.move_to(position)
	
	# Climate chamber --------------------------------------------------
	
	def set_temperature(self, celsius:float):
		"""Set the temperature."""
		with self._climate_chamber_Lock:
			self._climate_chamber.temperature_set_point = celsius
	
	def start_climate_chamber(self):
		"""Start the climate chamber."""
		with self._climate_chamber_Lock:
			self._climate_chamber.dryer = True
			self._climate_chamber.start()
	
	def stop_climate_chamber(self):
		"""Stop the climate chamber."""
		with self._climate_chamber_Lock:
			self._climate_chamber.stop()
	
	# Others -----------------------------------------------------------
	
	def get_name_of_device_in_slot_number(self, slot_number:int)->str:
		"""Get the name of the device in the given slot."""
		return self.slots_configuration_df.loc[slot_number,'device_name']
	
	def _caen_channel_given_slot_number(self, slot_number:int):
		caen_serial_number = self.slots_configuration_df.loc[slot_number,'caen_serial_number']
		caen_channel_number = int(self.slots_configuration_df.loc[slot_number,'caen_channel_number'])
		return OneCAENChannel(self._caens[caen_serial_number], caen_channel_number)
	
	def turn_whole_setup_off_in_controlled_way(self, who:str):
		"""Turns the whole setup off in a controlled and safe way. This
		means e.g. setting the bias voltages to 0 before warming up the
		climate chamber. This function will block the execution until
		it has finished, which may take about 20 minutes."""
		with self._caen_Lock, self._climate_chamber_Lock:
			for slot_number in self.slots_configuration_df.index:
				self.set_bias_voltage(slot_number, 0, block_until_not_ramping_anymore=False, who=who)
			self.reset_robocold(who=who)
			while any([self.is_ramping_bias_voltage(slot_number) for slot_number in self.slots_configuration_df.index]):
				sleep(10)
			# Warming up sequence ---
			current_set_T = self.temperature
			while self.temperature < 20:
				current_set_T += 5
				self.set_temperature(current_set_T)
				while self.temperature<current_set_T-1 or self.humidity > 10:
					sleep(10)
			self.set_temperature(20)
			sleep(60) # An extra minute.
			self.stop_climate_chamber()
	
	def start_setup(self, who):
		"""Starts the setup in a controlled and safe way. Blocks the 
		execution until the setup is ready to operate at low temperature."""
		with self._caen_Lock, self._climate_chamber_Lock:
			for slot_number in self.slots_configuration_df.index:
				self.set_bias_voltage(slot_number, 0, block_until_not_ramping_anymore=False, who=who)
			self.reset_robocold(who=who)
			while any([self.is_ramping_bias_voltage(slot_number) for slot_number in self.slots_configuration_df.index]):
				sleep(10)
			self.set_temperature(-20)
			self.start_climate_chamber()
			while self.temperature > -20:
				sleep(10)
	
	def get_description(self)->str:
		"""Same as `.description` but without the property decorator,
		because the properties fail in multiprocess applications."""
		return self.description
	
def give_me_the_setup():
	class TheSetup(BaseManager):
		pass

	TheSetup.register('get_the_setup')
	m = TheSetup(address=('', 50000), authkey=b'abracadabra')
	m.connect()
	the_setup = m.get_the_setup()
	return the_setup
	
if __name__=='__main__':
	class TheSetupManager(BaseManager):
		pass
	
	print('Opening the setup...')
	the_setup = TheRobocoldBetaSetup(
		path_to_slots_configuration_file = Path('slots_configuration.csv'),
		path_to_oscilloscope_configuration_file = Path('oscilloscope_configuration.csv'),
	)
	
	TheSetupManager.register('get_the_setup', callable=lambda:the_setup)
	m = TheSetupManager(address=('', 50000), authkey=b'abracadabra')
	s = m.get_server()
	print('Ready!')
	s.serve_forever()

 
