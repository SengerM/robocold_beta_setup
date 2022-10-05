from TheSetup import connect_me_with_the_setup
import time

SLOT_NUMBER = 2
BIAS_VOLTAGE = 180
CURRENT_COMPLIANCE = 10e-6
ROBOCOLD_OFFSET = (0,0)

the_setup = connect_me_with_the_setup()

print('Waiting for hardware...')
with the_setup.hold_control_of_bias_for_slot_number(SLOT_NUMBER,'me'), the_setup.hold_signal_acquisition('me'), the_setup.hold_control_of_robocold('me'):
	print('Hardware acquired!')
	print('Connecting to oscilloscope...')
	the_setup.connect_slot_to_oscilloscope(SLOT_NUMBER, 'me')
	print('Moving Robocold...')
	# ~ the_setup.reset_robocold('me')
	the_setup.move_to_slot(SLOT_NUMBER, 'me')
	the_setup.set_robocold_position(tuple([x+offset for x,offset in zip(the_setup.get_robocold_position(),ROBOCOLD_OFFSET)]), who='me')
	print('Setting bias voltage...')
	the_setup.set_current_compliance(SLOT_NUMBER, amperes=CURRENT_COMPLIANCE, who='me')
	the_setup.set_bias_voltage(SLOT_NUMBER,BIAS_VOLTAGE,'me')
	while True:
		print(f'Slot {SLOT_NUMBER}: {the_setup.get_name_of_device_in_slot_number(SLOT_NUMBER)} | {the_setup.measure_bias_voltage(slot_number=SLOT_NUMBER):.2f} V | {the_setup.measure_bias_current(slot_number=SLOT_NUMBER)*1e6:.2f} µA')
		time.sleep(1)
