from TheSetup import connect_me_with_the_setup
import os

onoff = input(f'Switch setup on or off? ')

s = connect_me_with_the_setup()

if onoff == 'off':
	print('Turning the setup off...')
	s.turn_whole_setup_off_in_controlled_way(who=f'turning onoff setup {os.getpid()}')
	print('The setup is now off.')
elif onoff == 'on':
	print(f'Turning the setup on...')
	s.start_setup(who=f'turning onoff setup {os.getpid()}')
	print(f'Setup is ready to operate!')
else:
	raise ValueError(f'Enter either "on" or "off", you entered {repr(onoff)}.')
