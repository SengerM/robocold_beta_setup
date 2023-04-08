from TheSetup import connect_me_with_the_setup
import os
import my_telegram_bots
from progressreporting.TelegramProgressReporter import SafeTelegramReporter4Loops # https://github.com/SengerM/progressreporting

onoff = input(f'Switch setup on or off? ')
if onoff not in {'on','off'}:
	raise ValueError(f'Enter either "on" or "off", you entered {repr(onoff)}.')

s = connect_me_with_the_setup()

reporter = SafeTelegramReporter4Loops(
	bot_token = my_telegram_bots.robobot.token,
	chat_id = my_telegram_bots.chat_ids['Robobot beta setup'],
)
try:
	if onoff == 'off':
		print('Turning the setup off...')
		reporter.send_message('Turning the setup off...')
		s.turn_whole_setup_off_in_controlled_way(who=f'turning onoff setup {os.getpid()}')
		print('The setup is now off.')
		reporter.send_message('Setup is off, can be opened now. ✅')
	elif onoff == 'on':
		print(f'Turning the setup on...')
		reporter.send_message('Turning the setup on...')
		s.start_setup(who=f'turning onoff setup {os.getpid()}')
		print(f'Setup is ready to operate!')
		reporter.send_message('Setup is on and ready to start operating. ✅')
	else:
		raise ValueError(f'Enter either "on" or "off", you entered {repr(onoff)}.')
except Exception as e:
	reporter.send_message(f'Something went wrong when turning the setup {onoff}... The issue: {repr(e)}')
	raise e
