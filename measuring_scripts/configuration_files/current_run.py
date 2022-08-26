from pathlib import Path
from the_bureaucrat.bureaucrats import RunBureaucrat # https://github.com/SengerM/the_bureaucrat

Alberto = RunBureaucrat(Path.home()/Path('measurements_data/20220826000000_Robocold_setup_test_run_2'))
