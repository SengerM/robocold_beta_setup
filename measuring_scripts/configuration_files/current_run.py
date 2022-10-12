from pathlib import Path
from the_bureaucrat.bureaucrats import RunBureaucrat # https://github.com/SengerM/the_bureaucrat

Alberto = RunBureaucrat(Path.home()/Path('measurements_data/20221006_IrradiatedTI_Load2'))
Alberto.create_run()
