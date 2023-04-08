from pathlib import Path
from the_bureaucrat.bureaucrats import RunBureaucrat # https://github.com/SengerM/the_bureaucrat

Alberto = RunBureaucrat(Path('/home/sengerm/data/TI-LGAD/FBK_RD50_TI-LGAD/beta_scans/20230407_250x250um'))
Alberto.create_run(if_exists='skip')
