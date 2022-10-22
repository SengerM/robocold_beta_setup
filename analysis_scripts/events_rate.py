import pandas
from the_bureaucrat.bureaucrats import RunBureaucrat # https://github.com/SengerM/the_bureaucrat
from pathlib import Path
import numpy
from huge_dataframe.SQLiteDataFrame import load_whole_dataframe # https://github.com/SengerM/huge_dataframe
from clean_beta_scan import tag_n_trigger_as_background_according_to_the_result_of_clean_beta_scan
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import expon
from scipy.optimize import curve_fit
from uncertainties import ufloat

def non_binned_fit(samples:numpy.array, cdf_model, guess:list, nan_policy='drop', maxfev=0):
	if nan_policy == 'drop':
		samples = samples[~numpy.isnan(samples)]
	else:
		raise NotImplementedError(f'`nan_policy={nan_policy}` not implemented.')
	if len(samples) == 0:
		raise ValueError(f'`samples` is an empty array.')
	samples_sorted = numpy.sort(samples)
	samples_ecdf = numpy.arange(1, len(samples_sorted)+1)/float(len(samples_sorted))
	popt, pcov = curve_fit(
		f = cdf_model,
		xdata = samples_sorted,
		ydata = samples_ecdf,
		p0 = guess,
	)
	return popt, pcov

def fit_exponential_to_time_differences(time_differences:numpy.array, measurement_seconds:float):
	fits_params = []
	popt, pcov = non_binned_fit(
		samples = time_differences,
		cdf_model = lambda x, loc, scale: expon.cdf(x,loc=loc,scale=scale),
		guess = [0, measurement_seconds/len(time_differences)],
	)
	return {
		'Rate (events s^-1)': popt[1]**-1,
		'Offset (s)': popt[0],
	}

def events_rate(bureaucrat:RunBureaucrat, n_bootstraps:int=99):
	with bureaucrat.handle_task('events_rate') as employee:
		bureaucrat.check_these_tasks_were_run_successfully(['beta_scan'])
		data = load_whole_dataframe(bureaucrat.path_to_directory_of_task('beta_scan')/'measured_stuff.sqlite')
		data = data[['When']]
		data['When'] = pandas.to_datetime(data['When'])
		if bureaucrat.was_task_run_successfully('clean_beta_scan'):
			data = tag_n_trigger_as_background_according_to_the_result_of_clean_beta_scan(bureaucrat, data)
		else:
			data['is_background'] = False
		
		time_differences = []
		for fitting_to in ['all','signal','background']:
			if fitting_to == 'all':
				samples = data
			elif fitting_to == 'signal':
				samples = data.query('is_background==False')
			elif fitting_to == 'background':
				samples = data.query('is_background==True')
			else:
				raise ValueError()
			
			_ = samples['When'].diff().apply(lambda x: x.total_seconds()).to_frame()
			_.rename(columns={'When': 'Δt (s)'}, inplace=True)
			_['fitting_to'] = fitting_to
			time_differences.append(_)
		time_differences = pandas.concat(time_differences).dropna()
		time_differences.set_index('fitting_to',inplace=True,drop=True)
		
		measurement_total_seconds = (data['When'].max() - data['When'].min()).total_seconds()
		
		fitted_parameters = []
		for n_bootstrap in range(n_bootstraps+1):
			if n_bootstrap == 0:
				samples = time_differences.copy()
			else:
				samples = time_differences.sample(frac=1, replace=True)
			for fitting_to in set(samples.index.get_level_values('fitting_to')):
				_ = fit_exponential_to_time_differences(samples.query(f'fitting_to=={repr(fitting_to)}').to_numpy(), measurement_seconds = measurement_total_seconds)
				_['n_bootstrap'] = n_bootstrap
				_['fitting_to'] = fitting_to
				_ = pandas.DataFrame(_, index=[0])
				_.set_index(['n_bootstrap','fitting_to'], inplace=True)
				fitted_parameters.append(_)
		fitted_parameters = pandas.concat(fitted_parameters)
		
		results = fitted_parameters.groupby('fitting_to').agg([numpy.mean,numpy.std])
		results.columns = [' '.join(col).replace('mean','').replace('std','error').rstrip(' ') for col in results.columns]
		
		results.to_pickle(employee.path_to_directory_of_my_task/'rates.pickle')
		
		fig = px.ecdf(
			title = f'Events rates estimation<br><sup>Run: {bureaucrat.run_name}</sup>',
			data_frame = time_differences.reset_index(drop=False),
			x = 'Δt (s)',
			color = 'fitting_to',
			labels = {'fitting_to': 'type'},
		)
		for fitting_to in results.index.get_level_values('fitting_to'):
			x_axis_values = numpy.logspace(numpy.log10(min(time_differences.query(f'fitting_to=={repr(fitting_to)}')['Δt (s)'])),numpy.log10(max(time_differences.query(f'fitting_to=={repr(fitting_to)}')['Δt (s)'])))
			fig.add_trace(
				go.Scatter(
					x = x_axis_values,
					y = expon.cdf(x_axis_values, scale = results.loc[fitting_to,'Rate (events s^-1)']**-1, loc = results.loc[fitting_to,'Offset (s)']),
					mode = 'lines',
					name = f"{fitting_to}: λ={ufloat(results.loc[fitting_to,'Rate (events s^-1)'],results.loc[fitting_to,'Rate (events s^-1) error']):.2e} evnts/s".replace('+/-','±'),
					line = dict(
						dash = 'dash',
						color = 'black',
					),
				)
			)
		fig.write_html(
			employee.path_to_directory_of_my_task/'events_rates.html',
			include_plotlyjs = 'cdn',
		)
		
if __name__ == '__main__':
	import argparse

	parser = argparse.ArgumentParser()
	parser.add_argument(
		'--dir',
		metavar = 'path',
		help = 'Path to the base directory of a measurement.',
		required = True,
		dest = 'directory',
		type = str,
	)
	parser.add_argument(
		'--force',
		help = 'If this flag is passed, it will force the calculation even if it was already done beforehand. Old data will be deleted.',
		required = False,
		dest = 'force',
		action = 'store_true'
	)
	args = parser.parse_args()
	events_rate(RunBureaucrat(Path(args.directory)))
