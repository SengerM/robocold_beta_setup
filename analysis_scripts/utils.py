import numpy
import pandas
from scipy.stats import median_abs_deviation
from pathlib import Path
import multiprocessing

def starmap_checking_arguments_names(pool, func:callable, args:list):
	"""Same as `multiprocessing.starmap` but `args` is an iterable of 
	dictionaries instead of tuples, so it is more robust.
	
	Arguments
	---------
	pool:
		The `p` in `with multiprocessing.Pool(3) as p`.
	func: callable
		The function to evaluate, same as `multiprocessing.starmap`.
	args: list of dict
		A list of dictionaries, each dict containing the arguments for
		each evaluation of `f`. Similar to `multiprocessing.starmap` but
		instead of iterable of tuples an iterable of dicts where keys
		are `f` arguments names and items are the respective values.
	"""
	func_args_names = get_function_arguments_names(func)
	args_for_starmap = []
	for a in args:
		if set(func_args_names) != set(a):
			raise ValueError(f'Wrong arguments names, `func` arguments are {set(func_args_names)} and received {set(a)}. ')
		args_for_starmap.append(tuple([a[arg_name] for arg_name in func_args_names]))
	pool.starmap(func,args_for_starmap)

def my_std(x): 
	# This is because there is an issue with `ufloat` and `numpy.std`...
	x = numpy.array(x) 
	_x_ = numpy.mean(x) 
	return (sum((x-_x_)**2)/len(x))**.5

def save_dataframe(df, name:str, location:Path):
	for extension,method in {'pickle':df.to_pickle,'csv':df.to_csv}.items():
		method(location/f'{name}.{extension}')

def kMAD(x,nan_policy='omit'):
	"""Calculates the median absolute deviation multiplied by 1.4826... 
	which should converge to the standard deviation for Gaussian distributions,
	but is much more robust to outliers than the std."""
	k_MAD_TO_STD = 1.4826 # https://en.wikipedia.org/wiki/Median_absolute_deviation#Relation_to_standard_deviation
	return k_MAD_TO_STD*median_abs_deviation(x,nan_policy=nan_policy)

def get_function_arguments_names(f)->tuple:
	return tuple(f.__code__.co_varnames[:f.__code__.co_argcount])

def hex_to_rgba(h, alpha):
	return tuple([int(h.lstrip('#')[i:i+2], 16) for i in (0, 2, 4)] + [alpha])

def gaussian(x, mu, sigma, amplitude=1):
	return amplitude/sigma/(2*numpy.pi)**.5*numpy.exp(-((x-mu)/sigma)**2/2)

def resample_by_n_trigger(df):
	"""Produce a new sample of `df` using the value of `n_trigger`
	to group rows. Returns a new data frame of the same size."""
	was_series = False
	if isinstance(df, pandas.Series):
		df = df.to_frame()
		was_series = True
	resampled_df = df.reset_index(drop=False).pivot(
		index = 'n_trigger',
		columns = 'signal_name',
		values = list(set(df.columns)),
	)
	resampled_df = resampled_df.sample(frac=1, replace=True)
	resampled_df = resampled_df.stack()
	if was_series:
		resampled_df = resampled_df[resampled_df.columns[0]]
	return resampled_df
