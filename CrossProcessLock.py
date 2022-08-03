from pathlib import Path
import time
import atexit
import datetime

SEPARATION_STRING = '<-->' # This just has to be a very weird string that will never appear in the name of whoever wants to acquire the lock.

def _check_name(name:str):
	if not isinstance(name, str):
		raise TypeError(f'`name` must be a string.')
	if name == SEPARATION_STRING:
		raise ValueError(f'`name` cannot be "{SEPARATION_STRING}", please use another.')

class CrossProcessNamedLock:
	"""This class implements a named lock, i.e. you can lock it or not
	depending on your name. The lock is process safe and thread safe.
	"""
	def __init__(self, path_to_a_directory_with_writing_permission:Path):
		self._path_to_this_lock_file = path_to_a_directory_with_writing_permission/Path(f'NamedLock_{datetime.datetime.now().strftime("%Y%m%d%H%M%S%f")}.txt')
		time.sleep(1e-3) # This is to be sure that the timestamp in the previous line does not repeat.
		def release_lock_at_exit():
			if self._path_to_this_lock_file.is_file():
				self._path_to_this_lock_file.unlink()
		atexit.register(release_lock_at_exit)
	
	def locked(self):
		"""Return `True` if the lock is acquired."""
		return self._path_to_this_lock_file.is_file()
	
	def locked_by(self, name:str):
		"""Return `True` if the lock is acquired by `name`."""
		_check_name(name)
		if self.locked():
			return self._acquired_by()==name
		return False
	
	def _acquired_by(self):
		"""Returns the name of the whoever acquires the lock, or `None` if
		it is not acquired by anyone."""
		if self.locked():
			with open(self._path_to_this_lock_file, 'r') as f:
				return f.read().split(SEPARATION_STRING)[0]
		else:
			return None
	
	def acquire(self, name:str):
		"""Acquire the lock, or wait indefinitely until it can be acquired."""
		_check_name(name)
		if self.locked_by(name):
			with open(self._path_to_this_lock_file, 'a') as f:
				print(name, file=f, end=SEPARATION_STRING)
		else:
			while self.locked():
				time.sleep(.1)
			with open(self._path_to_this_lock_file, 'w') as f:
				print(name, file=f, end=SEPARATION_STRING)
	
	def release(self, name:str):
		"""Release the lock (only one recursive acquisition level)."""
		_check_name(name)
		if self.locked_by(name):
			with open(self._path_to_this_lock_file, 'r') as f:
				locks = f.read().split(SEPARATION_STRING)
			locks = locks[:-2]
			if len(locks) == 0: # Release the lock.
				self._path_to_this_lock_file.unlink()
			else:
				with open(self._path_to_this_lock_file, 'w') as f:
					print(SEPARATION_STRING.join(locks), file=f, end=SEPARATION_STRING)
	
	def __call__(self, name:str):
		_check_name(name)
		self._wants_to_acquire = name
		return self
	
	def __enter__(self):
		if not hasattr(self, '_wants_to_acquire'):
			raise RuntimeError(f'To use with a `with` statement you have to call the `NamedLock` object with a name, e.g.: `with my_named_lock("a name"):`.')
		self.acquire(f'{self._wants_to_acquire}')
		if hasattr(self, '_wants_to_acquire'): # We have to check because in the meantime someone else could have change this.
			delattr(self, '_wants_to_acquire')
		return self
	
	def __exit__(self, exc_type, exc_value, traceback):
		self.release(self._acquired_by())
