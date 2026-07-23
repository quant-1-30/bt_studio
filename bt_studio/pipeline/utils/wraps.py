from functools import wraps

def consume_time(func):

    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print('{} cost {}s'.format(func.__name__, round(end_time - start_time, 3)))
        return result
    return wrapper
