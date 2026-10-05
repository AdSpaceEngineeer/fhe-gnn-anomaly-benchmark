"""Validate submitter-reported compute parallelism."""


def validate_thread_report(value):
    if not isinstance(value, dict):
        raise ValueError('thread_report must return a dictionary')
    for name in ('compute_threads', 'worker_processes'):
        if type(value.get(name)) is not int or value[name] < 1:
            raise ValueError('thread_report requires a positive integer ' + name)
    if not isinstance(value.get('threading_model'), str) or not value['threading_model'].strip():
        raise ValueError('thread_report requires a threading_model description')
    return value
