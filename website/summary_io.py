"""Read benchmark CSV values without rounding small floating-point errors to zero."""
import csv
import math


def scalar(value):
    if value is None or value.strip() == '':
        return None
    if value in ('True', 'False'):
        return value == 'True'
    try:
        number = float(value)
    except ValueError:
        return value
    if not math.isfinite(number):
        return None
    if all(c not in value.lower() for c in ('.', 'e')):
        return int(value)
    return number


def read_summary(path):
    with open(path, newline='') as stream:
        return [{key: scalar(value) for key, value in row.items()}
                for row in csv.DictReader(stream)]
