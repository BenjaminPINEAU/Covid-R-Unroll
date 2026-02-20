import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import ScalarFormatter

"""
The following two functions are extracted from 
github.com/juliana-du/Covid-R-estim/blob/main/display/formattingFigures.py
"""

def adaptiveYLimit(Ydata):
    """
    If positive data, returns upper bound for set_ylim function in matplotlib.
    :param Ydata: ndarray of shape (days, )
    :return:
   """
    floorPowerTen = np.floor(np.log(np.max(Ydata)) / np.log(10))
    return np.ceil(np.max(Ydata) / (10 ** floorPowerTen)) * 10 ** floorPowerTen


def adaptiveDaysLocator(formattedDates):
    """
    Return an adaptive locator for the given dates.
    :param formattedDates : mdates.num array of shape (days,)
    :return:
    """
    days = len(formattedDates)
    firstDay = mdates.num2date(formattedDates[0]).day
    if firstDay in [29, 30, 31]:
        firstDay = 1
    if days < 35:
        interval = int(np.round(days / 5))
        return mdates.DayLocator(interval=interval), mdates.DateFormatter('%d~%b~%Y')
    elif 34 < days < 111:
        interval = int(np.round(days / 35))
        firstWeek = int(np.ceil(firstDay / 7))
        return mdates.WeekdayLocator(interval=interval, byweekday=firstWeek), mdates.DateFormatter('%d~%b~%Y')
    elif 110 < days < 550:
        interval = int(np.round(days / 140))
        return mdates.MonthLocator(interval=interval, bymonthday=firstDay), mdates.DateFormatter('%b~%Y')
    else:
        return mdates.YearLocator(), mdates.DateFormatter('%Y')
    
class ScalarFormatterClass(ScalarFormatter):
    def _set_format(self):
        self.format = "%1.1f"
