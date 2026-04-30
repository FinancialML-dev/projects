#Create Dollar bars
import pandas 
import math 
from datetime import datetime, timedelta

###########
#Collect historical candlestick data
#Inspired from https://davidzhao12.medium.com/advances-in-financial-machine-learning-for-dummies-part-1-7913aa7226f5

#We fetch our data from Alpaca
#https://alpaca.markets/sdks/python/getting_started.html#introduction

from alpaca.data.historical import CryptoHistoricalDataClient    #uv add alpaca-py
from alpaca.data.requests import CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame

#no keys required for crypto data
client = CryptoHistoricalDataClient()

request_params = CryptoBarsRequest(
                        #symbol_or_symbols=["BTC/USD", "ETH/USD"],
                        #timeframe=TimeFrame.Day, #Day gives you a faster loop/completes faster
                        symbol_or_symbols=["BTC/USD"],
                        #timeframe=TimeFrame.Minute,    #Gives volumes of 0 
                        #timeframe=TimeFrame.Hour,      #Gives still volumes of 0
                        #start="2026-04-01",
                        #start=datetime(2026, 4, 1),    #Start date year,month, day --> of minute data
                        #end=datetime(2026, 4, 2)       #End date year, month, day --> of minute data
                        timeframe=TimeFrame.Day,        #Gives better volumes data
                        #start="2026-04-01",
                        start=datetime(2026, 4, 1),     #Start date year,month, day --> of days data
                        end=datetime(2026, 4, 27)       #End date year, month, day --> of days data
                 )

#bars = client.get_crypto_bars(request_params)
candleBars = client.get_crypto_bars(request_params)

#print(candleBars) #print the data so we can see how it is stuctured. Often it is a Json object. Look for list `[...]` and objects `{...}` keys and values.
#print(candleBars.data["BTC/USD"][0].open)

time = []
open = []
high = []
low = []
close = []
volume = []

#Formatting the data to match the plotting library
#for item in candleBars:
    #print(item)
    #print(item[1]["BTC/USD"][0].timestamp)
    #time.append(item[1]["BTC/USD"][0].timestamp)
for item in candleBars.data["BTC/USD"]: #Version 1
    #print(item)                #Print the data, inspect how is it stuctured?
    #print(item.timestamp)      #test print the data to get a value from a key (test and see if you get what you want)
    time.append(item.timestamp)
    #time.append(item["timestamp"])
    open.append(item.open)
    high.append(item.high)
    low.append(item.low)
    close.append(item.close)
    volume.append(item.volume)

"""for index, item in enumerate(candleBars): #Version 2
    #print(item)
    #print(item[1]["BTC/USD"])
    #print(item[1]["BTC/USD"][0])
    #print(item[1]["BTC/USD"][0].timestamp)
    #print(item[1]["BTC/USD"][index])
    time.append(item[1]["BTC/USD"][index].timestamp)
    #time.append(item[1]["BTC/USD"][index]["timestamp"])
    open.append(item[1]["BTC/USD"][index].open)
    high.append(item[1]["BTC/USD"][index].high)
    low.append(item[1]["BTC/USD"][index].low)
    close.append(item[1]["BTC/USD"][index].close)
    volume.append(item[1]["BTC/USD"][index].volume)"""
###########

#Create/set datafram from list
dataframe = pandas.DataFrame(
    {
        #"time": [datetime.strptime(x, "%Y-%m-%dT%H:%M:S.#fZ") for x in time],  #formatting our timestamp
        #"open": [float(x.strip("0")) for x in open],                            #Removing traling 0's and converting to a float
        #"high": [float(x.strip("0")) for x in high],
        #"low": [float(x.strip("0")) for x in low],
        #"close": [float(x.strip("0")) for x in close],
        #"volume": [float(x.strip("0")) for x in volume]
        "time": [item for item in time],          #formatting our timestamp. List comprehension
        "open": [item for item in open],          #Removing traling 0's and converting to a float
        "high": [item for item in high],
        "low": [item for item in low],
        "close": [item for item in close],
        "volume": [item for item in volume]
    })

#Create dictionary from dataframe
dataframeDictionary = dataframe.to_dict("records") #"records" --> the return orientation 

#print(dataframeDictionary)
print(dataframeDictionary[:3])  #check if volume is 0. The three first items 

#dataframeDictionary should look like this
#[{'time': Timestamp('2026-04-01 00:00:00+0000', tz='UTC'), 'open': 68233.22, 'high': 68233.22, 'low': 68150.933, 'close': 68157.378, 'volume': 0.0}...{'time': Timestamp('2026-04-01 14:19:00+0000', tz='UTC'), 'open': 68234.432, 'high': 68289.925, 'low': 68198.25, 'close': 68279.6095, 'volume': 0.000476}...
#Our volumes are very small 

#Prettify the print output: so it is not just on one line
'''!pip install json
import json

def serializeTimestamp(_object):
    if isinstance(_object, pandas.Timestamp):
        return _object.isoformat() #Convert Timestamp object to ISO format string

    raise TypeError(f"Object of type {_object.__class__.__name__} is not JSON serializable")

print(json.dumps(dataframeDictionary, indent=4, default=serializeTimestamp))'''

def getDollarBars(_timeBars, _dollarThreshold):    #Function credit to Max Bodoia
    #Initialize an empty list of dollar bars
    dollarBars = []

    #Initialize the running dollar volume at 0
    runningVolume = 0

    #Initialize the running high and low with placeholder values
    runningHigh = 0
    runningLow = math.inf

    #For each time bar
    for i in range(len(_timeBars)):
        #Get the time, open, high, low, close and volume of the next bar
        nextTime, nextOpen, nextHigh, nextLow, nextClose, nextVolume = [_timeBars[i][k] for k in ["time", "open", "high", "low", "close", "volume"]]

        #Get the midpoint price of the next bar (the average of the open and the close)
        midpointPrice = (nextOpen + nextClose)/2

        #Get the approximate dollar volume of the bar using the volume and the midpoint price
        dollarVolume = nextVolume * midpointPrice

        #Update the running high and running low
        runningHigh, runningLow = max(runningHigh, nextHigh), min(runningLow, nextLow)

        #If the next bar's dollar volume would take us over the threshold...
        if dollarVolume + runningVolume >= _dollarThreshold:
            #Set the timestamp for the dollar bar as the timestamp at which the bar closed (i.e one after the timestamp of the last minutely bar included in the dollar bar)
            barTimestamp = nextTime + timedelta(minutes=1)

            #Add a new dollar bar to the list of dollar bars with the timestamp, running high/low, and next close
            dollarBars += [{"timestamp": barTimestamp, "open": nextOpen, "high": runningHigh, "low": runningLow, "close": nextClose}]

            #Reset the running volume to zero
            runningVolume = 0

            #Reset the running high and running low to placeholder values
            #runningHigh, runningLow = 0, math.inf #Version 2
            runningHigh = 0
            runningLow = math.inf

        #Else, increment the running volume
        else:
            runningVolume += dollarVolume

    return dollarBars   #Return the list of dollar bars


#Create bars
dollarBars = getDollarBars(dataframeDictionary, 500000) #5,000,000 is an arbritrarily selected threshold

#create dataframe
dataframe = pandas.DataFrame(dollarBars)

#View the first five entries
#dataframe.head()
print(dataframe.head())


#Insight
""" 
#Try: 
If volume is 0, the Alpaca free-tier crypto endpoint may not return real volume 
for minute bars. You could try switching to TimeFrame.Hour or TimeFrame.Day 
which tend to have populated volume fields, or lower your 
threshold dramatically (e.g. 50000, 5000, 500) just to confirm the function 
works when volume is non-zero.
"""