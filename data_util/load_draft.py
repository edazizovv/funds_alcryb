"""
api_key = '5EDZIQN359C119R6'

ts = TimeSeries(key=api_key, output_format='pandas')
# symbols = ['IVV', 'TLT', 'FXI', 'EFA', 'EEM']
symbols = ['VDE', 'GLD']
data_ = []
for symbol in symbols:
    data__ = ts.get_monthly(symbol=symbol)[0][['4. close']]
    data__ = data__.rename(columns={'4. close': symbol})
    data_.append(data__)
_data = pandas.concat(data_, axis=1)

_data.to_excel("./data/seriesx.xlsx")
"""