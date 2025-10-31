"""
Test InfluxDB Data - Quick verification
"""
from influxdb_client import InfluxDBClient

# InfluxDB Configuration
INFLUX_URL = "http://localhost:8086"
INFLUX_TOKEN = "your-super-secret-token"
INFLUX_ORG = "trading-org"
INFLUX_BUCKET = "stock-data"

print("="*80)
print("TESTING INFLUXDB DATA")
print("="*80)

# Connect
client = InfluxDBClient(url=INFLUX_URL, token=INFLUX_TOKEN, org=INFLUX_ORG)
query_api = client.query_api()

# Query 1: Count all records
print("\n1. Total records in database:")
query = f'''
from(bucket: "{INFLUX_BUCKET}")
  |> range(start: -30d)
  |> filter(fn: (r) => r["_measurement"] == "stock_data")
  |> count()
'''

result = query_api.query(query)
for table in result:
    for record in table.records:
        print(f"   {record.get_field()}: {record.get_value()} records")

# Query 2: List unique tickers
print("\n2. Unique tickers:")
query = f'''
from(bucket: "{INFLUX_BUCKET}")
  |> range(start: -30d)
  |> filter(fn: (r) => r["_measurement"] == "stock_data")
  |> keep(columns: ["ticker"])
  |> distinct(column: "ticker")
'''

result = query_api.query(query)
tickers = []
for table in result:
    for record in table.records:
        ticker = record.values.get('ticker')
        if ticker and ticker not in tickers:
            tickers.append(ticker)

print(f"   Found {len(tickers)} tickers: {', '.join(sorted(tickers)[:10])}...")

# Query 3: Latest data for AAPL
print("\n3. Latest AAPL data:")
query = f'''
from(bucket: "{INFLUX_BUCKET}")
  |> range(start: -30d)
  |> filter(fn: (r) => r["_measurement"] == "stock_data")
  |> filter(fn: (r) => r["ticker"] == "AAPL")
  |> last()
'''

result = query_api.query(query)
for table in result:
    for record in table.records:
        print(f"   {record.get_field()}: {record.get_value()}")

# Query 4: List all fields
print("\n4. Available fields:")
query = f'''
from(bucket: "{INFLUX_BUCKET}")
  |> range(start: -30d)
  |> filter(fn: (r) => r["_measurement"] == "stock_data")
  |> keep(columns: ["_field"])
  |> distinct(column: "_field")
'''

result = query_api.query(query)
fields = []
for table in result:
    for record in table.records:
        field = record.values.get('_field')
        if field and field not in fields:
            fields.append(field)

print(f"   Found {len(fields)} fields: {', '.join(sorted(fields))}")

client.close()
print("\n" + "="*80)
print("TEST COMPLETE!")
print("="*80)
