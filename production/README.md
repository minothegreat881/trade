# Production Deployment - Phase 1
## Real-time Data Pipeline & Monitoring

This directory contains the production infrastructure for the ML Trading System.

## Architecture

```
┌─────────────────┐     ┌──────────────┐     ┌──────────────┐
│  Yahoo Finance  │ --> │  Python      │ --> │  InfluxDB    │
│  (Real-time)    │     │  Pipeline    │     │  (Time Series)│
└─────────────────┘     └──────────────┘     └──────────────┘
                                                      │
                                                      v
                                              ┌──────────────┐
                                              │   Grafana    │
                                              │  (Dashboard) │
                                              └──────────────┘
```

## Components

### 1. Data Pipeline (`realtime_data_pipeline.py`)
- Fetches real-time stock data for 50 S&P 500 stocks
- Calculates technical indicators: RSI, MACD, MA, Volatility
- Updates InfluxDB every 60 seconds
- Includes error handling and logging

### 2. InfluxDB
- Time-series database for stock data
- Optimized for high-frequency writes
- Retains 30 days of data by default

### 3. Grafana
- Real-time visualization dashboard
- 11 panels showing:
  - Live stock prices
  - Technical indicators (RSI, MACD)
  - Moving averages
  - Volatility rankings
  - Volume analysis
  - Market statistics

## Installation & Setup

### Prerequisites

- Docker Desktop installed and running
- Python 3.11+ with required packages:
  ```bash
  pip install yfinance pandas numpy influxdb-client
  ```

### Step 1: Start Infrastructure

Navigate to the production directory:
```bash
cd production
```

Start Docker containers:
```bash
docker-compose up -d
```

This will start:
- InfluxDB on `http://localhost:8086`
- Grafana on `http://localhost:3000`

Wait 30 seconds for containers to initialize.

### Step 2: Verify Services

Check if services are running:
```bash
docker-compose ps
```

You should see both `trading_influxdb` and `trading_grafana` as `Up`.

Check logs if needed:
```bash
docker-compose logs influxdb
docker-compose logs grafana
```

### Step 3: Start Data Pipeline

Run the Python data pipeline:
```bash
python realtime_data_pipeline.py
```

The pipeline will:
1. Connect to InfluxDB
2. Fetch data for 50 stocks every minute
3. Calculate technical indicators
4. Write to InfluxDB
5. Log all activities to `production/logs/data_pipeline.log`

### Step 4: Access Grafana Dashboard

1. Open browser: `http://localhost:3000`
2. Login credentials:
   - Username: `admin`
   - Password: `admin`
3. Navigate to Dashboards > "Stock Trading Dashboard - Real-time"

The dashboard will automatically load and show real-time data.

## Configuration

### InfluxDB Settings

Default configuration (in `realtime_data_pipeline.py`):
```python
INFLUX_URL = "http://localhost:8086"
INFLUX_TOKEN = "your-super-secret-token"
INFLUX_ORG = "trading-org"
INFLUX_BUCKET = "stock-data"
```

**IMPORTANT**: Change `INFLUX_TOKEN` in production!

Update in both:
- `realtime_data_pipeline.py` (line 39)
- `docker-compose.yml` (line 19)
- `grafana/provisioning/datasources/influxdb.yml` (line 14)

### Stock Selection

Edit `TICKERS` list in `realtime_data_pipeline.py` (line 44-50) to change which stocks are tracked.

Default: Top 50 S&P 500 stocks by market cap.

### Data Refresh Rate

Default: 60 seconds (line 300 in `realtime_data_pipeline.py`)

To change:
```python
time.sleep(60)  # Change to desired seconds
```

**Note**: Yahoo Finance has rate limits. Don't go below 30 seconds.

## Dashboard Panels

### Panel 1: Top 5 Stocks - Real-time Prices
Line chart showing AAPL, MSFT, GOOGL, AMZN, NVDA prices over last hour.

### Panel 2: Current Stock Prices
Table with all 50 stocks and latest prices.

### Panel 3: RSI (14) - Top 3 Stocks
RSI indicator with overbought (>70) and oversold (<30) zones.

### Panel 4: MACD - AAPL
MACD and Signal lines for Apple stock.

### Panel 5: Volatility Ranking (20d)
Heatmap showing top 20 most volatile stocks.

### Panel 6: Moving Averages - AAPL
Price with SMA 10, 20, 50, 200 overlays.

### Panel 7: Trading Volume - Top 3 Stocks
Bar chart showing hourly volume.

### Panel 8-11: Market Statistics
- Active stocks count
- Average market volatility
- Average market RSI
- Recent data points

## Monitoring & Logs

### Pipeline Logs

All activity logged to:
```
production/logs/data_pipeline.log
```

View live logs:
```bash
tail -f production/logs/data_pipeline.log
```

### Docker Logs

```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f influxdb
docker-compose logs -f grafana
```

## Troubleshooting

### Issue: Pipeline fails to connect to InfluxDB

**Solution**:
1. Check if InfluxDB is running: `docker-compose ps`
2. Verify port 8086 is not in use: `netstat -an | findstr 8086`
3. Check InfluxDB logs: `docker-compose logs influxdb`

### Issue: No data in Grafana

**Solution**:
1. Verify pipeline is running and writing data
2. Check pipeline logs for errors
3. Verify InfluxDB datasource in Grafana:
   - Settings > Data Sources > InfluxDB_Trading
   - Click "Test" - should show "Data source is working"

### Issue: Dashboard shows "No data"

**Solution**:
1. Wait 2-3 minutes for initial data collection
2. Check time range (default: last 1 hour)
3. Verify data in InfluxDB:
   ```bash
   docker exec -it trading_influxdb influx query '
   from(bucket: "stock-data")
     |> range(start: -1h)
     |> filter(fn: (r) => r["_measurement"] == "stock_data")
     |> limit(n: 10)
   ' --org trading-org --token your-super-secret-token
   ```

### Issue: Permission denied on logs directory

**Solution**:
```bash
mkdir -p production/logs
chmod 755 production/logs
```

## Stopping Services

### Stop pipeline
Press `Ctrl+C` in the terminal running the pipeline.

### Stop Docker containers
```bash
docker-compose down
```

### Stop and remove all data
```bash
docker-compose down -v
```

**WARNING**: This will delete all historical data!

## Data Retention

InfluxDB default retention: Unlimited

To set retention policy (e.g., 30 days):
```bash
docker exec -it trading_influxdb influx bucket update \
  --id $(docker exec -it trading_influxdb influx bucket list --org trading-org --token your-super-secret-token --name stock-data --json | jq -r '.[0].id') \
  --retention 720h \
  --org trading-org \
  --token your-super-secret-token
```

## Performance

Expected resource usage:
- InfluxDB: ~200-500 MB RAM
- Grafana: ~100-200 MB RAM
- Python Pipeline: ~50-100 MB RAM
- Total: <1 GB RAM

Data storage (per day):
- 50 stocks × 1440 minutes × ~15 fields = ~1 MB/day
- 30 days retention = ~30 MB total

## Security Considerations

**Before deploying to cloud/production:**

1. Change default passwords:
   - InfluxDB admin password
   - Grafana admin password
   - InfluxDB token

2. Use environment variables instead of hardcoded credentials

3. Enable HTTPS for Grafana and InfluxDB

4. Restrict network access using firewall rules

5. Enable authentication on all services

## Next Steps

Phase 1 ✅ COMPLETE

Ready for Phase 2:
- Feature Engineering Service
- Real-time feature calculation
- Integration with trained models

## Support

For issues or questions:
1. Check logs in `production/logs/`
2. Review Docker logs: `docker-compose logs`
3. Verify all services are healthy: `docker-compose ps`

## File Structure

```
production/
├── docker-compose.yml              # Docker orchestration
├── realtime_data_pipeline.py       # Python data pipeline
├── README.md                        # This file
├── logs/                            # Pipeline logs
│   └── data_pipeline.log
└── grafana/
    ├── provisioning/
    │   ├── datasources/
    │   │   └── influxdb.yml        # InfluxDB connection
    │   └── dashboards/
    │       └── dashboard.yml       # Dashboard config
    └── dashboards/
        └── stock_trading_dashboard.json  # Dashboard definition
```

## Version

- Phase: 1
- Version: 1.0.0
- Last Updated: 2025-10-31
