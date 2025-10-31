# TODO - Zostávajúce úlohy

**Projekt:** ML Trading System
**Aktualizované:** 31. október 2025
**Status:** Phase 1 COMPLETE ✅

---

## ✅ HOTOVO - Phase 1

- [x] Real-time data pipeline (50 S&P 500 stocks)
- [x] Docker infrastructure (InfluxDB + Grafana)
- [x] Grafana dashboard (11 panels)
- [x] Individual stock models (50 XGBoost models)
- [x] Adaptive feature selection
- [x] Multi-scale feature engineering
- [x] Neural network alternative
- [x] Hybrid portfolio management
- [x] Complete documentation
- [x] Git commit & push

---

## 🔧 OPRAVY A VYLEPŠENIA - PRIORITA HIGH

### 1. Grafana Dashboard - Dokončenie vizualizácie
**Status:** ČIASTOČNE FUNGUJE ⚠️

**Problém:**
- Niektoré panely zobrazujú data (Moving Averages, Trading Volume, RSI, MACD)
- Hlavné grafy sú prázdne (Top 5 Stocks price lines)
- "Current Stock Prices" table - No data
- "Volatility Ranking" - No data

**Riešenie:**
```bash
Súbor: production/grafana/dashboards/stock_trading_dashboard.json
```

**Kroky:**
1. Overiť Flux queries pre prázdne panely
2. Skontrolovať aggregateWindow nastavenia
3. Upraviť time range pre line charts
4. Otestovať s real data v InfluxDB

**Odhadovaný čas:** 1-2 hodiny

---

### 2. InfluxDB Data Retention Policy
**Status:** NENASTAVENÉ ⚠️

**Problém:**
- Aktuálne: default retention (neobmedzené)
- Production: potrebujeme definovanú retention policy

**Riešenie:**
```python
# V realtime_data_pipeline.py alebo init script
# Nastaviť retention na 30 dní
```

**Kroky:**
1. Definovať retention policy (30d, 90d, alebo custom)
2. Implementovať v docker-compose.yml alebo init script
3. Otestovať automatické mazanie starých dát

**Odhadovaný čas:** 30 minút

---

### 3. Environment Variables Security
**Status:** HARDCODED ⚠️

**Problém:**
- Tokeny a heslá sú hardcoded v súboroch
- `.env` súbor neexistuje (len `.env.example`)

**Riešenie:**
```bash
# Vytvoriť .env súbor
# Použiť python-dotenv alebo docker-compose env_file
```

**Súbory na úpravu:**
- `production/realtime_data_pipeline.py`
- `production/docker-compose.yml`
- `production/test_influxdb_data.py`

**Kroky:**
1. Vytvoriť production `.env` súbor
2. Nahradiť hardcoded values s os.getenv()
3. Update docker-compose.yml na použitie env_file
4. Pridať `.env` do `.gitignore`

**Odhadovaný čas:** 1 hodina

---

### 4. Logging & Monitoring
**Status:** BASIC LOGGING ⚠️

**Problém:**
- Len basic console logging
- Žiadne error alerting
- Žiadny centralized logging

**Riešenie:**
```python
# Implementovať:
# - Structured logging (JSON format)
# - Log rotation
# - Error alerting (email/Slack)
```

**Kroky:**
1. Použiť `structlog` alebo `python-json-logger`
2. Nastaviť log rotation (max size, max files)
3. Implementovať error notifications
4. Pridať health check endpoint

**Odhadovaný čas:** 2-3 hodiny

---

### 5. Data Pipeline Error Handling
**Status:** BASIC ⚠️

**Problém:**
- Ak Yahoo Finance zlyhá, celá iterácia zlyhá
- Žiadny retry mechanism
- Žiadne fallback data source

**Riešenie:**
```python
# Implementovať:
# - Retry logic s exponential backoff
# - Fallback data sources (Alpha Vantage, IEX Cloud)
# - Graceful degradation
```

**Kroky:**
1. Pridať `tenacity` library pre retries
2. Implementovať fallback na iný data source
3. Cache last known good data
4. Alert pri dlhodobom výpadku

**Odhadovaný čas:** 3-4 hodiny

---

## 🚀 PHASE 2 - Feature Engineering Service

### Status: ⏳ NOT STARTED

**Popis:**
FastAPI service pre advanced feature calculation a caching.

**Komponenty:**

#### 1. FastAPI Service
```python
production/feature_service/
├── app.py                 # FastAPI application
├── indicators.py          # Advanced technical indicators
├── cache.py              # Redis caching layer
├── models.py             # Pydantic models
├── Dockerfile            # Container build
└── requirements.txt      # Dependencies
```

**Features:**
- RESTful API pre feature calculation
- Redis cache pre performance
- Support pre custom indicators
- Batch processing
- WebSocket pre real-time updates

**Endpointy:**
```
GET  /features/{ticker}           # Get all features
GET  /features/{ticker}/{indicator} # Get specific indicator
POST /features/batch              # Batch calculation
WS   /features/stream             # Real-time stream
```

**Odhadovaný čas:** 2-3 dni

---

#### 2. Advanced Technical Indicators

**Implementovať:**
- Bollinger Bands (BB)
- Average True Range (ATR)
- Stochastic Oscillator
- Williams %R
- On-Balance Volume (OBV)
- Accumulation/Distribution
- Chaikin Money Flow
- Fibonacci Retracements
- Ichimoku Cloud
- Parabolic SAR

**Library:** `ta-lib` alebo `pandas-ta`

**Odhadovaný čas:** 1-2 dni

---

#### 3. Feature Caching System

**Implementácia:**
- Redis pre hot cache (last 24h)
- PostgreSQL pre historical cache
- Cache invalidation strategy
- Automatic cache warming

**Kroky:**
1. Setup Redis container
2. Implementovať cache decorator
3. Define TTL policies
4. Monitoring cache hit rate

**Odhadovaný čas:** 1 deň

---

## 🤖 PHASE 3 - Model Predictions Integration

### Status: ⏳ NOT STARTED

**Popis:**
Integration trained models do production pipeline s real-time predictions.

**Komponenty:**

#### 1. Model Serving Service
```python
production/prediction_service/
├── app.py              # FastAPI application
├── model_loader.py     # Load XGBoost models
├── predictor.py        # Prediction logic
├── versioning.py       # Model versioning
├── Dockerfile
└── requirements.txt
```

**Features:**
- Load all 50 stock models
- Real-time predictions
- Model versioning (A/B testing)
- Confidence scores
- Prediction explainability (SHAP)

**Endpointy:**
```
POST /predict/{ticker}        # Single prediction
POST /predict/batch           # Batch predictions
GET  /models/{ticker}         # Model info
POST /models/{ticker}/update  # Update model
```

**Odhadovaný čas:** 3-4 dni

---

#### 2. Model Versioning System

**Implementácia:**
- Model registry (MLflow alebo custom)
- A/B testing framework
- Automatic model rollback
- Performance tracking

**Kroky:**
1. Setup MLflow server
2. Register all models
3. Implement A/B testing logic
4. Track prediction performance

**Odhadovaný čas:** 2 dni

---

#### 3. SHAP Explainability

**Implementácia:**
- Calculate SHAP values pre každú predikciu
- API endpoint pre explanation
- Vizualizácia v Grafane

**Kroky:**
1. Precompute SHAP explainers
2. Cache SHAP values
3. Create explanation endpoint
4. Add Grafana panel

**Odhadovaný čas:** 1-2 dni

---

## 💼 PHASE 4 - Portfolio Management

### Status: ⏳ NOT STARTED

**Popis:**
Automatický portfolio manager s risk management.

**Komponenty:**

#### 1. Position Sizing Algorithm
```python
production/portfolio_service/
├── app.py                # FastAPI application
├── position_sizing.py    # Kelly Criterion, Risk Parity
├── risk_manager.py       # Stop-loss, Take-profit
├── rebalancer.py         # Portfolio rebalancing
├── Dockerfile
└── requirements.txt
```

**Features:**
- Kelly Criterion position sizing
- Risk parity allocation
- Maximum position size limits
- Correlation-based diversification

**Odhadovaný čas:** 2-3 dni

---

#### 2. Risk Management System

**Implementácia:**
- Stop-loss triggers (fixed, trailing, time-based)
- Take-profit targets
- Maximum drawdown protection
- Portfolio-level risk limits
- VaR (Value at Risk) calculation

**Kroky:**
1. Implement stop-loss logic
2. Add take-profit triggers
3. Portfolio risk monitoring
4. Alert system

**Odhadovaný čas:** 2 dni

---

#### 3. Portfolio Rebalancing

**Implementácia:**
- Time-based rebalancing (daily, weekly)
- Threshold-based rebalancing
- Tax-loss harvesting
- Transaction cost optimization

**Kroky:**
1. Define rebalancing strategies
2. Calculate optimal trades
3. Minimize transaction costs
4. Execution simulation

**Odhadovaný čas:** 2 dni

---

## 📊 PHASE 5 - Monitoring & Alerting

### Status: ⏳ NOT STARTED

**Popis:**
Comprehensive monitoring a alerting system.

**Komponenty:**

#### 1. Prometheus Monitoring
```yaml
production/monitoring/
├── prometheus.yml        # Prometheus config
├── alertmanager.yml      # Alert rules
├── grafana-monitoring/   # Monitoring dashboards
└── docker-compose.monitoring.yml
```

**Metrics to track:**
- Data pipeline health (success rate, latency)
- Model prediction accuracy
- API response times
- Database performance
- System resources (CPU, RAM, disk)

**Odhadovaný čas:** 2 dni

---

#### 2. Alert Rules

**Implementovať alerty pre:**
- Data pipeline failures (>5 minutes no data)
- Model prediction errors (>10% error rate)
- System overload (CPU >80%, RAM >90%)
- Database connection issues
- API downtime

**Channels:**
- Email notifications
- Slack/Discord webhooks
- SMS (critical alerts)

**Odhadovaný čas:** 1 deň

---

#### 3. Performance Dashboards

**Grafana dashboards:**
- System health overview
- Model performance tracking
- Portfolio performance
- API metrics
- Cost tracking

**Odhadovaný čas:** 1-2 dni

---

## 🔄 PHASE 6 - Automation & Optimization

### Status: ⏳ NOT STARTED

**Popis:**
CI/CD pipeline a automated model retraining.

**Komponenty:**

#### 1. CI/CD Pipeline
```yaml
.github/workflows/
├── test.yml              # Run tests
├── deploy.yml            # Deploy to production
├── model-training.yml    # Automated training
└── docker-build.yml      # Build containers
```

**Features:**
- Automated testing (pytest)
- Linting (black, flake8)
- Docker image building
- Automated deployment
- Rollback mechanism

**Odhadovaný čas:** 2-3 dni

---

#### 2. Automated Model Retraining

**Implementácia:**
- Scheduled retraining (weekly/monthly)
- Trigger-based retraining (performance drop)
- Automatic validation
- A/B testing new models
- Automatic deployment

**Kroky:**
1. Setup training pipeline
2. Define training schedule
3. Validation metrics
4. Automatic deployment

**Odhadovaný čas:** 3-4 dni

---

#### 3. Performance Optimization

**Oblasti optimalizácie:**
- Database query optimization
- Caching strategy
- API response time
- Memory usage
- Batch processing

**Odhadovaný čas:** Ongoing

---

## 🔐 SECURITY & COMPLIANCE

### Status: ⚠️ NEEDS ATTENTION

**Úlohy:**

#### 1. Authentication & Authorization
- [ ] API key authentication
- [ ] JWT tokens
- [ ] Role-based access control (RBAC)
- [ ] API rate limiting

**Odhadovaný čas:** 2 dni

---

#### 2. Data Encryption
- [ ] TLS/SSL certificates
- [ ] Database encryption at rest
- [ ] Secure credential storage (Vault)
- [ ] API encryption in transit

**Odhadovaný čas:** 1-2 dni

---

#### 3. Audit Logging
- [ ] Log all API access
- [ ] Track model predictions
- [ ] User activity logs
- [ ] Compliance reporting

**Odhadovaný čas:** 1 deň

---

## 📚 DOCUMENTATION

### Status: ⚠️ PARTIAL

**Potrebné dokumenty:**

#### 1. API Documentation
- [ ] OpenAPI/Swagger spec
- [ ] Usage examples
- [ ] Authentication guide
- [ ] Rate limits documentation

**Odhadovaný čas:** 1 deň

---

#### 2. Deployment Guide
- [ ] Production deployment steps
- [ ] Infrastructure requirements
- [ ] Scaling guidelines
- [ ] Disaster recovery plan

**Odhadovaný čas:** 1 deň

---

#### 3. User Guide
- [ ] End-user documentation
- [ ] Dashboard usage
- [ ] Troubleshooting guide
- [ ] FAQ

**Odhadovaný čas:** 1 deň

---

## 🧪 TESTING

### Status: ⚠️ MINIMAL

**Potrebné testy:**

#### 1. Unit Tests
- [ ] Data pipeline tests
- [ ] Model loading tests
- [ ] Feature calculation tests
- [ ] API endpoint tests

**Coverage target:** >80%

**Odhadovaný čas:** 3-4 dni

---

#### 2. Integration Tests
- [ ] End-to-end pipeline test
- [ ] Database integration test
- [ ] Model prediction test
- [ ] API integration test

**Odhadovaný čas:** 2-3 dni

---

#### 3. Performance Tests
- [ ] Load testing (Apache Bench, Locust)
- [ ] Stress testing
- [ ] Endurance testing
- [ ] Spike testing

**Odhadovaný čas:** 2 dni

---

## 📱 MOBILE/WEB INTERFACE (OPTIONAL)

### Status: ⏳ NOT STARTED

**Možné rozšírenia:**

#### 1. Web Dashboard
- React/Vue.js frontend
- Real-time charts (WebSocket)
- Portfolio overview
- Trade history
- Settings management

**Odhadovaný čas:** 2-3 týždne

---

#### 2. Mobile App
- React Native alebo Flutter
- Push notifications
- Real-time alerts
- Portfolio tracking

**Odhadovaný čas:** 3-4 týždne

---

## 🎯 PRIORITY MATRIX

### IMMEDIATE (Do týždňa)
1. ✅ Grafana dashboard - dokončiť vizualizácie
2. ✅ Environment variables security
3. ✅ Data retention policy
4. ✅ Error handling vylepšenia

### SHORT TERM (Do mesiaca)
1. 🔵 Phase 2: Feature Engineering Service
2. 🔵 Phase 3: Model Predictions Integration
3. 🔵 Basic monitoring & alerting
4. 🔵 Unit tests (core functionality)

### MEDIUM TERM (1-3 mesiace)
1. 🟡 Phase 4: Portfolio Management
2. 🟡 Phase 5: Complete Monitoring
3. 🟡 Phase 6: CI/CD Pipeline
4. 🟡 Security & Compliance
5. 🟡 Integration tests

### LONG TERM (3+ mesiace)
1. 🟢 Web/Mobile interface
2. 🟢 Advanced ML features
3. 🟢 Multi-broker integration
4. 🟢 Social trading features

---

## 💰 COST ESTIMATION

### Infrastructure Costs (Monthly)
```
DigitalOcean/AWS:
- 2x vCPU, 4GB RAM: $24/month
- 50GB SSD Storage:  $5/month
- Managed Database:  $15/month
- Load Balancer:     $10/month
Total:               ~$54/month
```

### Development Time
```
Phase 2: 5-7 days
Phase 3: 7-10 days
Phase 4: 6-8 days
Phase 5: 4-6 days
Phase 6: 7-10 days
Total:   ~30-40 days (1-2 mesiace full-time)
```

---

## 📋 CHECKLIST PRE PRODUCTION LAUNCH

### Pre-Launch Checklist
- [ ] All tests passing (unit + integration)
- [ ] Security audit complete
- [ ] Performance testing done
- [ ] Documentation complete
- [ ] Monitoring & alerts configured
- [ ] Backup & disaster recovery plan
- [ ] SSL certificates installed
- [ ] Environment variables secured
- [ ] Error handling comprehensive
- [ ] Logging properly configured
- [ ] Health checks implemented
- [ ] Load balancer configured
- [ ] Database backups automated
- [ ] API rate limiting enabled
- [ ] User acceptance testing complete

---

## 🔗 USEFUL LINKS

- **GitHub Repository:** https://github.com/minothegreat881/trade
- **InfluxDB Docs:** https://docs.influxdata.com/
- **Grafana Docs:** https://grafana.com/docs/
- **FastAPI Docs:** https://fastapi.tiangolo.com/
- **XGBoost Docs:** https://xgboost.readthedocs.io/

---

## 📝 NOTES

### Known Issues
1. Grafana dashboard - niektoré grafy prázdne (line charts)
2. Hardcoded credentials in source files
3. Žiadny automated testing
4. Minimal error handling
5. No backup strategy

### Future Ideas
- Multi-timeframe analysis (1min, 5min, 15min, 1h, 1d)
- Sentiment analysis from news/Twitter
- Options trading strategies
- Crypto support
- Paper trading mode
- Backtesting platform with UI
- Strategy marketplace

---

**Last Updated:** 31. október 2025
**Next Review:** 7. november 2025

**Status Summary:**
- ✅ Phase 1: COMPLETE
- ⚠️ Immediate fixes: IN PROGRESS
- ⏳ Phase 2-6: PLANNED
- 🎯 Production Ready: 30-40 days estimate
