# RÝCHLY ŠTART - Slovensky

## 1. Spusti Docker kontajnery

Otvor PowerShell alebo CMD v priečinku `production`:

```bash
cd C:\Users\milan\Desktop\Git-Projects\ml_trading_system\production
docker-compose up -d
```

Počkaj 30 sekúnd, kým sa všetko inicializuje.

## 2. Spusti Python skript (zber dát)

V tom istom okne:

```bash
python realtime_data_pipeline.py
```

**Alebo ešte jednoduchšie - spusti `start.bat`:**
```bash
start.bat
```

Skript sa spustí a uvidíš:
```
================================================================================
REAL-TIME DATA PIPELINE STARTING
================================================================================
InfluxDB URL: http://localhost:8086
Bucket: stock-data
Tickers: 50
================================================================================

Processing AAPL...
  AAPL: Data written to InfluxDB
Processing MSFT...
  MSFT: Data written to InfluxDB
...
```

**NECHAJ TO BEŽAŤ!** Každú minútu sťahuje nové dáta.

## 3. Otvor Grafana Dashboard

Otvor prehliadač (Chrome, Firefox...):

```
http://localhost:3000
```

### Prihlásenie do Grafany:
- **Username:** `admin`
- **Password:** `admin`

Grafana ťa možno požiada zmeniť heslo → klikni **Skip** (preskočiť)

### Otvor Dashboard:

1. Klikni na hamburger menu **≡** (vľavo hore)
2. Klikni **Dashboards**
3. Uvidíš: **"Stock Trading Dashboard - Real-time"**
4. Klikni naň

**HOTOVO!** Uvidíš 11 panelov s live dátami:

- ✅ Ceny akcií v reálnom čase
- ✅ RSI indikátory
- ✅ MACD analýza
- ✅ Volatilita
- ✅ Moving averages
- ✅ Volume
- ✅ Štatistiky

Dashboard sa automaticky aktualizuje každých 5 sekúnd!

## 4. Čo robiť keď niečo nefunguje?

### Žiadne dáta v Grafane?

**Riešenie 1:** Počkaj 2-3 minúty (prvé dáta sa zbierajú)

**Riešenie 2:** Skontroluj, či beží Python skript:
- Mal by si vidieť výpis: "Processing AAPL...", "Processing MSFT..." atď.

**Riešenie 3:** Skontroluj Docker kontajnery:
```bash
docker-compose ps
```

Mali by bežať oba (influxdb + grafana):
```
NAME                 STATUS
trading_influxdb     Up 2 minutes (healthy)
trading_grafana      Up 2 minutes (healthy)
```

### Python skript hlási chybu?

**Chyba: Cannot connect to InfluxDB**
→ Docker kontajnery ešte nie sú ready, počkaj 30 sekúnd a skús znova

**Chyba: Module not found**
→ Nainštaluj balíčky:
```bash
pip install yfinance pandas numpy influxdb-client
```

### Grafana dashboard je prázdny?

1. Klikni na **Settings** (ikona ozubeného kolieska hore)
2. Klikni **Data sources** v ľavom menu
3. Klikni **InfluxDB_Trading**
4. Klikni **Test** (dole)
5. Malo by ukázať: **"Data source is working"**

Ak nie, skontroluj token v súboroch (mal by byť všade rovnaký):
- `realtime_data_pipeline.py` (riadok 39)
- `docker-compose.yml` (riadok 19)

## 5. Zastavenie systému

### Zastaviť Python skript:
Stlač `Ctrl+C` v okne, kde beží

### Zastaviť Docker kontajnery:
```bash
docker-compose down
```

### Zastaviť všetko vrátane vymazania dát:
```bash
docker-compose down -v
```
**POZOR:** Toto vymaže všetky historické dáta!

## 6. TOKEN - Čo je to?

**Token** je ako heslo medzi Python skriptom a InfluxDB databázou.

**Už je nastavený automaticky:**
- Token: `your-super-secret-token`
- Python ho používa na zápis dát do InfluxDB
- Grafana ho používa na čítanie dát z InfluxDB

**NEMUSÍŠ NIC ROBIŤ S TOKENOM!**

Je už nastavený vo všetkých potrebných súboroch.

## 7. Prihlasovacie údaje (zhrnutie)

### Grafana (TU sa prihlasuj TY):
```
URL:      http://localhost:3000
Username: admin
Password: admin
```

### InfluxDB (Tu sa prihlasovať NEMUSÍŠ):
```
URL:      http://localhost:8086
Username: admin
Password: adminpassword123
Token:    your-super-secret-token
```

Python skript používa token automaticky.

## Čo uvidíš v Grafane?

Po prihlásení a otvorení dashboardu uvidíš:

### Panel 1: Top 5 Stocks - Real-time Prices
Graf s cenami AAPL, MSFT, GOOGL, AMZN, NVDA (posledná hodina)

### Panel 2: Current Stock Prices
Tabuľka so všetkými 50 akciami a aktuálnymi cenami

### Panel 3: RSI (14) - Top 3 Stocks
RSI indikátor pre top 3 akcie
- Červená zóna >70 = prekúpené
- Červená zóna <30 = prepredané

### Panel 4: MACD - AAPL
MACD indikátor pre Apple

### Panel 5: Volatility Ranking (20d)
Volatilita akcií zoradená od najvyššej
- Červená = vysoká volatilita
- Zelená = nízka volatilita

### Panel 6: Moving Averages - AAPL
Cena Apple s kĺzavými priemrami (SMA 10, 20, 50, 200)

### Panel 7: Trading Volume - Top 3 Stocks
Objem obchodovania (hodinové stĺpce)

### Panel 8-11: Štatistiky
- Počet aktívnych akcií
- Priemerná volatilita trhu
- Priemerný RSI trhu
- Počet dátových bodov (posledná minúta)

Dashboard sa **automaticky aktualizuje každých 5 sekúnd!**

## Hotovo!

Teraz máš bežiaci:
✅ Real-time zber dát (každú minútu)
✅ Time-series databázu (InfluxDB)
✅ Live dashboard s grafmi (Grafana)

Všetko pre 50 S&P 500 akcií!
