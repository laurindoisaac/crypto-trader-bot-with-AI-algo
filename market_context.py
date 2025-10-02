import logging
import datetime
import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import time
from kiteconnect.exceptions import DataException, NetworkException

class EconomicCalendar:
    """
    Dynamically scrapes and provides dates for major market-moving events.
    It fetches data from the official Federal Reserve and uses a reliable fallback for RBI.
    """
    def __init__(self):
        self.events = self._load_events()

    def _scrape_fed_dates(self):
        """Scrapes FOMC meeting dates from the Federal Reserve website."""
        fed_dates = {}
        try:
            url = "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm"
            headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36'}
            response = requests.get(url, headers=headers, timeout=15)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            panels = soup.find_all('div', class_='fomc-meeting')
            
            for panel in panels:
                year_tag = panel.find('h4')
                if not year_tag or not year_tag.text: continue
                
                # Extract year safely
                year_str_list = [s for s in year_tag.text.split() if s.isdigit()]
                if not year_str_list: continue
                year = year_str_list[0]

                meeting_entries = panel.find_all('div', class_='fomc-meeting__month')
                for entry in meeting_entries:
                    month_tag = entry.find('div', class_='fomc-meeting__month-name')
                    date_tags = entry.find_all('div', class_='fomc-meeting__date')
                    if month_tag and date_tags:
                        try:
                            month = month_tag.text.strip()
                            day = date_tags[-1].text.strip().split('-')[-1]
                            date_str = f"{day} {month} {year}"
                            event_date = datetime.datetime.strptime(date_str, "%d %B %Y").date()
                            fed_dates[event_date.strftime('%Y-%m-%d')] = "EVENT_FED_MEETING"
                        except (ValueError, IndexError):
                            continue # Skip if date parsing fails for an entry
            if fed_dates:
                logging.info(f"Successfully scraped {len(fed_dates)} FED meeting dates.")
            else:
                logging.warning("Scraping FED dates returned no results. Check website structure.")
        except Exception as e:
            logging.error(f"Could not scrape FED dates: {e}. Using fallback.")
            # Add a fallback for the current year in case of scraping failure
            fed_dates["2025-06-18"] = "EVENT_FED_MEETING"
            fed_dates["2025-07-30"] = "EVENT_FED_MEETING"
        return fed_dates

    def _scrape_rbi_dates(self):
        """Uses a static list for RBI dates as scraping is unreliable."""
        logging.warning("RBI date scraping is fragile; using a reliable hardcoded fallback list.")
        rbi_dates = {
            "2025-04-09": "EVENT_RBI_POLICY", "2025-06-06": "EVENT_RBI_POLICY",
            "2025-08-07": "EVENT_RBI_POLICY", "2025-10-01": "EVENT_RBI_POLICY",
            "2025-12-05": "EVENT_RBI_POLICY", "2026-02-06": "EVENT_RBI_POLICY",
        }
        logging.info(f"Loaded {len(rbi_dates)} RBI policy dates from static list.")
        return rbi_dates

    def _load_events(self):
        """Loads events from all sources."""
        logging.info("EconomicCalendar: Loading dynamic event dates...")
        fed = self._scrape_fed_dates()
        rbi = self._scrape_rbi_dates()
        return {**fed, **rbi}

    def get_event_for_date(self, date):
        """Returns the event for a given date, if any."""
        return self.events.get(date.strftime('%Y-%m-%d'))


class MarketConditionIdentifier:
    """Identifies the market conditions for a given date using multiple factors."""
    def __init__(self, kite, config):
        self.kite = kite
        self.config = config
        self.calendar = EconomicCalendar()
        self.vix_token = self._get_instrument_token('INDIA VIX', 'NSE')
        self.nifty_token = self._get_instrument_token('NIFTY 50', 'NSE')

    def _get_instrument_token(self, name, exchange):
        """Helper to find instrument token with a retry mechanism."""
        for i in range(3): # Retry up to 3 times
            try:
                instruments = self.kite.instruments(exchange)
                return [i['instrument_token'] for i in instruments if i['tradingsymbol'] == name][0]
            except (DataException, NetworkException) as e:
                logging.warning(f"Attempt {i+1}/3: Failed to fetch instruments for {exchange}. Retrying... Error: {e}")
                time.sleep(2 * (i + 1)) # Wait for 2, 4, 6 seconds
        raise ConnectionError(f"Could not fetch instruments for {exchange} after multiple retries.")

    def get_conditions_for_date(self, target_date):
        """Fetches all relevant data for a target date and returns a set of condition tags."""
        from_date = target_date - datetime.timedelta(days=60)
        to_date = target_date
        conditions = set()

        try:
            # 1. Get VIX condition
            vix_hist = pd.DataFrame(self.kite.historical_data(self.vix_token, from_date, to_date, "day"))
            vix_hist['date'] = pd.to_datetime(vix_hist['date']).dt.date
            today_vix_data = vix_hist[vix_hist['date'] == target_date]

            if not today_vix_data.empty:
                vix_value = today_vix_data.iloc[0]['close']
                if vix_value < 17: conditions.add('VIX_LOW')
                elif 17 <= vix_value < 25: conditions.add('VIX_MEDIUM')
                else: conditions.add('VIX_HIGH')
            
            # 2. Get Event condition
            if event := self.calendar.get_event_for_date(target_date):
                conditions.add(event)

            # 3. Get Implied Volatility (IV) proxy condition
            nifty_hist = pd.DataFrame(self.kite.historical_data(self.nifty_token, from_date, to_date, "day"))
            nifty_hist['returns'] = nifty_hist['close'].pct_change()
            iv_proxy = nifty_hist['returns'].rolling(window=7).std().iloc[-1]
            if not np.isnan(iv_proxy):
                avg_iv_proxy = nifty_hist['returns'].rolling(window=30).std().mean()
                if iv_proxy > avg_iv_proxy * 1.2: conditions.add('IV_HIGH')
                else: conditions.add('IV_LOW')

            return conditions if conditions else {'NORMAL'}
        except Exception as e:
            logging.warning(f"Could not determine full conditions for {target_date}: {e}")
            return {'UNKNOWN'}

