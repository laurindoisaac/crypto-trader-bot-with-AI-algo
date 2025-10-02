import logging
import os
import smtplib
import datetime
import calendar
import pandas as pd
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

LOG_FILE = 'output/trade_log.xlsx'
os.makedirs('output', exist_ok=True)

def initialize_trade_log():
    """Creates the trade log Excel file with all necessary columns if it doesn't exist."""
    if not os.path.exists(LOG_FILE) or os.path.getsize(LOG_FILE) == 0:
        # --- FIX: Added 'OrderID' to distinguish live vs paper trades ---
        all_columns = [
            'Timestamp', 'OrderID', 'Symbol', 'TradeType', 'EntryPrice', 'ExitPrice', 
            'Quantity', 'ProfitLoss', 'ProfitLoss_Pct', 'Status', 'Strategy', 'Rationale'
        ]
        pd.DataFrame(columns=all_columns).to_excel(LOG_FILE, index=False)
        logging.info(f"Trade log created at {LOG_FILE}")

def log_trade(trade_details):
    """Appends a single trade record to the Excel log file, ensuring all columns are present."""
    try:
        if trade_details.get('EntryPrice') and trade_details.get('Quantity') and trade_details['EntryPrice'] > 0 and trade_details['Quantity'] > 0:
            pnl_pct = (trade_details.get('ProfitLoss', 0) / (trade_details['EntryPrice'] * trade_details['Quantity'])) * 100
            trade_details['ProfitLoss_Pct'] = round(pnl_pct, 2)
        else:
            trade_details['ProfitLoss_Pct'] = 0.0

        all_columns = ['Timestamp', 'OrderID', 'Symbol', 'TradeType', 'EntryPrice', 'ExitPrice', 'Quantity', 'ProfitLoss', 'ProfitLoss_Pct', 'Status', 'Strategy', 'Rationale']
        
        # Ensure all keys exist in the dictionary to prevent errors
        for col in all_columns:
            trade_details.setdefault(col, None) 
        
        # Read existing log or create a new DataFrame
        if os.path.exists(LOG_FILE) and os.path.getsize(LOG_FILE) > 0:
            df = pd.read_excel(LOG_FILE)
        else:
            df = pd.DataFrame(columns=all_columns)

        new_trade_df = pd.DataFrame([trade_details])
        df = pd.concat([df, new_trade_df], ignore_index=True)
        df.to_excel(LOG_FILE, index=False)
        logging.info(f"Successfully logged trade for {trade_details['Symbol']}")
    except Exception as e:
        logging.error(f"Failed to log trade to Excel: {e}", exc_info=True)


def send_daily_report(config, date_str, no_trades_reason=None):
    """Reads the trade log and sends a daily report with segregated live and paper trade stats."""
    email_conf = config.get('email_settings', {})
    if not email_conf.get('send_daily_report', False):
        logging.info("Email reporting is disabled."); return
        
    try:
        today = pd.to_datetime(date_str).date()
        df = pd.read_excel(LOG_FILE) if os.path.exists(LOG_FILE) and os.path.getsize(LOG_FILE) > 0 else pd.DataFrame()
        
        if not df.empty and 'Timestamp' in df.columns:
            df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        
        # --- FIX: Generate segregated summary and get separate P/L values ---
        daily_html, live_pnl, paper_pnl = generate_daily_summary(df, today, no_trades_reason)
        
        # Build a dynamic subject line
        subject_parts = [f"Trading Report for {today.strftime('%d %b, %Y')}"]
        if live_pnl is not None:
            subject_parts.append(f"Live P/L: {live_pnl:,.2f}")
        if paper_pnl is not None:
            subject_parts.append(f"Paper P/L: {paper_pnl:,.2f}")
        subject = " | ".join(subject_parts)

        msg = MIMEMultipart()
        msg['From'] = email_conf['sender_email']
        msg['To'] = email_conf['receiver_email']
        msg['Subject'] = subject
        msg.attach(MIMEText(daily_html, 'html'))
        
        with smtplib.SMTP(email_conf['smtp_server'], email_conf['smtp_port']) as server:
            server.starttls()
            server.login(email_conf['sender_email'], email_conf['sender_password'])
            server.send_message(msg)
        logging.info("Successfully sent daily email report.")
    except Exception as e:
        logging.error(f"Failed to send email report: {e}", exc_info=True)

def _generate_summary_table(df, title):
    """Helper function to generate an HTML summary table for a given dataframe."""
    if df.empty:
        return f"<h3>{title}</h3><p>No trades were executed in this mode today.</p>", None

    total_pnl = df['ProfitLoss'].sum()
    wins = (df['ProfitLoss'] > 0).sum()
    losses = len(df) - wins
    win_rate = (wins / len(df) * 100) if len(df) > 0 else 0
    
    summary_html = f"""
    <h3>{title}</h3>
    <table style="width:400px;">
        <tr><td>Winning Trades</td><td>{wins}</td></tr>
        <tr><td>Losing Trades</td><td>{losses}</td></tr>
        <tr><td>Win Rate</td><td>{win_rate:.2f}%</td></tr>
        <tr><td><strong>Total P/L</strong></td><td><strong>{total_pnl:,.2f}</strong></td></tr>
    </table>
    """
    
    df_display = df.copy()
    for col in ['ProfitLoss', 'ProfitLoss_Pct', 'EntryPrice', 'ExitPrice']:
        if col in df_display.columns:
            df_display[col] = df_display[col].apply(lambda x: f'{x:,.2f}' if pd.notna(x) else 'N/A')
    
    df_display.fillna('', inplace=True)
    trades_html = df_display.to_html(index=False)
    
    return f"{summary_html}<h4>Trade Details:</h4>{trades_html}", total_pnl

def generate_daily_summary(df, date_obj, reason):
    """Generates the HTML summary for a single day, segregating live and paper trades."""
    html_style = "<style>body{font-family:Arial,sans-serif;margin:20px;} table{border-collapse:collapse;width:100%;} th,td{border:1px solid #ddd;padding:8px;text-align:left;} th{background-color:#f2f2f2;}</style>"
    
    header = f"<h2>Daily Summary: {date_obj.strftime('%d %b, %Y')}</h2>"
    
    if reason:
        body = f"{header}<p><strong>No trades were placed today. Reason:</strong> {reason}</p>"
        return f"<html><head>{html_style}</head><body>{body}</body></html>", None, None

    if df.empty or 'Timestamp' not in df.columns:
        body = f"{header}<p>No trades found in log file.</p>"
        return f"<html><head>{html_style}</head><body>{body}</body></html>", None, None

    daily_trades = df[df['Timestamp'].dt.date == date_obj].copy()
    if daily_trades.empty:
        body = f"{header}<p>No trades were executed today.</p>"
        return f"<html><head>{html_style}</head><body>{body}</body></html>", None, None

    # --- FIX: Segregate trades based on OrderID ---
    if 'OrderID' not in daily_trades.columns:
        # Fallback for old logs without OrderID
        live_trades = daily_trades
        paper_trades = pd.DataFrame()
    else:
        daily_trades['IsPaper'] = daily_trades['OrderID'].astype(str).str.startswith('PAPER_')
        live_trades = daily_trades[~daily_trades['IsPaper']].copy()
        paper_trades = daily_trades[daily_trades['IsPaper']].copy()

    live_html, live_pnl = _generate_summary_table(live_trades, "Live Trades Summary")
    paper_html, paper_pnl = _generate_summary_table(paper_trades, "Paper Trades Summary")
    
    full_html_body = f"{header}{live_html}<hr>{paper_html}"
    
    return f"<html><head>{html_style}</head><body>{full_html_body}</body></html>", live_pnl, paper_pnl

def send_monthly_report(config, date_str):
    """Generates and sends a summary report for the entire month's performance."""
    # This function can also be updated to segregate results if needed in the future.
    # For now, it will report on all trades combined for the month.
    email_conf = config.get('email_settings', {})
    if not email_conf.get('send_daily_report', False): return
    logging.info("Generating monthly report...")
    today = pd.to_datetime(date_str).date()
    
    try:
        df = pd.read_excel(LOG_FILE) if os.path.exists(LOG_FILE) and os.path.getsize(LOG_FILE) > 0 else pd.DataFrame()
        if df.empty:
            body = f"<p>No trades were executed during {today.strftime('%B %Y')}.</p>"
        else:
            df['Timestamp'] = pd.to_datetime(df['Timestamp'])
            monthly_trades = df[(df['Timestamp'].dt.year == today.year) & (df['Timestamp'].dt.month == today.month)]
            if monthly_trades.empty:
                body = f"<p>No trades were executed during {today.strftime('%B %Y')}.</p>"
            else:
                total_pnl = monthly_trades['ProfitLoss'].sum()
                wins = (monthly_trades['ProfitLoss'] > 0).sum()
                losses = len(monthly_trades) - wins
                win_rate = (wins / len(monthly_trades) * 100) if len(monthly_trades) > 0 else 0
                body = f"""
                <h3>Monthly Performance Summary for {today.strftime('%B %Y')}</h3>
                <table border="1" cellpadding="5" cellspacing="0" style="border-collapse: collapse; width: 400px;">
                    <tr><td>Total P/L</td><td>{total_pnl:,.2f}</td></tr>
                    <tr><td>Winning Trades</td><td>{wins}</td></tr>
                    <tr><td>Losing Trades</td><td>{losses}</td></tr>
                    <tr><td><strong>Monthly Win Rate</strong></td><td><strong>{win_rate:.2f}%</strong></td></tr>
                </table><hr>
                <h3>All Trades for the Month:</h3>
                {monthly_trades.to_html(index=False)}
                """
        
        subject = f"Monthly Trading Summary: {today.strftime('%B %Y')}"
        msg = MIMEMultipart(); msg['From'], msg['To'], msg['Subject'] = email_conf['sender_email'], email_conf['receiver_email'], subject
        msg.attach(MIMEText(f"<html><body>{body}</body></html>", 'html'))
        
        with smtplib.SMTP(email_conf['smtp_server'], email_conf['smtp_port']) as server:
            server.starttls(); server.login(email_conf['sender_email'], email_conf['sender_password'])
            server.send_message(msg)
        logging.info("Successfully sent monthly report.")
    except Exception as e:
        logging.error(f"Failed to send monthly report: {e}", exc_info=True)
