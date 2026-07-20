#!/usr/bin/env python3
"""
Pi Network Wallet Transaction Exporter

Queries the Pi Network Horizon API for ALL transactions and operations
for a given wallet address and exports them to CSV files.

Usage: python3 pi_wallet_export.py
"""

import csv
import json
import sys
import time
from datetime import datetime

try:
    import requests
except ImportError:
    print("Installing requests...")
    import subprocess
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'requests'])
    import requests

HORIZON_URL = "https://api.mainnet.minepi.com"
ACCOUNT_ID = "GABT7EMPGNCQSZM22DIYC4FNKHUVJTXITUF6Y5HNIWPU4GA7BHT4GC5G"
PAGE_LIMIT = 200

session = requests.Session()
session.headers.update({"Accept": "application/json"})


def fetch_all_pages(url):
    """Follow Horizon pagination links to fetch every record."""
    records = []
    page = 1
    while url:
        for attempt in range(5):
            try:
                resp = session.get(url, timeout=30)
                resp.raise_for_status()
                break
            except requests.exceptions.RequestException as e:
                wait = 2 ** attempt
                print(f"  Retry {attempt+1}/5 after error: {e} (waiting {wait}s)")
                time.sleep(wait)
        else:
            print(f"  FAILED after 5 retries, stopping pagination.")
            break

        data = resp.json()
        page_records = data.get("_embedded", {}).get("records", [])

        if not page_records:
            break

        records.extend(page_records)
        print(f"  Page {page}: fetched {len(page_records)} records "
              f"(total: {len(records)})")
        page += 1

        # Follow next page link
        next_link = data.get("_links", {}).get("next", {}).get("href")
        if next_link and page_records:
            url = next_link
        else:
            break

        time.sleep(0.25)

    return records


def export_transactions(account_id):
    """Fetch all transactions for the account."""
    print(f"\n[1/3] Fetching TRANSACTIONS for {account_id[:12]}...")
    url = (f"{HORIZON_URL}/accounts/{account_id}/transactions"
           f"?limit={PAGE_LIMIT}&order=asc")
    records = fetch_all_pages(url)
    print(f"  Total transactions: {len(records)}")

    if not records:
        return []

    filename = f"pi_transactions_{account_id[:8]}.csv"
    fields = [
        "id", "hash", "created_at", "source_account", "fee_charged",
        "operation_count", "memo_type", "memo", "successful",
        "ledger", "paging_token"
    ]

    with open(filename, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for r in records:
            row = {k: r.get(k, "") for k in fields}
            writer.writerow(row)

    print(f"  Saved to {filename}")
    return records


def export_operations(account_id):
    """Fetch all operations for the account."""
    print(f"\n[2/3] Fetching OPERATIONS for {account_id[:12]}...")
    url = (f"{HORIZON_URL}/accounts/{account_id}/operations"
           f"?limit={PAGE_LIMIT}&order=asc")
    records = fetch_all_pages(url)
    print(f"  Total operations: {len(records)}")

    if not records:
        return []

    filename = f"pi_operations_{account_id[:8]}.csv"
    fields = [
        "id", "type", "created_at", "transaction_hash",
        "source_account", "from", "to", "amount", "asset_type",
        "asset_code", "asset_issuer", "starting_balance",
        "funder", "account", "trustor", "trustee",
        "buying_asset_type", "buying_asset_code",
        "selling_asset_type", "selling_asset_code",
        "offer_id", "price", "paging_token"
    ]

    with open(filename, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for r in records:
            row = {k: r.get(k, "") for k in fields}
            writer.writerow(row)

    print(f"  Saved to {filename}")
    return records


def export_payments(account_id):
    """Fetch all payments (subset of operations: payments, create_account, merges)."""
    print(f"\n[3/3] Fetching PAYMENTS for {account_id[:12]}...")
    url = (f"{HORIZON_URL}/accounts/{account_id}/payments"
           f"?limit={PAGE_LIMIT}&order=asc")
    records = fetch_all_pages(url)
    print(f"  Total payments: {len(records)}")

    if not records:
        return []

    filename = f"pi_payments_{account_id[:8]}.csv"
    fields = [
        "id", "type", "created_at", "transaction_hash",
        "source_account", "from", "to", "amount",
        "asset_type", "asset_code", "asset_issuer",
        "starting_balance", "funder", "account",
        "paging_token"
    ]

    with open(filename, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for r in records:
            row = {k: r.get(k, "") for k in fields}
            writer.writerow(row)

    print(f"  Saved to {filename}")
    return records


def export_effects(account_id):
    """Fetch all effects for the account (most granular view)."""
    print(f"\n[BONUS] Fetching EFFECTS for {account_id[:12]}...")
    url = (f"{HORIZON_URL}/accounts/{account_id}/effects"
           f"?limit={PAGE_LIMIT}&order=asc")
    records = fetch_all_pages(url)
    print(f"  Total effects: {len(records)}")

    if not records:
        return []

    filename = f"pi_effects_{account_id[:8]}.csv"
    fields = [
        "id", "type", "created_at", "account", "amount",
        "asset_type", "asset_code", "asset_issuer",
        "starting_balance", "paging_token"
    ]

    with open(filename, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for r in records:
            row = {k: r.get(k, "") for k in fields}
            writer.writerow(row)

    print(f"  Saved to {filename}")
    return records


def main():
    print("=" * 60)
    print("PI NETWORK WALLET TRANSACTION EXPORTER")
    print("=" * 60)
    print(f"Account: {ACCOUNT_ID}")
    print(f"Horizon: {HORIZON_URL}")

    # Verify account exists
    print("\nVerifying account...")
    try:
        resp = session.get(f"{HORIZON_URL}/accounts/{ACCOUNT_ID}", timeout=15)
        resp.raise_for_status()
        acct = resp.json()
        balances = acct.get("balances", [])
        print(f"  Account found. Balances:")
        for b in balances:
            code = b.get("asset_code", "PI (native)")
            bal = b.get("balance", "0")
            print(f"    {code}: {bal}")
    except Exception as e:
        print(f"  ERROR: Could not verify account: {e}")
        print("  Check the account ID and your internet connection.")
        sys.exit(1)

    start = time.time()

    txns = export_transactions(ACCOUNT_ID)
    ops = export_operations(ACCOUNT_ID)
    pays = export_payments(ACCOUNT_ID)
    effs = export_effects(ACCOUNT_ID)

    elapsed = time.time() - start

    print("\n" + "=" * 60)
    print("EXPORT COMPLETE")
    print("=" * 60)
    print(f"Time elapsed: {elapsed:.1f}s")
    print(f"Transactions: {len(txns)}")
    print(f"Operations:   {len(ops)}")
    print(f"Payments:     {len(pays)}")
    print(f"Effects:      {len(effs)}")
    print(f"\nFiles saved:")
    print(f"  pi_transactions_{ACCOUNT_ID[:8]}.csv")
    print(f"  pi_operations_{ACCOUNT_ID[:8]}.csv")
    print(f"  pi_payments_{ACCOUNT_ID[:8]}.csv")
    print(f"  pi_effects_{ACCOUNT_ID[:8]}.csv")
    print("=" * 60)


if __name__ == "__main__":
    main()
