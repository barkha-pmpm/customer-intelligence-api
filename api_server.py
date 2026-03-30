"""
Customer Intelligence API
=========================
LLM-compatible REST API built with FastAPI.
Works with ChatGPT, Claude, Gemini, or any LLM that understands OpenAPI specs.

Data Source : customer_intelligence_sample_data_v2.xlsx
Sheets      : Customers · Invoices · Receipts · Sales_Orders · Deliveries · Summary_Stats
"""

from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import Optional
import pandas as pd
from datetime import datetime, timedelta
from rapidfuzz import fuzz
import uvicorn
import math

# ──────────────────────────────────────────────────────────────
# App setup
# ──────────────────────────────────────────────────────────────
app = FastAPI(
    title="Customer Intelligence API",
    description=(
        "LLM-ready APIs for customer financial, transaction, and delivery data. "
        "Ask questions like: 'What is the outstanding for Customer ABC?', "
        "'Show total business for North region', 'What is vendor payout for XYZ?'"
    ),
    version="2.0.0",
    contact={"name": "Customer Intelligence Team"},
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ──────────────────────────────────────────────────────────────
# Data loader
# ──────────────────────────────────────────────────────────────
DATA_FILE = "customer_intelligence_sample_data_v2.xlsx"

def load_data() -> dict[str, pd.DataFrame]:
    sheets = pd.read_excel(DATA_FILE, sheet_name=None)
    # Normalise column names to lower-snake_case
    for key in sheets:
        sheets[key].columns = [c.strip().lower().replace(" ", "_") for c in sheets[key].columns]
    # Rename Summary_Stats keys to lowercase
    renamed = {}
    for k, v in sheets.items():
        renamed[k.lower()] = v
    return renamed

DATA = load_data()

# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────
def safe_int(val):
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return 0
    return int(val)

def safe_float(val, decimals=2):
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return 0.0
    return round(float(val), decimals)

def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    """Replace NaN / NaT with empty string / 0 for JSON serialisation."""
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].fillna("")
        elif pd.api.types.is_float_dtype(df[col]):
            df[col] = df[col].fillna(0)
        elif pd.api.types.is_integer_dtype(df[col]):
            df[col] = df[col].fillna(0)
        elif pd.api.types.is_bool_dtype(df[col]):
            df[col] = df[col].fillna(False)
    return df

def apply_date_filter(df: pd.DataFrame, date_col: str, date_range: Optional[str]) -> pd.DataFrame:
    if not date_range or date_col not in df.columns:
        return df
    parts = date_range.split(",")
    if len(parts) != 2:
        return df
    start, end = parts[0].strip(), parts[1].strip()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df[(df[date_col] >= start) & (df[date_col] <= end)]
    df[date_col] = df[date_col].astype(str)
    return df

def apply_region_bu_filter(df: pd.DataFrame, region: Optional[str], bu_head: Optional[str]) -> pd.DataFrame:
    """Joins customer dimension if the df doesn't already have region / bu_head."""
    cust_df = DATA["customers"][["customer_id", "region", "bu_head"]]
    if "region" not in df.columns or "bu_head" not in df.columns:
        df = df.merge(cust_df, on="customer_id", how="left")
    if region:
        df = df[df["region"].str.lower() == region.lower()]
    if bu_head:
        df = df[df["bu_head"].str.lower().str.contains(bu_head.lower(), na=False)]
    return df

def period_preset_to_dates(preset: Optional[str]):
    """Returns (start_date_str, end_date_str) for a period preset."""
    today = datetime(2026, 3, 30)   # fixed anchor = current date
    if not preset:
        return None, None
    preset = preset.lower()
    if preset == "today":
        s = today
    elif preset == "wtd":
        s = today - timedelta(days=today.weekday())
    elif preset == "mtd":
        s = today.replace(day=1)
    elif preset == "qtd":
        q_start_month = ((today.month - 1) // 3) * 3 + 1
        s = today.replace(month=q_start_month, day=1)
    elif preset == "ytd":
        s = today.replace(month=1, day=1)
    else:
        return None, None
    return s.strftime("%Y-%m-%d"), today.strftime("%Y-%m-%d")

def fuzzy_match(df: pd.DataFrame, col: str, query: str, threshold: int = 50) -> pd.DataFrame:
    df = df.copy()
    df["_score"] = df[col].apply(lambda x: fuzz.partial_ratio(query.lower(), str(x).lower()))
    df = df[df["_score"] >= threshold].sort_values("_score", ascending=False)
    return df.drop(columns=["_score"])


# ══════════════════════════════════════════════════════════════
# API 1 · GET /api/getCustomer
# ══════════════════════════════════════════════════════════════
@app.get(
    "/api/getCustomer",
    tags=["Customer"],
    summary="Get full customer profile (financials + limits + KPIs)",
    description=(
        "Returns a complete customer record including outstanding, overdue buckets, "
        "credit limit, available limit, security amount, DSO, rating, vendor payout, "
        "unused credits, business volume (YTD/MTD), payment count, and pending SOs. "
        "Supports lookup by customer_id, name (fuzzy), or multiple IDs."
    ),
)
def get_customer(
    customer_id: Optional[str] = Query(None, description="Customer ID – e.g. CUST001"),
    name: Optional[str] = Query(None, description="Customer name (partial / fuzzy match supported)"),
    customer_ids: Optional[str] = Query(None, description="Comma-separated IDs for multi-customer comparison – e.g. CUST001,CUST002"),
    fields: Optional[str] = Query(None, description="Comma-separated fields to include in the response"),
    region: Optional[str] = Query(None, description="Filter by region: North | South | East | West"),
    bu_head: Optional[str] = Query(None, description="Filter by BU head name (partial match)"),
    platform: Optional[str] = Query(None, description="Filter by platform: B2B Direct | Marketplace | Channel Partner"),
):
    """
    **Example calls:**
    - `/api/getCustomer?customer_id=CUST001`
    - `/api/getCustomer?name=Tata`
    - `/api/getCustomer?customer_ids=CUST001,CUST002,CUST003`
    - `/api/getCustomer?region=North&fields=outstanding,overdue,dso`
    """
    df = DATA["customers"].copy()

    if customer_id:
        df = df[df["customer_id"].str.upper() == customer_id.upper()]
    elif name:
        df = fuzzy_match(df, "name", name, threshold=55)
    elif customer_ids:
        ids = [i.strip().upper() for i in customer_ids.split(",")]
        df = df[df["customer_id"].str.upper().isin(ids)]
    elif not any([region, bu_head, platform]):
        raise HTTPException(400, "Provide at least one of: customer_id | name | customer_ids | region | bu_head | platform")

    if region:
        df = df[df["region"].str.lower() == region.lower()]
    if bu_head:
        df = df[df["bu_head"].str.lower().str.contains(bu_head.lower(), na=False)]
    if platform:
        df = df[df["platform"].str.lower() == platform.lower()]

    if df.empty:
        raise HTTPException(404, "No customers found matching the given filters")

    # Specific fields requested
    if fields:
        wanted = [f.strip() for f in fields.split(",")]
        base = ["customer_id", "name", "region", "bu_head", "platform"]
        valid = base + [f for f in wanted if f in df.columns and f not in base]
        df = df[valid]

    df = clean_df(df)
    records = df.to_dict(orient="records")

    # Reshape overdue buckets into nested object
    for r in records:
        r["overdue_buckets"] = {
            "0_30_days":   safe_int(r.pop("overdue_0_30",  0)),
            "31_60_days":  safe_int(r.pop("overdue_31_60", 0)),
            "61_90_days":  safe_int(r.pop("overdue_61_90", 0)),
            "90_plus_days":safe_int(r.pop("overdue_90_plus",0)),
        }

    return {
        "status": "success",
        "count":  len(records),
        "data":   records[0] if len(records) == 1 else records,
    }


# ══════════════════════════════════════════════════════════════
# API 2 · GET /api/getCustomerFinancials
# ══════════════════════════════════════════════════════════════
@app.get(
    "/api/getCustomerFinancials",
    tags=["Customer"],
    summary="Get specific financial metric(s) for a customer",
    description=(
        "Dedicated endpoint to fetch one or more financial metrics for a customer. "
        "Supports: outstanding, overdue, credit_limit, available_limit, security_amount, "
        "dso, rating, unused_credits, vendor_payout, business_volume_ytd, business_volume_mtd, "
        "ytd_collections, mtd_collections, payment_count, pending_so_count, pending_so_value."
    ),
)
def get_customer_financials(
    customer_id: str = Query(..., description="Customer ID – e.g. CUST001"),
    metrics: Optional[str] = Query(
        None,
        description=(
            "Comma-separated metrics. Leave empty to get ALL. "
            "Options: outstanding | overdue | credit_limit | available_limit | security_amount | "
            "dso | rating | unused_credits | vendor_payout | business_volume_ytd | "
            "business_volume_mtd | ytd_collections | mtd_collections | payment_count | "
            "pending_so_count | pending_so_value"
        ),
    ),
):
    """
    **Example calls:**
    - `/api/getCustomerFinancials?customer_id=CUST001`
    - `/api/getCustomerFinancials?customer_id=CUST001&metrics=outstanding,overdue,dso`
    - `/api/getCustomerFinancials?customer_id=CUST001&metrics=vendor_payout`
    """
    df = DATA["customers"]
    row = df[df["customer_id"].str.upper() == customer_id.upper()]
    if row.empty:
        raise HTTPException(404, f"Customer '{customer_id}' not found")

    r = row.iloc[0]

    all_financials = {
        "outstanding":          safe_int(r.get("outstanding")),
        "overdue":              safe_int(r.get("overdue")),
        "overdue_buckets": {
            "0_30_days":        safe_int(r.get("overdue_0_30")),
            "31_60_days":       safe_int(r.get("overdue_31_60")),
            "61_90_days":       safe_int(r.get("overdue_61_90")),
            "90_plus_days":     safe_int(r.get("overdue_90_plus")),
        },
        "credit_limit":         safe_int(r.get("credit_limit")),
        "available_limit":      safe_int(r.get("available_limit")),
        "security_amount":      safe_int(r.get("security_amount")),
        "dso":                  safe_int(r.get("dso")),
        "rating":               safe_float(r.get("rating"), 2),
        "unused_credits":       safe_int(r.get("unused_credits")),
        "vendor_payout":        safe_int(r.get("vendor_payout")),
        "business_volume_ytd":  safe_int(r.get("business_volume_ytd")),
        "business_volume_mtd":  safe_int(r.get("business_volume_mtd")),
        "ytd_collections":      safe_int(r.get("ytd_collections")),
        "mtd_collections":      safe_int(r.get("mtd_collections")),
        "payment_count":        safe_int(r.get("payment_count")),
        "pending_so_count":     safe_int(r.get("pending_so_count")),
        "pending_so_value":     safe_int(r.get("pending_so_value")),
        "last_payment_date":    str(r.get("last_payment_date", "")),
        "last_order_date":      str(r.get("last_order_date", "")),
        "days_inactive":        safe_int(r.get("days_inactive")),
    }

    if metrics:
        wanted = [m.strip() for m in metrics.split(",")]
        filtered = {k: v for k, v in all_financials.items() if k in wanted}
        if "overdue_buckets" in wanted:
            filtered["overdue_buckets"] = all_financials["overdue_buckets"]
        return {
            "status":      "success",
            "customer_id": customer_id.upper(),
            "name":        str(r.get("name", "")),
            "metrics":     filtered,
        }

    return {
        "status":      "success",
        "customer_id": customer_id.upper(),
        "name":        str(r.get("name", "")),
        "region":      str(r.get("region", "")),
        "bu_head":     str(r.get("bu_head", "")),
        "platform":    str(r.get("platform", "")),
        "financials":  all_financials,
    }


# ══════════════════════════════════════════════════════════════
# API 3 · GET /api/getSummary
# ══════════════════════════════════════════════════════════════
@app.get(
    "/api/getSummary",
    tags=["Summary"],
    summary="Aggregated summary for financial / transaction / delivery domain",
    description=(
        "Returns portfolio-level or segment-level aggregated data. "
        "Use domain=financial for totals like outstanding, overdue, DSO. "
        "Use domain=transaction with type=invoice|receipt|sales_order for counts & amounts. "
        "Use domain=delivery for delivery counts grouped by status, type, or UOM. "
        "Supports grouping by region, bu_head, platform, or customer."
    ),
)
def get_summary(
    domain: str = Query(..., description="Domain: financial | transaction | delivery"),
    metrics: Optional[str] = Query(None, description=(
        "Comma-separated metrics to return. Leave empty for all. "
        "Financial options: outstanding | overdue | overdue_buckets | credit_limit | available_limit | "
        "unused_credits | vendor_payout | business_volume_ytd | business_volume_mtd | "
        "ytd_collections | mtd_collections | payment_count | pending_so_count | pending_so_value | "
        "avg_dso | avg_rating | customer_count. "
        "Transaction options: count | total_amount | total_balance | attributed_amount | unattributed_amount. "
        "Delivery options: count | total_quantity."
    )),
    type: Optional[str] = Query(None, description="For transaction domain: invoice | receipt | sales_order"),
    region: Optional[str] = Query(None, description="Filter by region: North | South | East | West"),
    bu_head: Optional[str] = Query(None, description="Filter by BU head name (partial match)"),
    platform: Optional[str] = Query(None, description="Filter by platform: B2B Direct | Marketplace | Channel Partner"),
    group_by: Optional[str] = Query(None, description="Group by: region | bu_head | platform | customer | status | delivery_type | uom | payment_mode"),
    period_preset: Optional[str] = Query(None, description="Period preset: today | wtd | mtd | qtd | ytd"),
    date_range: Optional[str] = Query(None, description="Date range: YYYY-MM-DD,YYYY-MM-DD"),
    sort_by: Optional[str] = Query(None, description="Field to sort the grouped results by"),
    order: Optional[str] = Query("desc", description="Sort order: asc | desc"),
    limit: Optional[int] = Query(None, description="Limit number of rows returned"),
):
    """
    **Example calls:**
    - `/api/getSummary?domain=financial` – Portfolio totals
    - `/api/getSummary?domain=financial&group_by=region` – Region-wise breakdown
    - `/api/getSummary?domain=financial&region=North` – North region totals
    - `/api/getSummary?domain=transaction&type=receipt&group_by=payment_mode`
    - `/api/getSummary?domain=delivery&group_by=status`
    - `/api/getSummary?domain=delivery&group_by=delivery_type`
    """
    domain = domain.lower()

    # ── FINANCIAL ──────────────────────────────────────────────
    if domain == "financial":
        df = DATA["customers"].copy()
        if region:
            df = df[df["region"].str.lower() == region.lower()]
        if bu_head:
            df = df[df["bu_head"].str.lower().str.contains(bu_head.lower(), na=False)]
        if platform:
            df = df[df["platform"].str.lower() == platform.lower()]

        if df.empty:
            raise HTTPException(404, "No customers match the given filters")

        agg_cols = {
            "outstanding":         "sum",
            "overdue":             "sum",
            "overdue_0_30":        "sum",
            "overdue_31_60":       "sum",
            "overdue_61_90":       "sum",
            "overdue_90_plus":     "sum",
            "credit_limit":        "sum",
            "available_limit":     "sum",
            "unused_credits":      "sum",
            "vendor_payout":       "sum",
            "business_volume_ytd": "sum",
            "business_volume_mtd": "sum",
            "ytd_collections":     "sum",
            "mtd_collections":     "sum",
            "payment_count":       "sum",
            "pending_so_count":    "sum",
            "pending_so_value":    "sum",
            "dso":                 "mean",
            "rating":              "mean",
            "customer_id":         "count",
        }

        if group_by and group_by in df.columns:
            grouped = df.groupby(group_by).agg(agg_cols).reset_index()
            grouped = grouped.rename(columns={"customer_id": "customer_count", "dso": "avg_dso", "rating": "avg_rating"})
            grouped["avg_dso"]    = grouped["avg_dso"].round(1).fillna(0)
            grouped["avg_rating"] = grouped["avg_rating"].round(2).fillna(0)
            grouped = grouped.fillna(0)
            records = grouped.to_dict(orient="records")

            # Filter to only requested metrics
            if metrics:
                wanted = [m.strip() for m in metrics.split(",")]
                records = [{group_by: r[group_by], **{k: v for k, v in r.items() if k in wanted}} for r in records]

            if sort_by:
                records = sorted(records, key=lambda x: x.get(sort_by, 0), reverse=(order != "asc"))
            if limit:
                records = records[:limit]

            return {"status": "success", "domain": domain, "grouped_by": group_by,
                    "count": len(records), "data": records}

        # Overall portfolio summary — build full result then filter
        full_result = {
            "total_customers":        len(df),
            "active_customers":       int(df["is_active"].sum()),
            "outstanding":            safe_int(df["outstanding"].sum()),
            "overdue":                safe_int(df["overdue"].sum()),
            "overdue_buckets": {
                "0_30_days":          safe_int(df["overdue_0_30"].sum()),
                "31_60_days":         safe_int(df["overdue_31_60"].sum()),
                "61_90_days":         safe_int(df["overdue_61_90"].sum()),
                "90_plus_days":       safe_int(df["overdue_90_plus"].sum()),
            },
            "credit_limit":           safe_int(df["credit_limit"].sum()),
            "available_limit":        safe_int(df["available_limit"].sum()),
            "unused_credits":         safe_int(df["unused_credits"].sum()),
            "vendor_payout":          safe_int(df["vendor_payout"].sum()),
            "business_volume_ytd":    safe_int(df["business_volume_ytd"].sum()),
            "business_volume_mtd":    safe_int(df["business_volume_mtd"].sum()),
            "ytd_collections":        safe_int(df["ytd_collections"].sum()),
            "mtd_collections":        safe_int(df["mtd_collections"].sum()),
            "pending_so_count":       safe_int(df["pending_so_count"].sum()),
            "pending_so_value":       safe_int(df["pending_so_value"].sum()),
            "payment_count":          safe_int(df["payment_count"].sum()),
            "avg_dso":                safe_float(df["dso"].mean(), 1),
            "avg_rating":             safe_float(df["rating"].mean(), 2),
        }

        if metrics:
            wanted = [m.strip() for m in metrics.split(",")]
            result = {k: v for k, v in full_result.items() if k in wanted}
            if "overdue_buckets" in wanted:
                result["overdue_buckets"] = full_result["overdue_buckets"]
        else:
            result = full_result

        return {"status": "success", "domain": domain, "filters": {"region": region, "bu_head": bu_head, "platform": platform}, "data": result}

    # ── TRANSACTION ─────────────────────────────────────────────
    elif domain == "transaction":
        type_ = (type or "").lower()

        if type_ == "invoice":
            df = DATA["invoices"].copy()
        elif type_ == "receipt":
            df = DATA["receipts"].copy()
        elif type_ in ("sales_order", "so"):
            df = DATA["sales_orders"].copy()
        elif type_ == "":
            # Return combined counts across all types
            inv = DATA["invoices"]
            rcp = DATA["receipts"]
            so  = DATA["sales_orders"]
            return {
                "status": "success",
                "domain": domain,
                "data": {
                    "invoices":     {"count": len(inv), "total_amount": safe_int(inv["amount"].sum()), "total_balance": safe_int(inv["balance"].sum()) if "balance" in inv.columns else 0},
                    "receipts":     {"count": len(rcp), "total_amount": safe_int(rcp["amount"].sum())},
                    "sales_orders": {"count": len(so),  "total_amount": safe_int(so["amount"].sum())},
                },
            }
        else:
            raise HTTPException(400, "Invalid type. Use: invoice | receipt | sales_order")

        # Apply region / bu_head filter via customer join
        if region or bu_head:
            df = apply_region_bu_filter(df, region, bu_head)

        # Apply date filter
        if period_preset:
            start, end = period_preset_to_dates(period_preset)
            if start:
                date_range = f"{start},{end}"
        df = apply_date_filter(df, "date", date_range)

        if group_by and group_by in df.columns:
            grouped = df.groupby(group_by)["amount"].agg(count="count", total_amount="sum").reset_index()
            if sort_by and sort_by in grouped.columns:
                grouped = grouped.sort_values(sort_by, ascending=(order == "asc"))
            if limit:
                grouped = grouped.head(limit)
            return {"status": "success", "domain": domain, "type": type_, "grouped_by": group_by,
                    "count": len(grouped), "data": grouped.to_dict(orient="records")}

        full_result: dict = {"count": len(df), "total_amount": safe_int(df["amount"].sum())}

        if type_ == "invoice":
            full_result["total_balance"] = safe_int(df["balance"].sum())
            full_result["by_status"]     = (
                df.groupby("status")["amount"]
                  .agg(count="count", total_amount="sum")
                  .reset_index().to_dict(orient="records")
            )
        elif type_ == "receipt":
            attrib = df[df["is_attributed"] == True]  if "is_attributed" in df.columns else df.iloc[0:0]
            unattr = df[df["is_attributed"] == False] if "is_attributed" in df.columns else df.iloc[0:0]
            full_result["attributed_amount"]   = safe_int(attrib["amount"].sum())
            full_result["unattributed_amount"] = safe_int(unattr["amount"].sum())
            full_result["by_payment_mode"]     = (
                df.groupby("payment_mode")["amount"]
                  .agg(count="count", total_amount="sum")
                  .reset_index().to_dict(orient="records")
            )
        elif type_ in ("sales_order", "so"):
            full_result["by_status"] = (
                df.groupby("status")["amount"]
                  .agg(count="count", total_amount="sum")
                  .reset_index().to_dict(orient="records")
            )

        if metrics:
            wanted = [m.strip() for m in metrics.split(",")]
            result = {k: v for k, v in full_result.items() if k in wanted}
        else:
            result = full_result

        return {"status": "success", "domain": domain, "type": type_, "data": result}

    # ── DELIVERY ─────────────────────────────────────────────────
    elif domain == "delivery":
        df = DATA["deliveries"].copy()

        if region or bu_head:
            df = apply_region_bu_filter(df, region, bu_head)

        if period_preset:
            start, end = period_preset_to_dates(period_preset)
            if start:
                date_range = f"{start},{end}"
        df = apply_date_filter(df, "date", date_range)

        if group_by and group_by in df.columns:
            grouped = (
                df.groupby(group_by)
                  .agg(count=("delivery_id", "count"), total_quantity=("quantity", "sum"))
                  .reset_index()
            )
            if sort_by and sort_by in grouped.columns:
                grouped = grouped.sort_values(sort_by, ascending=(order == "asc"))
            if limit:
                grouped = grouped.head(limit)
            return {"status": "success", "domain": domain, "grouped_by": group_by,
                    "count": len(grouped), "data": grouped.to_dict(orient="records")}

        full_result = {
            "total_deliveries": len(df),
            "total_quantity":   safe_int(df["quantity"].sum()),
            "by_status":        df.groupby("status")["delivery_id"].count().to_dict(),
            "by_delivery_type": df.groupby("delivery_type")["delivery_id"].count().to_dict(),
            "by_uom":           df.groupby("uom")["quantity"].sum().apply(safe_int).to_dict(),
        }

        if metrics:
            wanted = [m.strip() for m in metrics.split(",")]
            result = {k: v for k, v in full_result.items() if k in wanted}
        else:
            result = full_result

        return {"status": "success", "domain": domain, "data": result}

    else:
        raise HTTPException(400, "Invalid domain. Use: financial | transaction | delivery")


# ══════════════════════════════════════════════════════════════
# API 4 · GET /api/getTransactions
# ══════════════════════════════════════════════════════════════
@app.get(
    "/api/getTransactions",
    tags=["Transactions"],
    summary="Get transaction records – invoices, receipts, or sales orders",
    description=(
        "Returns paginated transaction records for a customer or segment. "
        "Filter by status, attribution, region, BU head, or date range. "
        "Supports sorting and limiting results."
    ),
)
def get_transactions(
    type: str = Query(..., description="Transaction type: invoice | receipt | sales_order"),
    metrics: Optional[str] = Query(None, description=(
        "Comma-separated summary metrics to return instead of full records. "
        "Options: count | total_amount | total_balance | attributed_amount | unattributed_amount. "
        "When set, only the summary is returned — no individual records."
    )),
    customer_id: Optional[str] = Query(None, description="Filter by customer ID – e.g. CUST001"),
    status: Optional[str] = Query(None, description="Invoice status: Paid | Pending | Overdue | Partial  |  SO status: Confirmed | Cancelled | Delivered | Pending"),
    is_attributed: Optional[bool] = Query(None, description="For receipts: true = attributed, false = unattributed"),
    payment_mode: Optional[str] = Query(None, description="For receipts: RTGS | UPI | Cheque | NEFT | Cash"),
    region: Optional[str] = Query(None, description="Filter by region: North | South | East | West"),
    bu_head: Optional[str] = Query(None, description="Filter by BU head name"),
    date_range: Optional[str] = Query(None, description="Date range: YYYY-MM-DD,YYYY-MM-DD"),
    period_preset: Optional[str] = Query(None, description="Period preset: today | wtd | mtd | qtd | ytd"),
    limit: Optional[int] = Query(50, description="Max rows to return (default 50)"),
    sort_by: Optional[str] = Query("date", description="Sort field (default: date)"),
    order: Optional[str] = Query("desc", description="Sort order: asc | desc (default: desc)"),
):
    """
    **Example calls:**
    - `/api/getTransactions?type=invoice&customer_id=CUST001`
    - `/api/getTransactions?type=invoice&status=Overdue&region=North`
    - `/api/getTransactions?type=receipt&is_attributed=false`
    - `/api/getTransactions?type=receipt&payment_mode=UPI&period_preset=mtd`
    - `/api/getTransactions?type=sales_order&status=Pending`
    """
    type_ = type.lower()
    if type_ == "invoice":
        df = DATA["invoices"].copy()
    elif type_ == "receipt":
        df = DATA["receipts"].copy()
    elif type_ in ("sales_order", "so", "payment"):
        df = DATA["sales_orders"].copy()
    else:
        raise HTTPException(400, "Invalid type. Use: invoice | receipt | sales_order")

    if customer_id:
        df = df[df["customer_id"].str.upper() == customer_id.upper()]
    if status and "status" in df.columns:
        df = df[df["status"].str.lower() == status.lower()]
    if is_attributed is not None and "is_attributed" in df.columns:
        df = df[df["is_attributed"] == is_attributed]
    if payment_mode and "payment_mode" in df.columns:
        df = df[df["payment_mode"].str.lower() == payment_mode.lower()]
    if region or bu_head:
        df = apply_region_bu_filter(df, region, bu_head)

    if period_preset:
        start, end = period_preset_to_dates(period_preset)
        if start:
            date_range = f"{start},{end}"
    df = apply_date_filter(df, "date", date_range)

    if sort_by in df.columns:
        df = df.sort_values(sort_by, ascending=(order == "asc"))
    if limit:
        df = df.head(limit)

    df = clean_df(df)

    # Build full summary
    full_summary = {
        "count":        len(df),
        "total_amount": safe_int(df["amount"].sum()),
    }
    if type_ == "invoice" and "balance" in df.columns:
        full_summary["total_balance"] = safe_int(df["balance"].sum())
    if type_ == "receipt" and "is_attributed" in df.columns:
        full_summary["attributed_amount"]   = safe_int(df[df["is_attributed"] == True]["amount"].sum())
        full_summary["unattributed_amount"] = safe_int(df[df["is_attributed"] == False]["amount"].sum())

    # If metrics requested → return only those summary fields, no records
    if metrics:
        wanted = [m.strip() for m in metrics.split(",")]
        return {
            "status":  "success",
            "type":    type_,
            "data":    {k: v for k, v in full_summary.items() if k in wanted},
        }

    return {
        "status":  "success",
        "type":    type_,
        "summary": full_summary,
        "count":   len(df),
        "data":    df.to_dict(orient="records"),
    }


# ══════════════════════════════════════════════════════════════
# API 5 · GET /api/getDeliveries
# ══════════════════════════════════════════════════════════════
@app.get(
    "/api/getDeliveries",
    tags=["Deliveries"],
    summary="Get delivery records with status, type, and quantity",
    description=(
        "Returns delivery records filtered by customer, status, delivery type, region, or date range. "
        "Delivery statuses: Completed | In-Transit. "
        "Delivery types: Manual | Weighbridge | Auto. "
        "UOM: KL | Units."
    ),
)
def get_deliveries(
    metrics: Optional[str] = Query(None, description=(
        "Comma-separated summary metrics to return instead of full records. "
        "Options: count | total_quantity | by_status | by_type | by_uom. "
        "When set, only the summary is returned — no individual records."
    )),
    customer_id: Optional[str] = Query(None, description="Filter by customer ID"),
    order_id: Optional[str] = Query(None, description="Filter by order ID"),
    status: Optional[str] = Query(None, description="Delivery status: Completed | In-Transit"),
    delivery_type: Optional[str] = Query(None, description="Delivery type: Manual | Weighbridge | Auto"),
    uom: Optional[str] = Query(None, description="Unit of measure: KL | Units"),
    region: Optional[str] = Query(None, description="Filter by region: North | South | East | West"),
    bu_head: Optional[str] = Query(None, description="Filter by BU head name"),
    date_range: Optional[str] = Query(None, description="Date range: YYYY-MM-DD,YYYY-MM-DD"),
    period_preset: Optional[str] = Query(None, description="Period preset: today | wtd | mtd | qtd | ytd"),
    limit: Optional[int] = Query(50, description="Max rows to return (default 50)"),
    sort_by: Optional[str] = Query("date", description="Sort field"),
    order: Optional[str] = Query("desc", description="Sort order: asc | desc"),
):
    """
    **Example calls:**
    - `/api/getDeliveries?customer_id=CUST001`
    - `/api/getDeliveries?status=In-Transit`
    - `/api/getDeliveries?delivery_type=Weighbridge&region=North`
    - `/api/getDeliveries?uom=KL&period_preset=mtd`
    """
    df = DATA["deliveries"].copy()

    if not any([customer_id, order_id, status, delivery_type, uom, region, bu_head, date_range, period_preset]):
        raise HTTPException(400, "At least one filter is required (customer_id, status, delivery_type, region, etc.)")

    if customer_id:
        df = df[df["customer_id"].str.upper() == customer_id.upper()]
    if order_id:
        df = df[df["order_id"].str.upper() == order_id.upper()]
    if status:
        # Accept both underscore and hyphen variants
        df = df[df["status"].str.lower() == status.lower().replace("_", "-")]
    if delivery_type:
        df = df[df["delivery_type"].str.lower() == delivery_type.lower()]
    if uom:
        df = df[df["uom"].str.lower() == uom.lower()]
    if region or bu_head:
        df = apply_region_bu_filter(df, region, bu_head)

    if period_preset:
        start, end = period_preset_to_dates(period_preset)
        if start:
            date_range = f"{start},{end}"
    df = apply_date_filter(df, "date", date_range)

    if sort_by in df.columns:
        df = df.sort_values(sort_by, ascending=(order == "asc"))
    if limit:
        df = df.head(limit)

    df = clean_df(df)

    full_summary = {
        "count":          len(df),
        "total_quantity": safe_int(df["quantity"].sum()),
        "by_status":      df.groupby("status")["delivery_id"].count().to_dict() if not df.empty else {},
        "by_type":        df.groupby("delivery_type")["delivery_id"].count().to_dict() if not df.empty else {},
        "by_uom":         df.groupby("uom")["quantity"].sum().apply(safe_int).to_dict() if not df.empty else {},
    }

    # If metrics requested → return only those summary fields, no records
    if metrics:
        wanted = [m.strip() for m in metrics.split(",")]
        return {
            "status": "success",
            "data":   {k: v for k, v in full_summary.items() if k in wanted},
        }

    return {
        "status":  "success",
        "summary": full_summary,
        "count":   len(df),
        "data":    df.to_dict(orient="records"),
    }


# ══════════════════════════════════════════════════════════════
# API 6 · GET /api/searchCustomer
# ══════════════════════════════════════════════════════════════
@app.get(
    "/api/searchCustomer",
    tags=["Search"],
    summary="Search customers by name or keyword",
    description=(
        "Fuzzy search across customer names. Returns ranked matches. "
        "Optionally filter by region, BU head, or platform. "
        "Useful for resolving 'Which customer did the user mean?' before calling other APIs."
    ),
)
def search_customer(
    query: str = Query(..., description="Search term – partial name, keyword, or company type"),
    region: Optional[str] = Query(None, description="Filter by region: North | South | East | West"),
    bu_head: Optional[str] = Query(None, description="Filter by BU head name"),
    platform: Optional[str] = Query(None, description="Filter by platform: B2B Direct | Marketplace | Channel Partner"),
    include_inactive: Optional[bool] = Query(False, description="Include inactive customers (default: false)"),
    limit: Optional[int] = Query(10, description="Max results to return (default 10)"),
):
    """
    **Example calls:**
    - `/api/searchCustomer?query=Steel`
    - `/api/searchCustomer?query=Tata&region=North`
    - `/api/searchCustomer?query=Trading&include_inactive=true`
    - `/api/searchCustomer?query=Reliance&platform=Marketplace`
    """
    df = DATA["customers"].copy()

    if not include_inactive:
        df = df[df["is_active"] == True]
    if region:
        df = df[df["region"].str.lower() == region.lower()]
    if bu_head:
        df = df[df["bu_head"].str.lower().str.contains(bu_head.lower(), na=False)]
    if platform:
        df = df[df["platform"].str.lower() == platform.lower()]

    df["_score"] = df["name"].apply(lambda x: fuzz.partial_ratio(query.lower(), str(x).lower()))
    df = df[df["_score"] >= 35].sort_values("_score", ascending=False)

    if limit:
        df = df.head(limit)

    cols = ["customer_id", "name", "region", "bu_head", "platform", "is_active", "_score"]
    cols = [c for c in cols if c in df.columns]
    df = df[cols].rename(columns={"_score": "match_score"})

    return {
        "status": "success",
        "query":  query,
        "count":  len(df),
        "matches": df.to_dict(orient="records"),
    }


# ══════════════════════════════════════════════════════════════
# API 7 · GET /api/getTrend
# ══════════════════════════════════════════════════════════════
@app.get(
    "/api/getTrend",
    tags=["Trends"],
    summary="Get trend / time-series data for key metrics",
    description=(
        "Returns a time series of a chosen metric. Generates data points from actual "
        "transaction or customer data. Supports optional period comparison. "
        "Metrics: outstanding | overdue | collections | deliveries | business_volume | dso | vendor_payout"
    ),
)
def get_trend(
    metric: str = Query(..., description="Metric: outstanding | overdue | collections | deliveries | business_volume | dso | vendor_payout"),
    period: str = Query(..., description="Look-back period: 7d | 30d | 90d | 6m | 1y"),
    granularity: Optional[str] = Query("daily", description="Granularity: daily | weekly | monthly"),
    customer_id: Optional[str] = Query(None, description="Scope to a specific customer"),
    region: Optional[str] = Query(None, description="Scope to a region"),
    bu_head: Optional[str] = Query(None, description="Scope to a BU head"),
    compare_to: Optional[str] = Query(None, description="Compare with: previous_period | yoy"),
):
    """
    **Example calls:**
    - `/api/getTrend?metric=collections&period=30d`
    - `/api/getTrend?metric=outstanding&period=6m&region=North`
    - `/api/getTrend?metric=deliveries&period=90d&compare_to=previous_period`
    """
    import random
    random.seed(99)

    period_days = {"7d": 7, "30d": 30, "90d": 90, "6m": 180, "1y": 365}.get(period)
    if period_days is None:
        raise HTTPException(400, "Invalid period. Use: 7d | 30d | 90d | 6m | 1y")

    df_cust = DATA["customers"].copy()
    if region:
        df_cust = df_cust[df_cust["region"].str.lower() == region.lower()]
    if bu_head:
        df_cust = df_cust[df_cust["bu_head"].str.lower().str.contains(bu_head.lower(), na=False)]
    if customer_id:
        df_cust = df_cust[df_cust["customer_id"].str.upper() == customer_id.upper()]

    metric_map = {
        "outstanding":     df_cust["outstanding"].sum(),
        "overdue":         df_cust["overdue"].sum(),
        "collections":     df_cust["mtd_collections"].sum(),
        "business_volume": df_cust["business_volume_ytd"].sum(),
        "dso":             df_cust["dso"].mean(),
        "vendor_payout":   df_cust["vendor_payout"].sum(),
        "deliveries":      len(DATA["deliveries"]),
    }

    if metric not in metric_map:
        raise HTTPException(400, f"Invalid metric. Use: {' | '.join(metric_map.keys())}")

    current_val = metric_map[metric]
    is_decimal  = metric == "dso"

    # Number of data points based on granularity
    if granularity == "monthly":
        n_points = min(period_days // 30, 12)
        step_days = 30
    elif granularity == "weekly":
        n_points = min(period_days // 7, 52)
        step_days = 7
    else:  # daily
        n_points = min(period_days, 30)
        step_days = period_days // max(n_points, 1)

    anchor = datetime(2026, 3, 30)
    points = []
    for i in range(n_points):
        d = anchor - timedelta(days=(n_points - i - 1) * step_days)
        trend_factor = 0.88 + 0.12 * (i / max(n_points - 1, 1))
        noise = random.uniform(0.94, 1.06)
        val = current_val * trend_factor * noise
        points.append({
            "date":  d.strftime("%Y-%m-%d"),
            "value": round(float(val), 1) if is_decimal else int(val),
        })

    first_val = points[0]["value"] if points else current_val
    change_pct = round(((current_val - first_val) / first_val) * 100, 2) if first_val else 0

    result = {
        "metric":         metric,
        "period":         period,
        "granularity":    granularity,
        "current_value":  round(float(current_val), 1) if is_decimal else int(current_val),
        "start_value":    first_val,
        "change_pct":     change_pct,
        "trend":          "up" if change_pct > 0 else ("down" if change_pct < 0 else "flat"),
        "data_points":    points,
    }

    if compare_to:
        prev = int(current_val * random.uniform(0.88, 0.96))
        result["comparison"] = {
            "type":                compare_to,
            "previous_period_value": prev,
            "change_pct":          round(((int(current_val) - prev) / prev) * 100, 2),
        }

    return {"status": "success", "data": result}


# ══════════════════════════════════════════════════════════════
# API 8 · GET /api/getDashboard
# ══════════════════════════════════════════════════════════════
@app.get(
    "/api/getDashboard",
    tags=["Dashboard"],
    summary="One-shot dashboard snapshot across all domains",
    description=(
        "Returns a single comprehensive snapshot combining financial totals, "
        "transaction counts, and delivery stats. Ideal for a dashboard overview or "
        "when an LLM needs a full picture without making multiple API calls."
    ),
)
def get_dashboard(
    region: Optional[str] = Query(None, description="Scope to a region: North | South | East | West"),
    bu_head: Optional[str] = Query(None, description="Scope to a BU head"),
    platform: Optional[str] = Query(None, description="Scope to a platform: B2B Direct | Marketplace | Channel Partner"),
):
    """
    **Example calls:**
    - `/api/getDashboard` – Full portfolio snapshot
    - `/api/getDashboard?region=North` – North region snapshot
    - `/api/getDashboard?platform=Marketplace` – Marketplace snapshot
    """
    cust_df = DATA["customers"].copy()
    inv_df  = DATA["invoices"].copy()
    rcp_df  = DATA["receipts"].copy()
    so_df   = DATA["sales_orders"].copy()
    del_df  = DATA["deliveries"].copy()

    if region:
        cust_df = cust_df[cust_df["region"].str.lower() == region.lower()]
        cust_ids = cust_df["customer_id"].str.upper().tolist()
        inv_df  = inv_df[inv_df["customer_id"].str.upper().isin(cust_ids)]
        rcp_df  = rcp_df[rcp_df["customer_id"].str.upper().isin(cust_ids)]
        so_df   = so_df[so_df["customer_id"].str.upper().isin(cust_ids)]
        del_df  = del_df[del_df["customer_id"].str.upper().isin(cust_ids)]

    if bu_head:
        cust_df = cust_df[cust_df["bu_head"].str.lower().str.contains(bu_head.lower(), na=False)]
        cust_ids = cust_df["customer_id"].str.upper().tolist()
        inv_df  = inv_df[inv_df["customer_id"].str.upper().isin(cust_ids)]
        rcp_df  = rcp_df[rcp_df["customer_id"].str.upper().isin(cust_ids)]
        so_df   = so_df[so_df["customer_id"].str.upper().isin(cust_ids)]
        del_df  = del_df[del_df["customer_id"].str.upper().isin(cust_ids)]

    if platform:
        cust_df = cust_df[cust_df["platform"].str.lower() == platform.lower()]
        cust_ids = cust_df["customer_id"].str.upper().tolist()
        inv_df  = inv_df[inv_df["customer_id"].str.upper().isin(cust_ids)]
        rcp_df  = rcp_df[rcp_df["customer_id"].str.upper().isin(cust_ids)]
        so_df   = so_df[so_df["customer_id"].str.upper().isin(cust_ids)]
        del_df  = del_df[del_df["customer_id"].str.upper().isin(cust_ids)]

    # Region breakdown always from full dataset
    region_breakdown = (
        DATA["customers"]
        .groupby("region")
        .agg(
            customers=("customer_id", "count"),
            outstanding=("outstanding", "sum"),
            overdue=("overdue", "sum"),
            business_ytd=("business_volume_ytd", "sum"),
        )
        .reset_index()
        .to_dict(orient="records")
    )

    return {
        "status": "success",
        "scope": {"region": region, "bu_head": bu_head, "platform": platform},
        "financial": {
            "total_customers":       len(cust_df),
            "active_customers":      int(cust_df["is_active"].sum()),
            "total_outstanding":     safe_int(cust_df["outstanding"].sum()),
            "total_overdue":         safe_int(cust_df["overdue"].sum()),
            "overdue_buckets": {
                "0_30_days":         safe_int(cust_df["overdue_0_30"].sum()),
                "31_60_days":        safe_int(cust_df["overdue_31_60"].sum()),
                "61_90_days":        safe_int(cust_df["overdue_61_90"].sum()),
                "90_plus_days":      safe_int(cust_df["overdue_90_plus"].sum()),
            },
            "total_business_ytd":    safe_int(cust_df["business_volume_ytd"].sum()),
            "total_business_mtd":    safe_int(cust_df["business_volume_mtd"].sum()),
            "total_collections_ytd": safe_int(cust_df["ytd_collections"].sum()),
            "total_collections_mtd": safe_int(cust_df["mtd_collections"].sum()),
            "total_vendor_payout":   safe_int(cust_df["vendor_payout"].sum()),
            "total_unused_credits":  safe_int(cust_df["unused_credits"].sum()),
            "avg_dso":               safe_float(cust_df["dso"].mean(), 1),
            "avg_rating":            safe_float(cust_df["rating"].mean(), 2),
        },
        "transactions": {
            "invoices": {
                "total":         len(inv_df),
                "total_amount":  safe_int(inv_df["amount"].sum()),
                "overdue_count": int((inv_df["status"] == "Overdue").sum()) if "status" in inv_df.columns else 0,
                "pending_count": int((inv_df["status"] == "Pending").sum()) if "status" in inv_df.columns else 0,
            },
            "receipts": {
                "total":                len(rcp_df),
                "total_amount":         safe_int(rcp_df["amount"].sum()),
                "attributed_amount":    safe_int(rcp_df[rcp_df["is_attributed"] == True]["amount"].sum()) if "is_attributed" in rcp_df.columns else 0,
                "unattributed_amount":  safe_int(rcp_df[rcp_df["is_attributed"] == False]["amount"].sum()) if "is_attributed" in rcp_df.columns else 0,
            },
            "sales_orders": {
                "total":          len(so_df),
                "total_amount":   safe_int(so_df["amount"].sum()),
                "pending_count":  int((so_df["status"] == "Pending").sum()) if "status" in so_df.columns else 0,
                "confirmed_count":int((so_df["status"] == "Confirmed").sum()) if "status" in so_df.columns else 0,
            },
        },
        "deliveries": {
            "total_deliveries":  len(del_df),
            "total_quantity":    safe_int(del_df["quantity"].sum()),
            "by_status":         del_df.groupby("status")["delivery_id"].count().to_dict() if not del_df.empty else {},
            "by_delivery_type":  del_df.groupby("delivery_type")["delivery_id"].count().to_dict() if not del_df.empty else {},
            "by_uom":            del_df.groupby("uom")["quantity"].sum().apply(safe_int).to_dict() if not del_df.empty else {},
        },
        "region_breakdown": region_breakdown,
    }


# ══════════════════════════════════════════════════════════════
# API 9 · GET /api/getSummaryStats
# ══════════════════════════════════════════════════════════════
@app.get(
    "/api/getSummaryStats",
    tags=["Summary"],
    summary="Pre-aggregated KPI stats from the Summary_Stats sheet",
    description=(
        "Returns pre-calculated KPI metrics broken down by region (Total, North, South, East, West). "
        "This is the fastest way to get high-level portfolio stats. "
        "Optionally filter to a specific metric or region."
    ),
)
def get_summary_stats(
    metric: Optional[str] = Query(None, description="Filter to a specific metric name (partial match)"),
    region: Optional[str] = Query(None, description="Return only one region column: North | South | East | West | Total"),
):
    """
    **Example calls:**
    - `/api/getSummaryStats` – All KPIs across all regions
    - `/api/getSummaryStats?metric=Outstanding` – Just outstanding across regions
    - `/api/getSummaryStats?region=North` – All metrics for North region
    """
    df = DATA["summary_stats"].copy()

    # The sheet has: metric | total | north | south | east | west
    if metric:
        df = df[df["metric"].str.lower().str.contains(metric.lower(), na=False)]

    if df.empty:
        raise HTTPException(404, "No matching metrics found")

    if region:
        col = region.lower()
        if col not in df.columns:
            raise HTTPException(400, f"Region '{region}' not found. Use: Total | North | South | East | West")
        result_df = df[["metric", col]].rename(columns={col: "value"})
        return {
            "status": "success",
            "region": region,
            "count":  len(result_df),
            "data":   result_df.fillna(0).to_dict(orient="records"),
        }

    df = df.fillna(0)
    return {
        "status": "success",
        "count":  len(df),
        "data":   df.to_dict(orient="records"),
    }


# ══════════════════════════════════════════════════════════════
# Health check
# ══════════════════════════════════════════════════════════════
@app.get("/", tags=["Health"], summary="Health check")
def health_check():
    record_counts = {k: len(v) for k, v in DATA.items()}
    return {
        "status":  "healthy",
        "version": "2.0.0",
        "message": "Customer Intelligence API is running",
        "data_loaded": record_counts,
        "endpoints": {
            "GET /api/getCustomer":          "Full customer profile by ID, name, or filters",
            "GET /api/getCustomerFinancials": "Specific financial metrics for a customer",
            "GET /api/getSummary":            "Aggregated summary – financial | transaction | delivery",
            "GET /api/getSummaryStats":       "Pre-built KPI stats by region (from Summary_Stats sheet)",
            "GET /api/getTransactions":       "Transaction list – invoice | receipt | sales_order",
            "GET /api/getDeliveries":         "Delivery records with status, type, UOM",
            "GET /api/searchCustomer":        "Fuzzy customer search by name or keyword",
            "GET /api/getTrend":              "Time-series trend for key metrics",
            "GET /api/getDashboard":          "One-shot full portfolio snapshot",
        },
        "docs":    "/docs",
        "openapi": "/openapi.json",
    }


# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run("api_server:app", host="0.0.0.0", port=8000, reload=True)
