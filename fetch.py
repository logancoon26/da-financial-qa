"""
Download 10-K filings from SEC EDGAR.

Output:  data/raw/<cik>_<accession>.html   (one file per filing)
         data/filings.json                 (metadata for all fetched filings)

How EDGAR works:
  - Every public company has a CIK (Central Index Key), a unique numeric ID.
  - https://data.sec.gov/submissions/CIK<cik>.json  lists all of that company's
    filings with metadata (date, accession number, form type, etc.).
  - Each filing lives at:
      https://www.sec.gov/Archives/edgar/data/<cik>/<accession_no_dashes>/
    Inside that folder is an index page listing all the documents in the filing.
  - Parse the index to find the primary 10-K HTML document, then download it.

SEC rate limit: max 10 requests/second. We sleep 0.12s between requests (~8/s).

"""

import json
import time
import re
from pathlib import Path

import requests
from bs4 import BeautifulSoup

# ── Config ────────────────────────────────────────────────────────────────────

USER_AGENT = "logancoon26@gmail.com"

# Companies to fetch: {readable_name: CIK}
COMPANIES = {
    # ── Technology ────────────────────────────────────────────────────────────
    "Apple":                "320193",
    "Microsoft":            "789019",
    "Alphabet":             "1652044",
    "Amazon":               "1018724",
    "Meta":                 "1326801",
    "NVIDIA":               "1045810",
    "Intel":                "50863",
    "AMD":                  "2488",
    "Qualcomm":             "804328",
    "Texas Instruments":    "97476",
    "Broadcom":             "1730168",
    "Applied Materials":    "3153",
    "Salesforce":           "1108524",
    "Oracle":               "1341439",
    "SAP":                  "1016708",
    "IBM":                  "51143",
    "Cisco":                "858877",
    "Accenture":            "1467373",
    "Adobe":                "796343",
    "ServiceNow":           "1373715",
    "Snowflake":            "1640147",
    "Palantir":             "1321655",
    "CrowdStrike":          "1535527",
    "Palo Alto Networks":   "1327567",
    "Fortinet":             "1262039",
    "Workday":              "1327811",
    "Intuit":               "896878",
    "Autodesk":             "769397",
    "Synopsys":             "883241",
    "Cadence":              "813672",
 
    # ── Finance & Banking ─────────────────────────────────────────────────────
    "JPMorgan Chase":       "19617",
    "Bank of America":      "70858",
    "Wells Fargo":          "72971",
    "Citigroup":            "831001",
    "Goldman Sachs":        "886982",
    "Morgan Stanley":       "895421",
    "BlackRock":            "1364742",
    "Berkshire Hathaway":   "1067983",
    "American Express":     "4962",
    "Visa":                 "1403161",
    "Mastercard":           "1141391",
    "PayPal":               "1633917",
    "Charles Schwab":       "316709",
    "Fidelity":             "883948",
    "CME Group":            "1156375",
    "Intercontinental Exchange": "1571949",
    "Moody's":              "1059556",
    "S&P Global":           "64040",
    "Aflac":                "4977",
    "MetLife":              "1099219",
 
    # ── Healthcare & Pharma ───────────────────────────────────────────────────
    "Johnson & Johnson":    "200406",
    "UnitedHealth":         "72971",
    "Pfizer":               "78003",
    "Eli Lilly":            "59478",
    "AbbVie":               "1551152",
    "Merck":                "310158",
    "Bristol-Myers Squibb": "14272",
    "Abbott":               "1800",
    "Medtronic":            "1613103",
    "Thermo Fisher":        "97745",
    "Danaher":              "313616",
    "Becton Dickinson":     "10795",
    "Baxter":               "10456",
    "Cigna":                "1739940",
    "CVS Health":           "64803",
    "Humana":               "49071",
    "Regeneron":            "872589",
    "Vertex Pharmaceuticals":"875320",
    "Gilead Sciences":      "882095",
    "Biogen":               "875045",
 
    # ── Energy ───────────────────────────────────────────────────────────────
    "ExxonMobil":           "34088",
    "Chevron":              "93410",
    "ConocoPhillips":       "1163165",
    "EOG Resources":        "821189",
    "Pioneer Natural Resources": "1038357",
    "Schlumberger":         "87347",
    "Halliburton":          "45012",
    "Williams Companies":   "107263",
    "Kinder Morgan":        "1110805",
    "Dominion Energy":      "715072",
    "Southern Company":     "92122",
    "Exelon":               "1109357",
 
    # ── Consumer & Retail ─────────────────────────────────────────────────────
    "Walmart":              "104169",
    "Target":               "27419",
    "Costco":               "909832",
    "Home Depot":           "354950",
    "Lowe's":               "60667",
    "Nike":                 "320187",
    "Starbucks":            "829224",
    "McDonald's":           "63754",
    "Coca-Cola":            "21344",
    "PepsiCo":              "77476",
    "Procter & Gamble":     "80424",
    "Colgate-Palmolive":    "21665",
    "Mondelez":             "1418135",
    "General Mills":        "40704",
    "Estee Lauder":         "1001250",
    "Dollar General":       "34408",
    "Dollar Tree":          "935703",
 
    # ── Industrials ───────────────────────────────────────────────────────────
    "Boeing":               "12927",
    "Lockheed Martin":      "936468",
    "Raytheon":             "1047122",
    "General Electric":     "40533",
    "Honeywell":            "773840",
    "3M":                   "66740",
    "Caterpillar":          "18230",
    "Deere & Company":      "315189",
    "Emerson Electric":     "32604",
    "Parker Hannifin":      "76334",
    "Illinois Tool Works":  "49826",
    "Eaton":                "1551182",
    "Northrop Grumman":     "1133421",
    "UPS":                  "1090727",
    "Union Pacific":        "100885",
    "CSX":                  "277948",
    "Norfolk Southern":     "702165",
 
    # ── Communication & Media ─────────────────────────────────────────────────
    "AT&T":                 "732717",
    "Verizon":              "732712",
    "T-Mobile":             "1283699",
    "Comcast":              "1166691",
    "Walt Disney":          "1001039",
    "Netflix":              "1065280",
    "Charter Communications":"1091907",
    "Fox Corporation":      "1754301",
    "Warner Bros Discovery":"1437107",
 
    # ── Real Estate ───────────────────────────────────────────────────────────
    "American Tower":       "1053507",
    "Prologis":             "1045609",
    "Crown Castle":         "1051512",
    "Equinix":              "1101239",
    "Public Storage":       "1393311",
    "Simon Property Group": "1063761",
    "Welltower":            "766704",
    "CBRE Group":           "1138118",
}

FILINGS_PER_COMPANY = 5   # how many annual 10-Ks to grab per company
OUTPUT_DIR = Path("data/raw")
SLEEP = 0.12              # seconds between requests

# ── Helpers ───────────────────────────────────────────────────────────────────

HEADERS = {"User-Agent": USER_AGENT, "Accept-Encoding": "gzip, deflate"}
ARCHIVE = "https://www.sec.gov/Archives/edgar"


def get_filing_metadata(cik: str) -> list[dict]:
    """
    Fetch metadata for the most recent 10-K filings for a company.
    Returns a list of dicts with accession number, date, period, etc.
    """
    cik_padded = cik.zfill(10)
    url = f"https://data.sec.gov/submissions/CIK{cik_padded}.json"

    resp = requests.get(url, headers=HEADERS, timeout=15)
    resp.raise_for_status()
    time.sleep(SLEEP)

    data = resp.json()
    recent = data.get("filings", {}).get("recent", {})

    # All fields in 'recent' are parallel arrays — zip them together
    filings = []
    for i, form in enumerate(recent.get("form", [])):
        if form != "10-K":
            continue
        filings.append({
            "cik":         cik_padded,
            "company":     data.get("name", "Unknown"),
            "accession":   recent["accessionNumber"][i],   # e.g. "0000320193-24-000123"
            "filed":       recent["filingDate"][i],
            "period":      recent["reportDate"][i],
        })
        if len(filings) >= FILINGS_PER_COMPANY:
            break

    return filings


def find_primary_doc_url(cik: str, accession: str):
    """
    Given a filing's accession number, find the URL of the main 10-K HTML document.

    Every EDGAR filing has an index page at:
      /Archives/edgar/data/<cik>/<accession_nodashes>/<accession>-index.htm
    That page lists all documents in the filing. We parse it to find the
    primary 10-K document (not exhibits, not XBRL files).
    """
    acc_nodash = accession.replace("-", "")
    base = f"{ARCHIVE}/data/{cik}/{acc_nodash}"
    index_url = f"{base}/{accession}-index.htm"

    resp = requests.get(index_url, headers=HEADERS, timeout=15)
    time.sleep(SLEEP)

    if resp.status_code != 200:
        return None

    soup = BeautifulSoup(resp.text, "lxml")

    # The index page has a table: Seq | Description | Document | Type | Size
    # look for a row whose Type column says "10-K" or "10-K/A"
    for row in soup.select("table tr"):
        cells = row.find_all("td")
        if len(cells) < 4:
            continue
        doc_type = cells[3].get_text(strip=True)
        if doc_type in ("10-K", "10-K/A"):
            link = cells[2].find("a")
            if link:
                filename = link["href"].split("/")[-1]
                if filename.endswith((".htm", ".html")):
                    return f"{base}/{filename}"

    # Fallback: scan all links in the index for any .htm that isn't an exhibit
    for a in soup.find_all("a", href=True):
        name = a["href"].split("/")[-1].lower()
        if (name.endswith((".htm", ".html"))
                and "ex" not in name
                and "index" not in name):
            return f"{base}/{a['href'].split('/')[-1]}"

    return None


def fetch_html(url: str) -> str:
    resp = requests.get(url, headers=HEADERS, timeout=90)
    resp.raise_for_status()
    time.sleep(SLEEP)
    return resp.text


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_metadata = []

    for company_name, cik in COMPANIES.items():
        print(f"\n{company_name} (CIK {cik})")

        filings = get_filing_metadata(cik)
        print(f"  Found {len(filings)} 10-K filings")

        for filing in filings:
            cache_path = OUTPUT_DIR / f"{filing['cik']}_{filing['accession']}.html"
            filing["html_path"] = str(cache_path)

            # Skip if already downloaded
            if cache_path.exists():
                print(f"  [cached]  {filing['accession']} ({filing['period']})")
                all_metadata.append(filing)
                continue

            # Find the primary document URL
            url = find_primary_doc_url(filing["cik"], filing["accession"])
            if not url:
                print(f"  [skip]    {filing['accession']} — could not find doc URL")
                continue

            # Download and cache
            print(f"  [fetch]   {filing['accession']} ({filing['period']}) ← {url}")
            try:
                html = fetch_html(url)
                cache_path.write_text(html, encoding="utf-8")
                filing["doc_url"] = url
                all_metadata.append(filing)
            except Exception as e:
                print(f"  [error]   {e}")

    # Save metadata so later steps don't need to re-query EDGAR
    meta_path = Path("data/filings.json")
    meta_path.parent.mkdir(exist_ok=True)
    meta_path.write_text(json.dumps(all_metadata, indent=2))

    print(f"\n✓ Fetched {len(all_metadata)} filings → {meta_path}")


if __name__ == "__main__":
    main()
