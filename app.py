from pathlib import Path
import re
import smtplib
from email.message import EmailMessage
from io import BytesIO
from datetime import datetime
import numpy as np
import pandas as pd
import folium
import requests
import streamlit as st
from streamlit_folium import st_folium
from geopy.geocoders import Nominatim
from streamlit_geolocation import streamlit_geolocation
from PIL import Image
from ultralytics import YOLO


# =========================================================
# CONFIG
# =========================================================

MODEL_PATH = Path("models/road_damage_yolo11s.pt")
PWD_EMAIL = "mr.jrjetson@gmail.com"

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle


def create_pdf_report(report_df, road_id, location, latitude, longitude):
    """Create a professional RoadGuardAI assessment PDF in memory."""
    buffer = BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=landscape(A4),
        rightMargin=22,
        leftMargin=22,
        topMargin=22,
        bottomMargin=22,
    )

    styles = getSampleStyleSheet()
    title = ParagraphStyle(
        "RGTitle",
        parent=styles["Title"],
        fontSize=22,
        leading=26,
        alignment=TA_CENTER,
        spaceAfter=4,
    )
    subtitle = ParagraphStyle(
        "RGSubtitle",
        parent=styles["Normal"],
        fontSize=9,
        leading=12,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#555555"),
        spaceAfter=14,
    )
    heading = ParagraphStyle(
        "RGHeading",
        parent=styles["Heading2"],
        fontSize=13,
        leading=16,
        spaceBefore=8,
        spaceAfter=8,
    )
    cell = ParagraphStyle(
        "RGCell",
        parent=styles["Normal"],
        fontSize=7,
        leading=8.5,
    )

    story = [
        Paragraph("RoadGuardAI", title),
        Paragraph(
            "AI-Powered Road Damage Detection & Maintenance Assessment",
            subtitle,
        ),
    ]

    # Summary values
    if len(report_df) > 0 and "Risk score" in report_df.columns:
        risk_values = pd.to_numeric(
            report_df["Risk score"], errors="coerce"
        ).fillna(0)
        overall_risk = float(risk_values.max())
    else:
        overall_risk = 0.0

    if overall_risk >= 70:
        overall_severity = "Critical"
    elif overall_risk >= 40:
        overall_severity = "Moderate"
    else:
        overall_severity = "Minor"

    # Wrap long location text inside the table cell.
    # The Location value gets extra width so addresses/landmarks stay
    # inside the box instead of overflowing the PDF page.
    summary = Table(
        [
            [
                Paragraph("Road / Ward ID", cell),
                Paragraph(str(road_id or "Not provided"), cell),
                Paragraph("Location", cell),
                Paragraph(str(location or "Not provided"), cell),
            ],
            [
                Paragraph("GPS Latitude", cell),
                Paragraph(f"{float(latitude):.6f}", cell),
                Paragraph("GPS Longitude", cell),
                Paragraph(f"{float(longitude):.6f}", cell),
            ],
            [
                Paragraph("Damage Instances", cell),
                Paragraph(str(len(report_df)), cell),
                Paragraph("Overall Risk", cell),
                Paragraph(
                    f"{overall_risk:.0f}/100 — {overall_severity}",
                    cell,
                ),
            ],
        ],
        # Slightly wider Location value box; Paragraph automatically wraps
        # long addresses/landmarks onto multiple lines.
        colWidths=[95, 135, 95, 245],
    )
    summary.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#EAF0F6")),
        ("BACKGROUND", (2, 0), (2, -1), colors.HexColor("#EAF0F6")),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTNAME", (2, 0), (2, -1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#B8C2CC")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
    ]))
    story += [summary, Spacer(1, 12)]

    story.append(Paragraph("Damage Detection Details", heading))

    desired_columns = [
        "Damage type",
        "Confidence",
        "Area %",
        "Risk score",
        "Severity",
        "Priority",
        "Repair duration",
        "Estimated cost",
    ]
    available = [c for c in desired_columns if c in report_df.columns]

    header = [Paragraph(str(c), cell) for c in available]
    data = [header]

    for _, row in report_df.iterrows():
        data.append([
            Paragraph(str(row.get(c, "—")), cell)
            for c in available
        ])

    # Keep the report readable even if all columns are present.
    if available:
        widths = {
            "Damage type": 92,
            "Confidence": 58,
            "Area %": 48,
            "Risk score": 58,
            "Severity": 62,
            "Priority": 98,
            "Repair duration": 92,
            "Estimated cost": 95,
        }
        col_widths = [widths.get(c, 70) for c in available]
        details = Table(data, colWidths=col_widths, repeatRows=1)
        details.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#263746")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#B8C2CC")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1),
             [colors.white, colors.HexColor("#F5F7F9")]),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ]))
        story += [details, Spacer(1, 12)]

    # Recommended actions
    if "Recommended action" in report_df.columns:
        story.append(Paragraph("Recommended Actions", heading))
        actions = [
            [
                Paragraph("Damage Type", cell),
                Paragraph("Recommended Action", cell),
            ]
        ]
        for _, row in report_df.iterrows():
            actions.append([
                Paragraph(str(row.get("Damage type", "—")), cell),
                Paragraph(str(row.get("Recommended action", "—")), cell),
            ])

        action_table = Table(
            actions,
            colWidths=[150, 450],
            repeatRows=1,
        )
        action_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#263746")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#B8C2CC")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1),
             [colors.white, colors.HexColor("#F5F7F9")]),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ]))
        story.append(action_table)

    story += [
        Spacer(1, 14),
        Paragraph(
            "Generated by RoadGuardAI • YOLO11s road-damage detection • "
            "GPS/location-aware assessment",
            subtitle,
        ),
    ]

    doc.build(story)
    buffer.seek(0)
    return buffer



st.set_page_config(
    page_title="RoadGuardAI",
    page_icon="🛣️",
    layout="wide",
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    """
    <style>
        /* =====================================================
           ROADGUARDAI — PREMIUM RED / BLACK COMMAND CENTER
           ===================================================== */
        :root {
            --rg-red: #e5092f;
            --rg-red2: #ff3b57;
            --rg-deep: #050608;
            --rg-panel: #0b0d12;
            --rg-panel2: #101217;
            --rg-text: #f7f8fb;
            --rg-muted: #969ba6;
            --rg-green: #22d477;
        }

        .stApp {
            background:
                radial-gradient(circle at 91% 7%, rgba(229,9,47,.14), transparent 25%),
                radial-gradient(circle at 4% 92%, rgba(229,9,47,.09), transparent 23%),
                linear-gradient(135deg, #030405 0%, #07090d 54%, #040507 100%);
        }

        [data-testid="stHeader"] {
            background: rgba(3,4,6,.90);
            border-bottom: 1px solid rgba(229,9,47,.38);
        }

        [data-testid="stSidebar"] {
            background: linear-gradient(180deg,#090b10 0%,#050609 100%);
            border-right: 1px solid rgba(229,9,47,.52);
            box-shadow: 10px 0 38px rgba(0,0,0,.45);
        }
        [data-testid="stSidebar"] > div:first-child { background: transparent; }
        [data-testid="stSidebarContent"] { padding-top: 1.4rem; }

        h1,h2,h3,h4 { color: var(--rg-text) !important; font-weight: 850 !important; }
        h1 { letter-spacing: -1.5px; }
        h2,h3 { letter-spacing: -.5px; }
        p, label, [data-testid="stMarkdownContainer"] { color: #d7d9df; }
        .stCaption { color: #8e939e !important; }

        /* premium hero */
        .rg-hero {
            position: relative;
            overflow: hidden;
            min-height: 154px;
            padding: 27px 30px 25px;
            margin: 2px 0 20px;
            border: 1px solid rgba(229,9,47,.72);
            border-radius: 18px;
            background:
                linear-gradient(110deg, rgba(229,9,47,.18) 0%, rgba(229,9,47,.04) 30%, rgba(5,8,12,.88) 62%),
                radial-gradient(circle at 83% 45%, rgba(229,9,47,.16), transparent 25%),
                linear-gradient(135deg,#0e1117,#06080c 70%);
            box-shadow: 0 12px 38px rgba(0,0,0,.42), 0 0 34px rgba(229,9,47,.09);
        }
        .rg-hero:before {
            content:""; position:absolute; left:-5%; bottom:-55px; width:72%; height:110px;
            background: linear-gradient(165deg, transparent 0 42%, rgba(229,9,47,.18) 43% 48%, transparent 49%),
                        linear-gradient(165deg, transparent 0 58%, rgba(229,9,47,.08) 59% 64%, transparent 65%);
            transform: skewX(-25deg);
        }
        .rg-hero:after {
            content:""; position:absolute; right:-90px; top:-75px; width:360px; height:240px;
            border-left: 2px solid rgba(229,9,47,.50);
            border-bottom: 2px solid rgba(229,9,47,.22);
            transform: rotate(-24deg);
        }
        .rg-brand { position:relative; z-index:2; display:flex; align-items:center; gap:18px; }
        .rg-logo {
            width:64px; height:64px; border-radius:15px; display:flex; align-items:center; justify-content:center;
            font-size:38px; background:linear-gradient(145deg,#ff1a38,#9f071d);
            box-shadow:0 0 28px rgba(229,9,47,.32); border:1px solid rgba(255,111,127,.7);
        }
        .rg-title { font-size:43px; line-height:1; font-weight:900; color:#fff; letter-spacing:-2px; }
        .rg-title span { color:#ff1736; text-shadow:0 0 20px rgba(255,23,54,.38); }
        .rg-subtitle { margin-top:11px; font-size:14px; color:#9da2ad; }
        .rg-tagline { position:absolute; right:30px; top:43px; z-index:2; text-align:right; font-size:16px; font-style:italic; color:#ff4a5d; }
        .rg-tagline b { display:block; color:#fff; font-style:normal; font-size:18px; margin-top:5px; }
        .rg-line { width:58px; height:3px; background:#e5092f; margin:13px 0 0 auto; border-radius:3px; box-shadow:0 0 14px rgba(229,9,47,.7); }

        /* metrics */
        [data-testid="stMetric"] {
            position:relative; overflow:hidden; min-height:128px;
            background:linear-gradient(145deg,rgba(15,17,22,.98),rgba(7,9,13,.98));
            border:1px solid rgba(229,9,47,.55); border-radius:16px; padding:18px 20px;
            box-shadow:0 9px 30px rgba(0,0,0,.36), inset 0 1px 0 rgba(255,255,255,.025);
        }
        [data-testid="stMetric"]:after {
            content:""; position:absolute; right:-28px; bottom:-38px; width:130px; height:72px;
            border-top:2px solid rgba(229,9,47,.48); transform:rotate(-25deg);
        }
        [data-testid="stMetricLabel"] { color:#c9cdd5 !important; font-size:14px !important; }
        [data-testid="stMetricValue"] { color:#f7f8fb !important; font-size:32px !important; font-weight:900 !important; }
        [data-testid="stMetricDelta"] { color:var(--rg-green) !important; }

        .stButton > button, .stDownloadButton > button {
            background:linear-gradient(135deg,#ff1d3b,#d70827); color:white !important; font-weight:850;
            border:1px solid rgba(255,86,105,.78); border-radius:10px; box-shadow:0 7px 22px rgba(229,9,47,.22);
        }
        .stButton > button:hover, .stDownloadButton > button:hover {
            background:linear-gradient(135deg,#ff3850,#ed0e31); box-shadow:0 10px 28px rgba(229,9,47,.36); transform:translateY(-1px);
        }

        .stTextInput input,.stNumberInput input,.stSelectbox div[data-baseweb="select"] > div,.stMultiSelect div[data-baseweb="select"] > div {
            background:#0a0c11 !important; color:#f7f8fb !important; border:1px solid rgba(255,255,255,.13) !important; border-radius:10px !important;
        }
        .stTextInput input:focus,.stNumberInput input:focus { border-color:#e5092f !important; box-shadow:0 0 0 1px rgba(229,9,47,.5) !important; }
        .stSlider [role="slider"] { background:#e5092f !important; border-color:#e5092f !important; }

        [data-testid="stFileUploader"] {
            border:1px dashed rgba(229,9,47,.86); border-radius:14px; padding:10px;
            background:linear-gradient(145deg,rgba(23,10,14,.70),rgba(9,10,14,.92));
            box-shadow:inset 0 0 26px rgba(229,9,47,.04);
        }
        [data-testid="stFileUploader"] section { background:transparent !important; }
        [data-testid="stDataFrame"] { border:1px solid rgba(229,9,47,.38); border-radius:12px; overflow:hidden; }
        [data-testid="stAlert"] { border-radius:12px; border-left:3px solid #e5092f; background:rgba(15,17,22,.92); }
        [data-testid="stExpander"] { background:rgba(10,12,17,.82); border:1px solid rgba(229,9,47,.25); border-radius:12px; }
        hr { border-color:rgba(229,9,47,.25) !important; }
        a { color:#ff667d !important; }
        ::selection { background:rgba(229,9,47,.35); color:#fff; }


        /* ===== PREMIUM RESULT DASHBOARD V2 ===== */
        .rg-section {
            margin: 26px 0 14px; padding: 14px 18px;
            border-left: 4px solid #e5092f;
            background: linear-gradient(90deg, rgba(229,9,47,.10), rgba(229,9,47,0));
            border-radius: 10px;
        }
        .rg-section-title { font-size:24px; font-weight:900; color:#fff; letter-spacing:-.5px; }
        .rg-section-sub { color:#9298a4; font-size:13px; margin-top:3px; }
        .rg-card {
            background:linear-gradient(145deg,#101218,#080a0e);
            border:1px solid rgba(229,9,47,.42); border-radius:16px;
            padding:20px; box-shadow:0 12px 32px rgba(0,0,0,.32); min-height:118px;
        }
        .rg-card-label {
            color:#969ba6; font-size:12px; text-transform:uppercase;
            letter-spacing:1.2px; font-weight:800;
        }
        .rg-card-value { color:#fff; font-size:30px; font-weight:950; margin-top:8px; }
        .rg-card-value.red { color:#ff3b57; text-shadow:0 0 18px rgba(229,9,47,.24); }
        .rg-card-value.green { color:#22d477; }
        .rg-card-value.yellow { color:#ffc247; }
        .rg-pill {
            display:inline-block; padding:5px 11px; border-radius:999px;
            background:rgba(229,9,47,.13); border:1px solid rgba(229,9,47,.40);
            color:#ff667d; font-size:12px; font-weight:850; margin-top:7px;
        }
        .rg-image-head {
            padding:12px 15px; margin-bottom:8px; border-radius:10px;
            background:#0c0e13; border:1px solid rgba(229,9,47,.28);
            color:#fff; font-weight:850;
        }
        .rg-footer {
            margin-top:26px; padding:18px; border-top:1px solid rgba(229,9,47,.25);
            text-align:center; color:#747b87; font-size:12px;
        }

        /* sidebar polish */
        [data-testid="stSidebar"] h2,[data-testid="stSidebar"] h3 { color:#f5f6f8 !important; }
        [data-testid="stSidebar"] .stButton > button { min-height:44px; font-size:15px; }
        [data-testid="stSidebar"] [data-testid="stAlert"] { background:linear-gradient(135deg,rgba(18,53,39,.85),rgba(9,25,20,.85)); border-left-color:#22d477; }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# DAMAGE LABELS
# =========================================================

CLASS_LABELS = {
    "longitudinal_crack": "Longitudinal Crack",
    "transverse_crack": "Transverse Crack",
    "alligator_crack": "Alligator Crack",
    "pothole": "Pothole",
}


# =========================================================
# BASE RISK
# =========================================================

BASE_RISK = {
    "longitudinal_crack": 30,
    "transverse_crack": 35,
    "alligator_crack": 50,
    "pothole": 65,
}


# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():
    return YOLO(str(MODEL_PATH))


# =========================================================
# SMART LOCATION SEARCH
# =========================================================

# Known places make the demo reliable even when OpenStreetMap/Nominatim
# does not contain a school/college/landmark by its exact name.
KNOWN_LOCATIONS = {
    "sri muthukumaran institute of technology": {
        "name": (
            "Sri Muthukumaran Institute of Technology, "
            "Chikkarayapuram, Near Mangadu, Chennai - 600069"
        ),
        "latitude": 13.018809,
        "longitude": 80.107698,
    },
}

KNOWN_LOCATION_ALIASES = {
    "sri muthukumaran institute of technology":
        "sri muthukumaran institute of technology",
    "muthukumaran institute of technology":
        "sri muthukumaran institute of technology",
    "sri muthukumaran institute":
        "sri muthukumaran institute of technology",
    "smit":
        "sri muthukumaran institute of technology",
    "smit college":
        "sri muthukumaran institute of technology",
    "smit chikkarayapuram":
        "sri muthukumaran institute of technology",
    "muthukumaran technology college":
        "sri muthukumaran institute of technology",
}


def normalize_query(text):
    """Normalize a place name for reliable alias matching."""
    text = str(text or "").lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def known_location_search(search_text):
    """Return a verified known place when the query matches an alias."""
    normalized = normalize_query(search_text)

    if not normalized:
        return []

    matched_key = None

    # Prefer the longest alias so a specific phrase wins over a short alias.
    for alias in sorted(KNOWN_LOCATION_ALIASES, key=len, reverse=True):
        alias_normalized = normalize_query(alias)
        if (
            normalized == alias_normalized
            or alias_normalized in normalized
        ):
            matched_key = KNOWN_LOCATION_ALIASES[alias]
            break

    if matched_key and matched_key in KNOWN_LOCATIONS:
        place = KNOWN_LOCATIONS[matched_key]
        return [dict(place)]

    return []

@st.cache_data(ttl=3600)
def search_location(search_text):
    """
    Search for locations using OpenStreetMap/Nominatim.
    Supports places across India without hard-coded city restrictions.
    """

    if not search_text or not search_text.strip():
        return []

    query = search_text.strip()

    # Keep verified/demo locations as a fast fallback.
    known_results = known_location_search(query)

    # Search globally through Nominatim, while preferring India.
    search_variants = [
        query,
        f"{query}, India",
    ]

    geolocator = Nominatim(
        user_agent="roadguardai-location-search-v5"
    )

    locations = []
    seen = set()

    for search_query in search_variants:
        try:
            results = geolocator.geocode(
                search_query,
                exactly_one=False,
                limit=10,
                addressdetails=True,
                language="en",
                timeout=15,
            )

            if not results:
                continue

            for place in results:
                try:
                    latitude = float(place.latitude)
                    longitude = float(place.longitude)
                except (TypeError, ValueError):
                    continue

                key = (
                    round(latitude, 5),
                    round(longitude, 5),
                )

                if key in seen:
                    continue

                seen.add(key)

                locations.append(
                    {
                        "name": place.address,
                        "latitude": latitude,
                        "longitude": longitude,
                    }
                )

        except Exception:
            continue

    # If online search failed but a verified location exists,
    # still return the verified location.
    if not locations and known_results:
        return known_results

    return locations[:10]

# =========================================================
# LOCATION IMAGE SEARCH
# =========================================================

@st.cache_data(ttl=1800)
def get_location_image(location_name, latitude, longitude):
    """
    Find a real photo for the selected location.

    Search order:
    1. Wikimedia Commons nearby geotagged photos
    2. Wikipedia nearby pages with images
    3. KartaView nearby street-level imagery

    Returns:
        image URL or None
    """

    try:
        latitude = float(latitude)
        longitude = float(longitude)
    except (TypeError, ValueError):
        return None

    # -----------------------------------------------------
    # 1. WIKIMEDIA COMMONS - GPS PHOTO SEARCH
    # -----------------------------------------------------

    try:
        url = "https://commons.wikimedia.org/w/api.php"

        params = {
            "action": "query",
            "format": "json",
            "list": "geosearch",
            "gscoord": f"{latitude}|{longitude}",
            "gsradius": 5000,
            "gsnamespace": 6,
            "gslimit": 10,
            "gsprimary": "all",
        }

        response = requests.get(
            url,
            params=params,
            timeout=12,
            headers={
                "User-Agent":
                    "RoadGuardAI/1.0 "
                    "(road damage detection project)"
            }
        )

        if response.ok:
            data = response.json()

            pages = (
                data
                .get("query", {})
                .get("geosearch", [])
            )

            for page in pages:

                title = page.get("title")

                if not title:
                    continue

                image_params = {
                    "action": "query",
                    "format": "json",
                    "titles": title,
                    "prop": "imageinfo",
                    "iiprop": "url",
                    "iiurlwidth": 1000,
                }

                image_response = requests.get(
                    url,
                    params=image_params,
                    timeout=12,
                    headers={
                        "User-Agent":
                            "RoadGuardAI/1.0 "
                            "(road damage detection project)"
                    }
                )

                if not image_response.ok:
                    continue

                image_data = image_response.json()

                image_pages = (
                    image_data
                    .get("query", {})
                    .get("pages", {})
                )

                for image_page in image_pages.values():

                    image_info = image_page.get(
                        "imageinfo",
                        []
                    )

                    if not image_info:
                        continue

                    image = image_info[0]

                    image_url = (
                        image.get("thumburl")
                        or image.get("url")
                    )

                    if image_url:
                        return image_url

    except Exception:
        pass

    # -----------------------------------------------------
    # 2. WIKIPEDIA - NEARBY PLACES WITH IMAGES
    # -----------------------------------------------------

    try:
        url = "https://en.wikipedia.org/w/api.php"

        params = {
            "action": "query",
            "format": "json",
            "generator": "geosearch",
            "ggscoord": f"{latitude}|{longitude}",
            "ggsradius": 10000,
            "ggslimit": 10,
            "prop": "coordinates|pageimages",
            "piprop": "thumbnail",
            "pithumbsize": 1000,
        }

        response = requests.get(
            url,
            params=params,
            timeout=12,
            headers={
                "User-Agent":
                    "RoadGuardAI/1.0 "
                    "(road damage detection project)"
            }
        )

        if response.ok:

            data = response.json()

            pages = (
                data
                .get("query", {})
                .get("pages", {})
            )

            for page in pages.values():

                thumbnail = page.get(
                    "thumbnail",
                    {}
                )

                image_url = thumbnail.get(
                    "source"
                )

                if image_url:
                    return image_url

    except Exception:
        pass

    # -----------------------------------------------------
    # 3. KARTAVIEW - NEARBY STREET PHOTO
    # -----------------------------------------------------

    try:

        url = "https://api.openstreetcam.org/2.0/photo/"

        params = {
            "lat": latitude,
            "lng": longitude,
            "zoomLevel": 17,
            "join": "sequence",
            "orderBy": "id",
            "orderDirection": "desc",
        }

        response = requests.get(
            url,
            params=params,
            timeout=15,
            headers={
                "User-Agent":
                    "RoadGuardAI/1.0"
            }
        )

        if response.ok:

            data = response.json()

            # KartaView may return photos in different
            # structures depending on API response version.

            possible_lists = []

            if isinstance(data, dict):

                if isinstance(data.get("result"), list):
                    possible_lists.append(
                        data["result"]
                    )

                if isinstance(data.get("photos"), list):
                    possible_lists.append(
                        data["photos"]
                    )

                if isinstance(data.get("data"), list):
                    possible_lists.append(
                        data["data"]
                    )

                if isinstance(
                    data.get("result"),
                    dict
                ):

                    result_data = data["result"]

                    if isinstance(
                        result_data.get("data"),
                        list
                    ):
                        possible_lists.append(
                            result_data["data"]
                        )

                    if isinstance(
                        result_data.get("photos"),
                        list
                    ):
                        possible_lists.append(
                            result_data["photos"]
                        )

            for photo_list in possible_lists:

                for photo in photo_list:

                    if not isinstance(photo, dict):
                        continue

                    image_url = (
                        photo.get("lth_name")
                        or photo.get("name")
                        or photo.get("th_name")
                    )

                    if image_url:
                        return image_url

    except Exception:
        pass

    # -----------------------------------------------------
    # 4. NO PHOTO
    # -----------------------------------------------------

    return None

# =========================================================
# SNAP LOCATION TO NEAREST ROAD
# =========================================================

@st.cache_data(ttl=3600)
def get_nearest_road(latitude, longitude):
    """Find the nearest drivable road to a searched landmark."""
    try:
        url = (
            "https://router.project-osrm.org/nearest/v1/driving/"
            f"{longitude},{latitude}"
        )

        response = requests.get(
            url,
            params={"number": 1},
            timeout=10,
        )

        if response.status_code != 200:
            return latitude, longitude

        data = response.json()
        waypoints = data.get("waypoints", [])

        if not waypoints:
            return latitude, longitude

        snapped = waypoints[0].get("location")
        if not snapped or len(snapped) < 2:
            return latitude, longitude

        return float(snapped[1]), float(snapped[0])

    except Exception:
        return latitude, longitude


# =========================================================
# SEVERITY
# =========================================================

def severity_from_risk(risk):

    if risk >= 80:
        return "Critical"

    if risk >= 60:
        return "High"

    if risk >= 40:
        return "Medium"

    return "Low"


# =========================================================
# PRIORITY
# =========================================================

def priority_from_risk(risk):

    if risk >= 80:
        return "Immediate repair"

    if risk >= 60:
        return "Repair within 7 days"

    if risk >= 40:
        return "Schedule inspection"

    return "Monitor"


# =========================================================
# ROAD HEALTH SCORE
# =========================================================

def road_health_score(risks):

    if not risks:
        return 100, "Excellent Condition"

    average_risk = sum(risks) / len(risks)

    score = max(
        0,
        round(100 - average_risk)
    )

    if score >= 80:
        status = "Excellent Condition"

    elif score >= 60:
        status = "Good Condition"

    elif score >= 40:
        status = "Fair Condition"

    elif score >= 20:
        status = "Poor Condition"

    else:
        status = "Critical Condition"

    return score, status


# =========================================================
# REPAIR ESTIMATION
# =========================================================

def get_repair_estimate(
    damage_type,
    risk,
    area
):

    damage = str(damage_type).lower()

    if risk >= 80:
        duration = "Within 24 hours"
        action = "Emergency repair"

    elif risk >= 60:
        duration = "Within 7 days"
        action = "Priority repair"

    elif risk >= 40:
        duration = "Within 30 days"
        action = "Scheduled maintenance"

    else:
        duration = "Within 3 months"
        action = "Routine maintenance"

    if "pothole" in damage:

        if area >= 5:
            cost = "₹15,000 - ₹50,000+"

        elif area >= 2:
            cost = "₹5,000 - ₹25,000"

        else:
            cost = "₹2,000 - ₹10,000"

    elif "alligator" in damage:

        if area >= 5:
            cost = "₹20,000 - ₹60,000+"

        elif area >= 2:
            cost = "₹8,000 - ₹30,000"

        else:
            cost = "₹3,000 - ₹15,000"

    elif (
        "longitudinal" in damage
        or "transverse" in damage
    ):

        if area >= 5:
            cost = "₹10,000 - ₹30,000"

        elif area >= 2:
            cost = "₹5,000 - ₹15,000"

        else:
            cost = "₹2,000 - ₹8,000"

    else:
        cost = "₹2,000 - ₹15,000"

    return duration, cost, action

def send_pwd_alert(
    report,
    location,
    latitude,
    longitude,
    annotated_image,
    live_latitude=None,
    live_longitude=None,
    live_accuracy=None,
):

    sender_email = st.secrets["EMAIL_ADDRESS"]
    sender_password = st.secrets["EMAIL_PASSWORD"]

    highest_risk = int(
        report["Risk score"].max()
    )

    overall_severity = severity_from_risk(
        highest_risk
    )

    top_damage = report.iloc[0]

    # Prefer the latest browser/device GPS position for the alert.
    # If live GPS is unavailable, keep the existing searched/manual
    # location coordinates as a safe fallback.
    use_live_gps = (
        live_latitude is not None
        and live_longitude is not None
    )

    alert_latitude = (
        float(live_latitude)
        if use_live_gps
        else float(latitude)
    )
    alert_longitude = (
        float(live_longitude)
        if use_live_gps
        else float(longitude)
    )

    gps_source = (
        "Live device/browser GPS"
        if use_live_gps
        else "Selected/manual report location"
    )

    google_maps_link = (
        "https://www.google.com/maps?q="
        f"{alert_latitude:.6f},{alert_longitude:.6f}"
    )

    accuracy_text = (
        f"±{float(live_accuracy):.1f} m"
        if use_live_gps and live_accuracy is not None
        else "Not available"
    )

    message = EmailMessage()

    message["Subject"] = (
        f"RoadGuardAI Alert - "
        f"{overall_severity} Road Damage"
    )

    message["From"] = sender_email
    message["To"] = PWD_EMAIL

    message.set_content(
        f"""
ROADGUARDAI - ROAD DAMAGE ALERT
================================

Location:
{location or "Not provided"}

ALERT GPS LOCATION:
Source: {gps_source}
Latitude: {alert_latitude:.6f}
Longitude: {alert_longitude:.6f}
Accuracy: {accuracy_text}

Google Maps:
{google_maps_link}

Damage Type:
{top_damage["Damage type"]}

Confidence:
{top_damage["Confidence"]}%

Area:
{top_damage["Area (%)"]}%

Risk Score:
{highest_risk}/100

Severity:
{overall_severity}

Priority:
{top_damage["Priority"]}

Repair Duration:
{top_damage["Repair duration"]}

Estimated Cost:
{top_damage["Estimated cost"]}

Recommended Action:
{top_damage["Recommended action"]}

Total Damage Instances:
{len(report)}

--------------------------------
Please inspect this location and
take necessary road maintenance action.

This alert was generated automatically
by RoadGuardAI.
"""
    )

    image_buffer = BytesIO()

    Image.fromarray(
        annotated_image
    ).save(
        image_buffer,
        format="JPEG"
    )

    image_buffer.seek(0)

    message.add_attachment(
        image_buffer.read(),
        maintype="image",
        subtype="jpeg",
        filename="road_damage_detected.jpg",
    )

    with smtplib.SMTP_SSL(
        "smtp.gmail.com",
        465
    ) as server:

        server.login(
            sender_email,
            sender_password
        )

        server.send_message(message)

    return True

st.markdown(
    """
    <div class="rg-hero">
        <div class="rg-brand">
            <div class="rg-logo">🛣️</div>
            <div>
                <div class="rg-title">RoadGuard<span>AI</span></div>
                <div class="rg-subtitle">AI-powered road damage detection, severity scoring, GPS mapping, and repair prioritisation</div>
            </div>
        </div>
        <div class="rg-tagline">Safer Roads<b>Brighter Future</b><div class="rg-line"></div></div>
    </div>
    """,
    unsafe_allow_html=True,
)

s1, s2, s3, s4 = st.columns(4)

s1.metric(
    "AI engine",
    "RTX 3050",
    "GPU online"
)

s2.metric(
    "Detection model",
    "YOLO11s",
    "Road-damage trained"
)

s3.metric(
    "Mapping",
    "Satellite GPS",
    "Map enabled"
)

s4.metric(
    "Report",
    "CSV export",
    "Ready"
)


# =========================================================
# MODEL CHECK
# =========================================================

if not MODEL_PATH.exists():

    st.error(
        "Road-damage model not found. "
        "Check models/road_damage_yolo11s.pt"
    )

    st.stop()


# =========================================================
# SESSION STATE
# =========================================================

if "location_results" not in st.session_state:
    st.session_state.location_results = []

if "selected_location" not in st.session_state:
    st.session_state.selected_location = None

if "live_latitude" not in st.session_state:
    st.session_state.live_latitude = None

if "live_longitude" not in st.session_state:
    st.session_state.live_longitude = None

if "live_accuracy" not in st.session_state:
    st.session_state.live_accuracy = None


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.header("Detection settings")

    confidence = st.slider(
        "Detection confidence",
        0.05,
        0.80,
        0.10,
        0.05
    )

    image_size = st.selectbox(
        "Image quality",
        [640, 800],
        index=0
    )

    road_id = st.text_input(
        "Road / Ward ID",
        placeholder="e.g., Ward 12"
    )

    # -----------------------------------------------------
    # LIVE GPS LOCATION
    # -----------------------------------------------------

    st.divider()
    st.subheader("📡 Live GPS Location")

    st.caption(
        "Use your device/browser GPS for the PWD alert email. "
        "Your existing searched location and satellite map remain unchanged."
    )

    if st.button("📍 Get Live GPS Location"):
        live_location = streamlit_geolocation()
    else:
        live_location = {}

    if isinstance(live_location, dict):
        live_error = live_location.get("error")

        if live_error:
            error_message = live_error.get(
                "message",
                "Unable to get live GPS location."
            ) if isinstance(live_error, dict) else str(live_error)

            st.warning(
                f"📍 Live GPS unavailable: {error_message}"
            )

        else:
            current_live_latitude = live_location.get("latitude")
            current_live_longitude = live_location.get("longitude")
            current_live_accuracy = live_location.get("accuracy")

            if (
                current_live_latitude is not None
                and current_live_longitude is not None
            ):
                st.session_state.live_latitude = float(
                    current_live_latitude
                )
                st.session_state.live_longitude = float(
                    current_live_longitude
                )

                if current_live_accuracy is not None:
                    st.session_state.live_accuracy = float(
                        current_live_accuracy
                    )

    if (
        st.session_state.live_latitude is not None
        and st.session_state.live_longitude is not None
    ):
        st.success("✅ Live GPS location ready")
        st.caption(
            f"Live GPS: {st.session_state.live_latitude:.6f}, "
            f"{st.session_state.live_longitude:.6f}"
        )

        if st.session_state.live_accuracy is not None:
            st.caption(
                f"Accuracy: ±{st.session_state.live_accuracy:.1f} m"
            )
    else:
        st.info(
            "Click the GPS button above and allow browser location permission."
        )

    st.divider()

    st.subheader("📍 Location Search")

    location_query = st.text_input(
        "Search school, college, hospital, street, landmark",
        placeholder="e.g. Sri Muthukumaran Institute of Technology"
    )

    search_clicked = st.button(
        "🔎 Locate",
        use_container_width=True
    )

    if search_clicked:

        if len(location_query.strip()) < 3:

            st.warning(
                "Enter at least 3 characters."
            )

            st.session_state.location_results = []
            st.session_state.selected_location = None

        else:

            with st.spinner(
                "Searching location..."
            ):

                results = search_location(
                    location_query
                )

            if results:

                st.session_state.location_results = results
                st.session_state.selected_location = results[0]

                st.success(
                    f"Found {len(results)} location(s)."
                )

            else:

                st.session_state.location_results = []
                st.session_state.selected_location = None

                st.error(
                    "Location not found."
                )

                st.caption(
                    "Try adding Chennai, Mangadu, "
                    "Tamil Nadu or the PIN code."
                )

    if st.session_state.location_results:

        results = st.session_state.location_results

        result_names = [
            item["name"]
            for item in results
        ]

        selected_index = st.selectbox(
            "Choose the correct location",
            range(len(result_names)),
            format_func=lambda i: result_names[i],
            key="location_selection"
        )

        st.session_state.selected_location = (
            results[selected_index]
        )

    selected_location = (
        st.session_state.selected_location
    )

    # 📷 Location Photo
    location_image = None

    if selected_location:
        location_image = get_location_image(
            selected_location["name"],
            selected_location["latitude"],
            selected_location["longitude"],
        )

        if location_image:
            st.markdown("### 📷 Location Photo")
            st.image(
                location_image,
                caption=selected_location["name"],
                use_container_width=True,
            )
        else:
            st.info("📷 No photo found for this location.")

    if selected_location:

        # Original searched landmark/building coordinates
        place_name = selected_location["name"]
        place_latitude = selected_location["latitude"]
        place_longitude = selected_location["longitude"]

        # Snap the actual damage/report point to the nearest road
        with st.spinner("Finding nearest road..."):
            latitude, longitude = get_nearest_road(
                place_latitude,
                place_longitude,
            )

        location = place_name

        st.success("📍 Location selected")

        st.caption("🛣️ Damage point mapped to nearest road")
        st.caption(
            f"GPS: {latitude:.6f}, {longitude:.6f}"
        )

    else:

        location = location_query.strip()

        latitude = st.number_input(
            "Latitude",
            value=13.082700,
            format="%.6f"
        )

        longitude = st.number_input(
            "Longitude",
            value=80.270700,
            format="%.6f"
        )

    st.divider()

    st.success(
        "GPU inference enabled: "
        "NVIDIA RTX 3050"
    )

st.subheader("📷 Road Image Input")

input_method = st.radio(
    "Choose how to provide the road image:",
    [
        "📁 Upload Image",
        "📷 Take Photo"
    ],
    horizontal=True,
)


# =========================================================
# INPUT IMAGE
# =========================================================

uploaded_file = None
camera_file = None

if input_method == "📁 Upload Image":

    uploaded_file = st.file_uploader(
        "Upload a road image",
        type=[
            "jpg",
            "jpeg",
            "png"
        ],
    )

    if uploaded_file is None:

        st.info(
            "Upload a road image to start the analysis."
        )

        st.stop()

    image = Image.open(
        uploaded_file
    ).convert("RGB")

else:

    camera_file = st.camera_input(
        "📷 Capture road damage"
    )

    if camera_file is None:

        st.info(
            "Take a road photo to start the analysis."
        )

        st.stop()

    image = Image.open(
        camera_file
    ).convert("RGB")

model = load_model()

with st.spinner(
    "Analysing road damage with GPU..."
):

    result = model.predict(
        source=np.array(image),
        device=0,
        imgsz=image_size,
        conf=confidence,
        verbose=False,
    )[0]

annotated_image = (
    result.plot()[:, :, ::-1]
)

rows = []

image_width, image_height = image.size

image_area = (
    image_width * image_height
)


for box in result.boxes:

    class_id = int(
        box.cls[0].item()
    )

    class_name = result.names[
        class_id
    ]

    confidence_score = float(
        box.conf[0].item()
    )

    x1, y1, x2, y2 = (
        box.xyxy[0].tolist()
    )

    box_area = (
        max(0, x2 - x1)
        * max(0, y2 - y1)
    )

    area_percent = (
        box_area
        / image_area
    ) * 100

    base_risk = BASE_RISK.get(
        class_name,
        30
    )

    size_risk = min(
        30,
        area_percent * 3
    )

    confidence_risk = (
        confidence_score * 10
    )

    risk_score = min(
        100,
        round(
            base_risk
            + size_risk
            + confidence_risk
        )
    )

    if risk_score >= 70:

        traffic_light = (
            "🔴 Critical"
        )

    elif risk_score >= 40:

        traffic_light = (
            "🟡 Moderate"
        )

    else:

        traffic_light = (
            "🟢 Minor"
        )

    (
        repair_duration,
        repair_cost,
        repair_action,
    ) = get_repair_estimate(
        class_name,
        risk_score,
        area_percent,
    )

    rows.append(
        {
            "Road / Ward ID":
                road_id or "Not provided",

            "Location":
                location or "Not provided",

            "Damage type":
                CLASS_LABELS.get(
                    class_name,
                    class_name
                    .replace("_", " ")
                    .title(),
                ),

            "Confidence":
                round(
                    confidence_score * 100,
                    1
                ),

            "Area (%)":
                round(
                    area_percent,
                    2
                ),

            "Risk score":
                risk_score,

            "Traffic-light severity":
                traffic_light,

            "Severity":
                severity_from_risk(
                    risk_score
                ),

            "Priority":
                priority_from_risk(
                    risk_score
                ),

            "Repair duration":
                repair_duration,

            "Estimated cost":
                repair_cost,

            "Recommended action":
                repair_action,
        }
    )

st.markdown(
    '''
    <div class="rg-section">
        <div class="rg-section-title">🔍 AI Detection Results</div>
        <div class="rg-section-sub">YOLO11s road-damage analysis powered by NVIDIA RTX 3050</div>
    </div>
    ''',
    unsafe_allow_html=True,
)

if not rows:
    st.warning(
        "No road damage detected. "
        "Try reducing the detection confidence."
    )
    st.stop()

preview_report = pd.DataFrame(rows)
preview_risk = int(preview_report["Risk score"].max())
preview_severity = severity_from_risk(preview_risk)
risk_class = "red" if preview_risk >= 80 else ("yellow" if preview_risk >= 60 else "green")

m1, m2, m3, m4 = st.columns(4)

with m1:
    st.markdown(
        f'''
        <div class="rg-card">
            <div class="rg-card-label">Damage instances</div>
            <div class="rg-card-value">{len(preview_report)}</div>
            <div class="rg-pill">AI detected</div>
        </div>
        ''',
        unsafe_allow_html=True,
    )

with m2:
    st.markdown(
        f'''
        <div class="rg-card">
            <div class="rg-card-label">Highest risk</div>
            <div class="rg-card-value {risk_class}">{preview_risk}/100</div>
            <div class="rg-pill">Risk score</div>
        </div>
        ''',
        unsafe_allow_html=True,
    )

with m3:
    st.markdown(
        f'''
        <div class="rg-card">
            <div class="rg-card-label">Overall severity</div>
            <div class="rg-card-value {risk_class}">{preview_severity}</div>
            <div class="rg-pill">Traffic assessment</div>
        </div>
        ''',
        unsafe_allow_html=True,
    )

with m4:
    top_preview = preview_report.iloc[preview_report["Risk score"].idxmax()]
    st.markdown(
        f'''
        <div class="rg-card">
            <div class="rg-card-label">Top damage</div>
            <div class="rg-card-value" style="font-size:22px;">{top_preview["Damage type"]}</div>
            <div class="rg-pill">Confidence {top_preview["Confidence"]}%</div>
        </div>
        ''',
        unsafe_allow_html=True,
    )

st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

left, right = st.columns(2)

with left:
    st.markdown('<div class="rg-image-head">📷 Original Road Image</div>', unsafe_allow_html=True)
    st.image(image, use_container_width=True)

with right:
    st.markdown('<div class="rg-image-head">🧠 AI Annotated Detection</div>', unsafe_allow_html=True)
    st.image(annotated_image, use_container_width=True)

report = pd.DataFrame(rows)

risk_values = report[
    "Risk score"
].tolist()

health_score, health_status = (
    road_health_score(
        risk_values
    )
)

st.subheader(
    "🛣️ Overall Road Health"
)

col1, col2 = st.columns(2)

with col1:

    st.metric(
        "Road Health Score",
        f"{health_score}/100"
    )

with col2:

    st.metric(
        "Road Condition",
        health_status
    )

highest_risk = int(
    report["Risk score"].max()
)

overall_severity = (
    severity_from_risk(
        highest_risk
    )
)

a, b, c = st.columns(3)

a.metric(
    "Damage instances",
    len(report)
)

b.metric(
    "Highest risk score",
    f"{highest_risk}/100"
)

c.metric(
    "Overall severity",
    overall_severity
)

st.markdown(
    '''
    <div class="rg-section">
        <div class="rg-section-title">📊 Damage Assessment & Repair Priority</div>
        <div class="rg-section-sub">Actionable maintenance recommendations generated from detected road damage</div>
    </div>
    ''',
    unsafe_allow_html=True,
)

st.dataframe(
    report,
    use_container_width=True,
    hide_index=True
)

st.markdown(
    '''
    <div class="rg-section">
        <div class="rg-section-title">🛰️ Satellite Damage Location</div>
        <div class="rg-section-sub">Selected landmark and nearest-road damage point</div>
    </div>
    ''',
    unsafe_allow_html=True,
)

damage_map = folium.Map(
    location=[
        latitude,
        longitude
    ],
    
    zoom_start=18,
    min_zoom=3,
    max_zoom=18,
    tiles=None,
    control_scale=True,
    zoom_control=True,
    scrollWheelZoom=True,
)

folium.TileLayer(
    tiles="OpenStreetMap",
    name="🗺️ Road Map",
    attr="© OpenStreetMap contributors",
    overlay=False,
    control=True,
).add_to(damage_map)

satellite_layer = folium.TileLayer(
    tiles=(
        "https://server.arcgisonline.com/"
        "ArcGIS/rest/services/World_Imagery/"
        "MapServer/tile/{z}/{y}/{x}"
    ),
    name="🛰️ Satellite",
    attr="Tiles © Esri",
    overlay=False,
    control=True,
    max_native_zoom=18,
    max_zoom=18,
)
satellite_layer.add_to(damage_map)

folium.LayerControl(
    position="topright",
    collapsed=True,
).add_to(damage_map)

if st.session_state.selected_location:

    selected_place = st.session_state.selected_location

    # Blue marker = searched landmark/building
    folium.Marker(
        [
            selected_place["latitude"],
            selected_place["longitude"],
        ],
        popup=(
            "<b>Selected Place</b><br>"
            f"{selected_place['name']}<br>"
            "<b>Damage point:</b> nearest road"
        ),
        tooltip="📍 Selected Place",
        icon=folium.Icon(
            color="blue",
            icon="map-marker"
        ),
    ).add_to(damage_map)

folium.Marker(
    [
        latitude,
        longitude
    ],
    popup=(
        f"<b>RoadGuardAI Damage Report</b><br>"
        f"Location: {location}<br>"
        f"Mapped to: Nearest Road<br>"
        f"Damages: {len(report)}<br>"
        f"Risk: {highest_risk}/100<br>"
        f"Severity: {overall_severity}"
    ),
    tooltip=(
        f"⚠️ {len(report)} damage(s) | "
        f"Risk {highest_risk}/100"
    ),
    icon=folium.Icon(
        color="red",
        icon="warning-sign"
    ),
).add_to(damage_map)

legend_html = """
<div style="
    position: fixed;
    bottom: 30px;
    left: 30px;
    width: 220px;
    z-index: 9999;
    background-color: #ffffff !important;
    color: #111111 !important;
    border: 2px solid #333333;
    border-radius: 8px;
    padding: 12px;
    font-size: 14px;
    line-height: 1.7;
    box-shadow: 0 3px 12px rgba(0,0,0,0.35);
">
    <div style="color:#111111 !important; font-weight:800; font-size:16px;">
        RoadGuardAI Map
    </div>

    <div style="color:#111111 !important;">
        🔵 Selected Location
    </div>

    <div style="color:#111111 !important;">
        🔴 Road Damage Location
    </div>

    <div style="color:#111111 !important;">
        🛰️ Satellite View
    </div>
</div>
"""

damage_map.get_root().html.add_child(
    folium.Element(legend_html)
)

st_folium(
    damage_map,
    height=500,
    use_container_width=True,
    returned_objects=[],
)

st.markdown(
    '''
    <div class="rg-section">
        <div class="rg-section-title">🚨 PWD Road Damage Alert</div>
        <div class="rg-section-sub">Notify the configured PWD recipient when critical damage is detected</div>
    </div>
    ''',
    unsafe_allow_html=True,
)

if highest_risk >= 70:

    st.warning(
        f"Critical road damage detected at "
        f"{location or 'selected location'}."
    )

    if st.button(
        "📢 Send Alert to PWD",
        type="primary"
    ):

        try:

            send_pwd_alert(
                report,
                location,
                latitude,
                longitude,
                annotated_image,
                live_latitude=st.session_state.live_latitude,
                live_longitude=st.session_state.live_longitude,
                live_accuracy=st.session_state.live_accuracy,
            )

            st.success(
                "✅ Alert successfully sent "
                "to PWD office."
            )

        except Exception as e:

            st.error(
                f"❌ Alert failed: {e}"
            )

else:

    st.info(
        "No critical damage detected. "
        "PWD emergency alert is not required."
    )

st.markdown(
    '''
    <div class="rg-section">
        <div class="rg-section-title">📄 Assessment Report</div>
        <div class="rg-section-sub">Download a professional PDF report of the AI road-damage assessment</div>
    </div>
    ''',
    unsafe_allow_html=True,
)

pdf_report = create_pdf_report(
    report_df=report,
    road_id=road_id,
    location=location,
    latitude=latitude,
    longitude=longitude,
)

st.download_button(
    "📥 Download Assessment Report (PDF)",
    pdf_report.getvalue(),
    file_name="RoadGuardAI_Road_Damage_Report.pdf",
    mime="application/pdf",
)

st.markdown(
    '''
    <div class="rg-footer">
        RoadGuardAI • AI Road Monitoring & Maintenance Decision Support
    </div>
    ''',
    unsafe_allow_html=True,
)

st.caption(
    "Risk score is a prototype "
    "decision-support estimate based "
    "on damage type, detected area, "
    "and model confidence. It is not "
    "a structural engineering assessment."
)
