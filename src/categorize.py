"""Spending-group labelling for the mock database.

IMPORTANT (per project requirement): this is NOT a categorisation *pipeline*
or a trained classifier. It is a one-time, deterministic labelling of the
real transactions in the two Swedbank statements so that the spending group
lives as **data** inside the mock database. The keyword tables below were
curated by reading the actual merchant / counterparty strings that appear in
these specific statements, so coverage is high and every label is auditable.

Each transaction is assigned exactly one ``spending_group``.
"""

from __future__ import annotations

# Human-readable description of every spending group (also stored in the DB).
SPENDING_GROUPS = {
    "savings": "Round-up 'Rahakoguja' transfers and Easy Saver / own-account savings movements",
    "income": "Incoming money: salary and peer-to-peer reimbursements/refunds",
    "groceries": "Supermarkets and food shops",
    "dining": "Restaurants, cafes, fast food and food delivery",
    "transport": "Ride-hailing, taxis, public transport tickets and fuel",
    "travel": "Intercity buses, flights, hotels and bookings",
    "subscriptions": "Recurring digital subscriptions and app billing",
    "utilities": "Telecom, dormitory rent and household utilities",
    "leisure": "Gym, books, tickets, beauty and hobbies",
    "shopping": "Clothing, electronics and general retail",
    "health": "Pharmacies and medical services",
    "cash": "ATM cash withdrawals",
    "transfer": "Outgoing transfers to other people / external accounts",
    "other": "Uncategorised (fallback)",
}

# Keyword rules evaluated against an upper-cased "<counterparty> || <details>"
# string. Order matters: the first group whose any-keyword matches wins, so
# more specific groups are listed before broader ones.
_RULES: list[tuple[str, tuple[str, ...]]] = [
    # --- self / savings movements (matched on details) ---
    ("savings", (
        "RAHAKOGUJASSE", "EASY SAVER", "TRANSFER BETWEEN OWN ACCOUNTS",
    )),
    # --- cash ---
    ("cash", ("SULARAHA",)),
    # --- money-transfer providers / outgoing transfers ---
    ("transfer", (
        "WISE", "TRUSTLY", "REVOLUT", "HOLM BANK", "PAYSERA", "PAYPAL",
    )),
    # --- subscriptions (digital, recurring) ---
    ("subscriptions", (
        "SPOTIFY", "APPLE.COM/BILL", "NETFLIX", "YOUTUBE", "GOOGLE",
        "OPENAI", "CURSOR, AI POWERED IDE", "ICLOUD", "PATREON", "AUDIBLE",
        "DISNEY", "HBO", "MICROSOFT", "ADOBE", "TELEGRAM",
    )),
    # --- utilities / housing ---
    ("utilities", (
        "TELE2", "TELIA", "ELISA", "ELEKTRILEVI", "EESTI ENERGIA",
        "ULIOPILASKULA", "ÜLIÕPILASKÜLA", "TTU UE", "UTILITIES",
    )),
    # --- travel (before transport: intercity / hotels / attractions) ---
    ("travel", (
        "OMIO", "FLIXBUS", "BOOKING", "BKG*", "HOTEL", "AIRBNB", "RYANAIR",
        "WIZZ", "LUFTHANSA", "AIRBALTIC", "CONTBUS", "QUINTA", "RCV -",
        "HOSTEL", "EXPEDIA", "TRIP.COM", "LUX EXPRESS", "DB AUTOMATEN",
        "DB VERTRIEB", "CAB 20", "MOBILLY", "MINIATUR WUNDERLAND",
    )),
    # --- transport ---
    ("transport", (
        "BOLT.EU", "BOLT.EUO", "TAXI", "TAKSI", "PILET.EE", "TALLINN.PILET",
        "OLEREX", "NESTE", "CIRCLE K", "ALEXELA", "PARKING", "PARKIMINE",
        "BOLT.EUR",
    )),
    # --- dining ---
    ("dining", (
        "KOHVIK", "KOHV", "CAFE", "RESTORAN", "RESTAURANT", "SUSHI", "KFC",
        "MCDONALD", "HESBURGER", "BURGER KING", "PIZZA", "DODO", "KEBAB",
        "SHAURMA", "SHAWARMA", "WOLT", "LIDO", "ST.VITUS", "ARAXES",
        "PANNKOOGIMAJA", "TIN BHAI", "VEERENNI", "BISTRO", "GRILL", "BURGIR",
        "PLUS KOHVIKUD", "MATSIMOKA", "DRINK", "BAR ", "PUB", "VAPIANO",
        "SUBWAY", "CHICKEN", "FOOD", "RAMEN", "CHOPSTICKS", "WAFFLE",
        "CAFFEINE", "COFFEE", "PAGARID", "SIGA LA VACA", "UMAMI", "GREEK",
        "CAJNICA", "APELSIN", "ROTERMANNI SHOK", "GRUUV", "BABU", "TARO",
        "FRISCHEM", "WANDELHALLE", "DODO PIZZA", "NOODLE", "DELICE",
    )),
    # --- groceries ---
    ("groceries", (
        "RIMI", "LIDL", "MAXIMA", "PRISMA", "SOLARISE TOIDUPOOD", "COOP",
        "SELVER", "TIPI POOD", "SOPRUSE KAUPLUS", "PROMO CASH", "BARBORA",
        "GROSSI", "R-KIOSK", "RKIOSK", "TOIDUPOOD", "KAUPLUS", "EDEKA",
        "EDK*", "WOOLWORTH",
    )),
    # --- health ---
    ("health", (
        "APTE", "APTEEK", "BENU", "SUDAMEAPTEEK", "SÜDAMEAPTEEK", "CLINIC",
        "KLIINIK", "DENTAL", "HAMBA", "MEDICAL", "PHARMACY",
    )),
    # --- shopping (retail) ---
    ("shopping", (
        "H&M", "H & M", "PEPCO", "NEW YORKER", "RESERVED", "JO MALONE",
        "PANDORA", "EURONICS", "HUMANA", "ZARA", "PULL&BEAR", "BERSHKA",
        "SPORTSDIRECT", "DECATHLON", "IKEA", "PIKO", "RAVELSOFT", "MKK*",
        "STORE", "SHOP", "APRANGA", "HOUSE KRISTIINE", "KLICK", "CROPPTOWN",
        "FLYING TIGER", "MCPAPER", "ALIEXPRESS", "K-RAUTA", "FUTPAL",
        "EURONICS", "TOILET SERVICE", "AS LUX", "D3 ",
    )),
    # --- leisure (gym, books, tickets, beauty, hobbies) ---
    ("leisure", (
        "GYMEESTI", "GYM", "FITNESS", "RAHVA RAAMAT", "FIENTA", "TICKET",
        "PILET", "KINO", "CINEMA", "ART SALONG", "MUSEUM", "MUUSEUM",
        "SALOON", "SALON", "BEAUTY", "BOARD GAME", "STEAM", "LAUNDRY",
        "MNOGO KNIG", "KULTUURIVABRIK", "RAAMAT", "KNIG",
    )),
]


def categorize(counterparty: str, details: str, amount: float) -> str:
    """Return the spending group for one transaction.

    The label is derived deterministically from the merchant/counterparty and
    details text, with a couple of amount-aware fallbacks for inflows.
    """
    text = f"{counterparty or ''} || {details or ''}".upper()

    for group, keywords in _RULES:
        if any(kw in text for kw in keywords):
            return group

    # Inflows that did not match a specific merchant rule are income
    # (salary, peer-to-peer reimbursements, refunds).
    if amount > 0:
        return "income"

    # Remaining outgoing money with no merchant match: a person's name
    # counterparty (no digits / not a card line) is treated as a transfer.
    cp = (counterparty or "").strip()
    is_card_line = details.strip().startswith("'") or "516737" in (details or "")
    if cp and not is_card_line and not any(ch.isdigit() for ch in cp):
        return "transfer"

    return "other"
