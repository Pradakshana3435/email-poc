import re

MARKERS = {
    "no_reply":         r"do not reply|do-not-reply|\bno-?reply@|please do not respond",
    "auto_generated":   r"automatically generated|auto-generated|automated (?:message|email)|system-generated",
    "unsubscribe":      r"unsubscribe|manage (?:your )?(?:email )?preferences|opt[- ]out of|view (?:this email )?in (?:your )?browser",
    "if_this_was_you":  r"if this was you|if this wasn'?t you|if you (?:don'?t|do not) recogni[sz]e this",
    "new_signin":       r"new (?:sign-in|signin|login|log-in) (?:to|on|detected)|we noticed a new (?:sign-in|login)",
    "booking_confirmed": r"your (?:booking|order|payment|ticket|appointment) (?:is|has been) (?:confirmed|received|successful)|order (?:summary|confirmation)",
    "amount_paid":      r"amount paid|payment (?:received|successful)|invoice (?:no|number)",
    "shipping_update":  r"has been (?:shipped|dispatched)|out for delivery",
    "promo":            r"shop now|promo code|coupon code|limited[- ]time|bonus (?:miles|points)|(?:flat|extra|up to) \d+% ?(?:off|discount)|instant discount",
    "feedback_request": r"write (?:a )?review|share your (?:thoughts|feedback)|rate your (?:experience|purchase)|love to get your feedback",
    "why_received":     r"you(?:[’']re| are| have)? (?:receiving|received) this (?:email|message)",
    "no_action":        r"no action (?:is )?(?:needed|required)|nothing you need to do|nothing for you to do",
    "footer_legal":     r"©\s*\d{4}|all rights reserved",
}

MIN_HITS = 2


def detect_notice(subject, body):
    """Return (is_automated_notice, list_of_markers_hit)."""
    s = subject if isinstance(subject, str) else ""
    b = body if isinstance(body, str) else ""
    text = (s + " " + b).lower()
    hits = [name for name, pat in MARKERS.items() if re.search(pat, text)]
    return len(hits) >= MIN_HITS, hits