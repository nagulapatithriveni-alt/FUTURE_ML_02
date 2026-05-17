"""
generate_dataset.py
-------------------
This script creates a realistic synthetic dataset of customer support tickets.
Run this once to generate 'support_tickets.csv' in the data/ folder.
"""

import pandas as pd
import random

random.seed(42)

# ─────────────────────────────────────────────
# 1.  RAW TICKET TEMPLATES
#     Each entry is (ticket_text, category, priority)
# ─────────────────────────────────────────────
ticket_templates = [

    # ── Technical Issue ─────────────────────
    ("My application keeps crashing every time I try to open it. I've reinstalled it three times.",          "Technical Issue", "High"),
    ("The website is down and I cannot access any of my data. This is urgent for my business.",              "Technical Issue", "High"),
    ("I'm getting a 500 internal server error on every page. Nothing loads at all.",                         "Technical Issue", "High"),
    ("My account dashboard is completely broken. All features are unresponsive.",                             "Technical Issue", "High"),
    ("The mobile app crashes immediately on launch on iOS 17. Other apps work fine.",                         "Technical Issue", "High"),
    ("Cannot connect to the API. Getting timeout errors on every single request.",                            "Technical Issue", "High"),
    ("The system is extremely slow today. Every action takes several minutes.",                               "Technical Issue", "Medium"),
    ("Some images are not loading on the product page. The layout looks broken.",                             "Technical Issue", "Medium"),
    ("The search feature is returning wrong results. It shows unrelated items.",                              "Technical Issue", "Medium"),
    ("Dark mode toggle does not save my preference. Resets every time I reload.",                             "Technical Issue", "Medium"),
    ("The notification emails have a broken image at the top. Minor but looks unprofessional.",               "Technical Issue", "Low"),
    ("There is a small typo on the settings page under the 'Privacy' section.",                               "Technical Issue", "Low"),
    ("The loading spinner disappears too quickly before content finishes loading.",                            "Technical Issue", "Low"),
    ("Dropdown menu on mobile has a slight alignment issue.",                                                  "Technical Issue", "Low"),

    # ── Billing ─────────────────────────────
    ("I was charged twice for the same subscription this month. Please refund immediately.",                   "Billing",         "High"),
    ("My credit card was charged without my authorization. This is fraudulent!",                              "Billing",         "High"),
    ("I cancelled my plan but was still billed for the next cycle. This is unacceptable.",                   "Billing",         "High"),
    ("I am being charged the wrong amount. The price on my invoice doesn't match what was advertised.",       "Billing",         "High"),
    ("My payment failed but money was deducted from my account. Please investigate.",                         "Billing",         "High"),
    ("I want to upgrade my plan. Can you tell me what the pricing options are?",                              "Billing",         "Medium"),
    ("My invoice shows tax charges I don't think I should be paying. Can you explain?",                       "Billing",         "Medium"),
    ("I need a copy of all my invoices from last year for accounting purposes.",                               "Billing",         "Medium"),
    ("When does my current billing cycle end? I can't find this info in my dashboard.",                       "Billing",         "Low"),
    ("Is there a student discount available for the premium plan?",                                           "Billing",         "Low"),
    ("Can I switch from monthly to annual billing to save money?",                                            "Billing",         "Low"),
    ("What payment methods do you accept? Does the platform support PayPal?",                                 "Billing",         "Low"),

    # ── Account Access ───────────────────────
    ("I am locked out of my account and cannot reset my password. The email never arrives.",                  "Account Access",  "High"),
    ("Someone else is logged into my account. I think I've been hacked.",                                     "Account Access",  "High"),
    ("My account has been suspended without any reason or prior notice.",                                     "Account Access",  "High"),
    ("Two-factor authentication is not sending codes to my phone. I cannot log in.",                         "Account Access",  "High"),
    ("All my saved data has disappeared after I logged in today. Everything is gone.",                        "Account Access",  "High"),
    ("I forgot my email address associated with the account. How do I recover it?",                           "Account Access",  "Medium"),
    ("I want to change my username but the option seems greyed out in settings.",                             "Account Access",  "Medium"),
    ("How do I enable two-factor authentication on my account for better security?",                          "Account Access",  "Medium"),
    ("I need to update the email address on my account to a new one.",                                        "Account Access",  "Medium"),
    ("Can I merge two accounts I accidentally created with different emails?",                                 "Account Access",  "Medium"),
    ("How do I delete my account permanently? I want all my data removed.",                                   "Account Access",  "Low"),
    ("I'd like to download a copy of all my account data as required by GDPR.",                               "Account Access",  "Low"),
    ("How do I change the language preference on my account?",                                                "Account Access",  "Low"),

    # ── Refund Request ───────────────────────
    ("I want a full refund immediately. The product is completely unusable for my needs.",                    "Refund Request",  "High"),
    ("I was charged for an annual plan by mistake and need a full refund right away.",                        "Refund Request",  "High"),
    ("The service was down for 3 days and I want compensation for the downtime I experienced.",               "Refund Request",  "High"),
    ("I accidentally purchased the wrong plan and need the difference refunded urgently.",                    "Refund Request",  "High"),
    ("I requested a refund two weeks ago and still haven't received it. What is happening?",                  "Refund Request",  "High"),
    ("I cancelled within the 30-day trial but was charged anyway. Please refund.",                            "Refund Request",  "Medium"),
    ("The features promised at signup are not available. I want a partial refund.",                           "Refund Request",  "Medium"),
    ("I'd like a pro-rated refund since I'm cancelling halfway through the billing period.",                  "Refund Request",  "Medium"),
    ("How long does the refund process usually take once approved?",                                          "Refund Request",  "Low"),
    ("What is your refund policy for annual subscriptions?",                                                  "Refund Request",  "Low"),

    # ── General Inquiry ──────────────────────
    ("What are the differences between the Basic, Pro, and Enterprise plans?",                                "General Inquiry", "Low"),
    ("Does your platform integrate with Slack and Microsoft Teams?",                                          "General Inquiry", "Low"),
    ("Can I export my data to CSV or Excel format?",                                                          "General Inquiry", "Low"),
    ("Is your service GDPR and HIPAA compliant?",                                                             "General Inquiry", "Low"),
    ("Do you offer a free trial before I commit to a paid plan?",                                             "General Inquiry", "Low"),
    ("What are your customer support hours and how can I reach a live agent?",                                "General Inquiry", "Low"),
    ("Is there an API available for custom integrations with our internal tools?",                            "General Inquiry", "Low"),
    ("Do you have a mobile app for Android and iOS?",                                                         "General Inquiry", "Low"),
    ("Can multiple team members share one account, or do we each need our own?",                              "General Inquiry", "Medium"),
    ("We are a school. Do you offer any educational institution discounts?",                                  "General Inquiry", "Low"),
    ("How is my data backed up and how often?",                                                               "General Inquiry", "Medium"),
    ("What uptime SLA do you guarantee for enterprise customers?",                                            "General Inquiry", "Medium"),
]

# ─────────────────────────────────────────────
# 2.  EXPAND TO ~300 ROWS BY ADDING VARIATIONS
# ─────────────────────────────────────────────
prefixes  = ["Hello, ", "Hi there, ", "Good morning, ", "Dear support, ", "To whom it may concern, ", ""]
suffixes  = [" Please help.", " Thank you.", " Urgent!", " Looking forward to your response.", " This is very important to me.", ""]
mid_adds  = [" I have been a customer for 3 years.", " This is my second time contacting support.", " I already tried the FAQ.", ""]

rows = []
for text, cat, pri in ticket_templates:
    # Original
    rows.append({"ticket_text": text, "category": cat, "priority": pri})
    # 4 variations per template
    for _ in range(4):
        new_text = random.choice(prefixes) + text + random.choice(mid_adds) + random.choice(suffixes)
        rows.append({"ticket_text": new_text.strip(), "category": cat, "priority": pri})

df = pd.DataFrame(rows).sample(frac=1, random_state=42).reset_index(drop=True)

# ─────────────────────────────────────────────
# 3.  SAVE
# ─────────────────────────────────────────────
import os
os.makedirs(os.path.dirname(__file__), exist_ok=True)
out_path = os.path.join(os.path.dirname(__file__), "support_tickets.csv")
df.to_csv(out_path, index=False)

print(f"✅ Dataset saved → {out_path}")
print(f"   Total rows  : {len(df)}")
print(f"\nCategory distribution:\n{df['category'].value_counts()}")
print(f"\nPriority distribution:\n{df['priority'].value_counts()}")
