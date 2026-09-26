"""Call the local server the way you'd call TypeSafe.
TOKENFOLD_URL=http://localhost:8080 uv run python examples/client.py"""

import json
import os

import requests

req = {
    "state": {
        "ticket_message": (
            "Our API integration started returning 500 errors on every request about 20 minutes ago, "
            "and we can't process any customer orders until this is fixed."
        ),
        "plan": "enterprise",
    },
    "questions": {
        "department": {
            "type": "choice",
            "instructions": "Which team should handle `ticket_message`?",
            "criteria": {
                "billing": "Payment or subscription issues",
                "technical": "Bugs, errors, outages or integration problems",
                "sales": "Pricing or account questions",
            },
        },
        "is_urgent": {
            "type": "noul",
            "instructions": "Does `ticket_message` convey urgency or time-sensitivity?",
            "criteria": {
                "true": "explicitly time-sensitive, blocked, outage",
                "false": "no urgency expressed",
            },
        },
        "frustration": {
            "type": "score",
            "instructions": "How frustrated does the customer appear in `ticket_message`?",
            "criteria": ["Calm, just stating facts", "Frustrated but civil", "Very angry, strong language"],
        },
    },
}
r = requests.post(
    f"{os.environ.get('TOKENFOLD_URL', 'http://localhost:8080')}/v1/tokenfold", json=req, timeout=30
)
print(r.status_code)
print(json.dumps(r.json(), indent=2))
