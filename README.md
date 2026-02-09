# Agents
I am learning how to build production grade agents.

Here is the folder structure: (modular architecture)

├── app/                         # Main Application Source Code
│   ├── api/                     # API Route Handlers
│   │   └── v1/                  # Versioned API (v1 endpoints)
│   ├── core/                    # Core Application Config & Logic
│   │   ├── langgraph/           # AI Agent / LangGraph Logic
│   │   │   └── tools/           # Agent Tools (search, actions, etc.)
│   │   └── prompts/             # AI System & Agent Prompts
│   ├── models/                  # Database Models (SQLModel)
│   ├── schemas/                 # Data Validation Schemas (Pydantic)
│   ├── services/                # Business Logic Layer
│   └── utils/                   # Shared Helper Utilities
├── evals/                       # AI Evaluation Framework
│   └── metrics/                 # Evaluation Metrics & Criteria
│       └── prompts/             # LLM-as-a-Judge Prompt Definitions
├── grafana/                     # Grafana Observability Configuration
│   └── dashboards/              # Grafana Dashboards
│       └── json/                # Dashboard JSON Definitions
├── prometheus/                  # Prometheus Monitoring Configuration
├── scripts/                     # DevOps & Local Automation Scripts
│   └── rules/                   # Project Rules for Cursor
└── .github/                     # GitHub Configuration
    └── workflows/               # GitHub Actions CI/CD Workflows

Trying to follow separation of concern
- `app/` directory contains main application code, API routes, core logic, database models, and utility functions
- `evals/` consists of evaluation framework for assessing AI performance using various metrics and prompts
- `grafana/` and `prometheus/` config files for monitoring and observability tools.



Here I have used slowAPI for rate limiting
It is an alternative for fancy API gateways which are often expensive (money or resources)
If our endpoint's limit is set to 5/minute, the 1st 5 requests within 1 minute is ok.
But the 6th request within 1 min will fail.
USed to stop abusive behaviour from bad guys (DOS attacks)

One important thing in rate limitinng is identifying users
it is done based on IP addresses
the key function `request.client.host` is responsible for identifying users.
When the limit is exceeded, slowapi raises a RateLimitExceeded error.


Markers:
1. limiter.py -> parameter needs list[StrOrCallableStr] but passing list[str]
did an AI fix list(settings.RATE_LIMITER_DEFAULT) which fixed the warning but still doubtful.