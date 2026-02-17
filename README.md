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


The Service Layer for AI agents
Using API routes (Controllers)



Markers:
1. limiter.py -> parameter needs list[StrOrCallableStr] but passing list[str]
did an AI fix list(settings.RATE_LIMITER_DEFAULT) which fixed the warning but still doubtful.


Things I can improve:
1. Context management part where they are using trim_message langchain function for resizing 
the context passed to the LLM, I can improve it. 
Can have another agent to evaluate the context and trim only the parts that don't add value for the future purpose.
Alot can be added and improved, btu lets make everything work first and have a blueprint ready.

2. For connection pooling here they are using sqlalchemy.pool.QueuePool
We can use Pgbouncer or soemthing like this to scale it properly for produciton when users increase.
Will need to write an implementation of this.

3. LLM Registry -> we can have groups of models in the registry, separating reasoning and flash models.
Based on the user selection or tasks assigned, model selection will be done.

4. Have to make the long_term_memroy compatible with the other models.
The way it is storing the context now is only for openai models, cannot put in other models.