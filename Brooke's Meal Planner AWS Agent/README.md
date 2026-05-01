# Brooke's Meal Planner AWS Agent

A personal AI dinner-planning assistant that helps a single person plan realistic weeknight meals, reduce food waste, and generate grocery lists with real package sizes and store-specific prices.

The app combines:

- A static HTML/CSS/JavaScript chat frontend
- An AWS Lambda backend
- Amazon Bedrock Converse API for the LLM agent loop
- DynamoDB for pantry, preferences, meal plans, chat history, and cached recipes
- Spoonacular for recipe discovery and recipe details
- Kroger Public API for live grocery product/package/price lookup
- A curated fallback package catalog for grocery items that Kroger does not return cleanly

The project is designed for a personal meal-planning use case, but the architecture could be adapted for other lightweight agentic apps that need persistent user state, tool calls, and external API lookups.

---

## Features

- Chat-based weekly dinner planning
- Persistent user preferences, dislikes, pantry items, and meal plans
- Recipe search through Spoonacular
- Recipe caching in DynamoDB to reduce Spoonacular quota usage
- Support for user-added cookbook recipes alongside Spoonacular recipes
- Kroger-family store product lookup for realistic grocery package sizes and prices
- Curated package-size fallback catalog when live Kroger data is unavailable
- Structured shopping-list rendering in the frontend
- CloudWatch-friendly structured JSON logging
- Optional shared-secret protection for the backend endpoint

---

## Project Structure

```text
.
├── index.html                      # Static frontend chat UI
└── lambda/
    ├── lambda_function.py          # AWS Lambda entrypoint
    ├── agent.py                    # Bedrock Converse agent loop
    ├── tools.py                    # Tool schemas and dispatch logic
    ├── system_prompt.py            # System prompt for the meal-planning assistant
    ├── db.py                       # DynamoDB data access layer
    ├── spoonacular.py              # Spoonacular API client
    ├── kroger.py                   # Kroger Public API client
    ├── packages.py                 # Grocery package-size lookup and fallback catalog
    ├── obs.py                      # Structured logging helper
    ├── seed_recipes.py             # Optional recipe-cache seeding script
    └── requirements.txt            # Python dependencies
```

---

## How It Works

### 1. Frontend

`index.html` is a single-page chat interface. It stores a generated `user_id` in browser `localStorage`, sends messages to the backend endpoint, and renders assistant responses.

The frontend also supports structured shopping-list rendering. Instead of relying on the LLM to produce a markdown table, the backend emits a special encoded token like:

```text
[[SHOPPING_LIST:...]]
```

The frontend detects that token, decodes the JSON payload, and renders a styled shopping-list component.

Before deploying the frontend, update this section in `index.html`:

```javascript
const API_URL = "https://abc123.lambda-url.us-east-2.on.aws/";
const APP_TOKEN = "";
```

`API_URL` should point to your Lambda Function URL or API Gateway endpoint. `APP_TOKEN`, if used, must match the backend `APP_TOKEN` environment variable.

---

### 2. Lambda Backend

`lambda_function.py` is the HTTP entrypoint. It expects a JSON request body:

```json
{
  "user_id": "some-user-id",
  "message": "Help me plan dinners for this week"
}
```

It returns:

```json
{
  "reply": "assistant response text",
  "request_id": "aws-request-id"
}
```

If `APP_TOKEN` is configured, the request must include this header:

```text
x-app-token: your-token-here
```

This is not full user authentication, but it is useful as a lightweight protection layer for a personal app or demo.

---

### 3. Agent Loop

`agent.py` runs the Bedrock Converse API loop:

1. Load recent chat history from DynamoDB.
2. Append the latest user message.
3. Send the conversation, system prompt, and tool definitions to Bedrock.
4. If the model requests a tool call, execute the tool through `tools.py`.
5. Append the tool result and continue the loop.
6. Return the final assistant response once the model stops calling tools.

A max-turns limit prevents runaway tool loops.

---

### 4. Tools

`tools.py` defines and dispatches the available tools. The assistant can:

- Read and update pantry items
- Read and update preferences
- Search recipes by ingredients
- Search recipes with filters
- Fetch recipe details
- Save and retrieve meal plans
- Mark recipes as disliked
- Look up realistic package sizes and prices
- Format shopping lists for frontend rendering

The model never directly receives a different user's ID. The `user_id` is passed implicitly by the backend during tool dispatch.

Recipe IDs are passed and stored as strings to support both numeric Spoonacular IDs (e.g. `"649030"`) and user-defined cookbook IDs (e.g. `"cookbook_baked_tilapia"`). Recipe-detail lookups check the DynamoDB cache first, then fall back to Spoonacular only for numeric IDs — non-numeric IDs that miss the cache return a clear error rather than calling out to Spoonacular.

---

### 5. DynamoDB Persistence

The app uses a single-table DynamoDB design.

Default table name:

```text
grocery-agent
```

The table should have:

```text
Partition key: pk  (String)
Sort key:      sk  (String)
```

Example rows:

```text
pk                    sk
USER#<user_id>         PROFILE
USER#<user_id>         PANTRY#<item>
USER#<user_id>         MEALPLAN#<date>
USER#<user_id>         DISLIKED#<recipe_id>
USER#<user_id>         CHAT
CACHE                  RECIPE#<recipe_id>           # Spoonacular ID, e.g. RECIPE#649030
CACHE                  RECIPE#cookbook_<slug>       # User-added recipe, e.g. RECIPE#cookbook_baked_tilapia
CACHE                  KROGER_LOC#<zip_code>
```

The `CACHE` partition is shared across users for recipe details and Kroger location lookups.

---

## Dependencies

Python dependencies are listed in `requirements.txt`:

```text
boto3>=1.34.0
requests>=2.31.0
```

Install locally with:

```bash
pip install -r requirements.txt
```

For Lambda deployment, package these dependencies with the backend code or provide them through a Lambda layer.

---

## Environment Variables

### Required for the Backend

| Variable | Required? | Description |
|---|---:|---|
| `AWS_REGION` | Yes | AWS region used for Bedrock, DynamoDB, and Lambda. Example: `us-east-2`. |
| `TABLE_NAME` | Recommended | DynamoDB table name. Defaults to `grocery-agent`. |
| `SPOONACULAR_API_KEY` | Yes for recipe search | Spoonacular API key. Used for recipe search and recipe details. |
| `KROGER_CLIENT_ID` | Yes for live grocery prices | Kroger Public API client ID. |
| `KROGER_CLIENT_SECRET` | Yes for live grocery prices | Kroger Public API client secret. |
| `KROGER_ZIP_CODE` | Optional | ZIP code used to find the nearest Kroger-family store. Defaults to `37203` (Nashville). |

### Optional Backend Configuration

| Variable | Required? | Description |
|---|---:|---|
| `APP_TOKEN` | Optional | Shared secret required in the `x-app-token` request header. If unset, token checking is skipped. |
| `MODEL_ID` | Optional | Bedrock model ID. Defaults to `us.anthropic.claude-sonnet-4-6`. |
| `MAX_AGENT_TURNS` | Optional | Max tool-calling turns before stopping. Defaults to `24`. |
| `MAX_OUTPUT_TOKENS` | Optional | Max output tokens for Bedrock response. Defaults to `10000`. |
| `TEMPERATURE` | Optional | Model temperature. Defaults to `0.7`. |

### Frontend Configuration

These are edited directly inside `index.html`:

| Constant | Description |
|---|---|
| `API_URL` | Lambda Function URL or API Gateway endpoint. |
| `APP_TOKEN` | Must match backend `APP_TOKEN` if backend token checking is enabled. Leave empty only for local testing or a private demo. |

---

## Important Security Note

Do not commit secrets.

Do not commit:

- `SPOONACULAR_API_KEY`
- `KROGER_CLIENT_ID`
- `KROGER_CLIENT_SECRET`
- `APP_TOKEN`
- AWS credentials
- `.env` files containing real values

For deployment, set secrets through AWS Lambda environment variables, AWS Secrets Manager, or your deployment platform's secret manager.

---

## AWS Setup

### 1. Create the DynamoDB Table

Create a DynamoDB table with:

```text
Table name: grocery-agent
Partition key: pk
Sort key: sk
Billing mode: On-demand
```

If you use a different table name, set:

```bash
TABLE_NAME=your-table-name
```

---

### 2. Enable Bedrock Model Access

In the AWS console:

1. Go to Amazon Bedrock.
2. Enable access to the model used by `MODEL_ID`.
3. Confirm the model is available in your selected region.

The default model ID in the code is:

```text
us.anthropic.claude-sonnet-4-6
```

You can override this with the `MODEL_ID` environment variable.

---

### 3. Create the Lambda Function

Runtime:

```text
Python 3.12
```

Handler:

```text
lambda_function.lambda_handler
```

From the project root, run:

```bash
cd lambda
rm -rf build grocery-agent.zip
mkdir build

cp *.py build/

pip install --target build/ -r requirements.txt

cd build
zip -r ../grocery-agent.zip .
cd ..
```

Then upload the generated deployment package to AWS Lambda:

```text
lambda/grocery-agent.zip
```

Make sure the Lambda package or layer includes:

```text
boto3
requests
```

---

### 4. Configure Lambda Environment Variables

Example:

```bash
AWS_REGION=us-east-2
TABLE_NAME=grocery-agent
MODEL_ID=us.anthropic.claude-sonnet-4-6
MAX_AGENT_TURNS=24

SPOONACULAR_API_KEY=your_spoonacular_key
KROGER_CLIENT_ID=your_kroger_client_id
KROGER_CLIENT_SECRET=your_kroger_client_secret
KROGER_ZIP_CODE=98052

APP_TOKEN=your_shared_secret
```

---

### 5. Configure IAM Permissions

The Lambda execution role needs permissions for:

- Writing CloudWatch logs
- Reading and writing the DynamoDB table
- Invoking the selected Bedrock model

At minimum, the role should be able to:

```text
logs:CreateLogGroup
logs:CreateLogStream
logs:PutLogEvents

dynamodb:GetItem
dynamodb:PutItem
dynamodb:DeleteItem
dynamodb:Query

bedrock:InvokeModel
```

Scope the DynamoDB permissions to your table ARN and the Bedrock permission to the model or models you intend to use.

---

### 6. Expose the Lambda

You can expose the backend through either:

- Lambda Function URL (strongly recommended due to API Gateway's timeout constraints)
- API Gateway HTTP API

If using a Lambda Function URL, configure CORS to allow your frontend origin.

The backend also returns JSON response headers, but CORS should be configured at the Function URL or API Gateway layer.

---

### 7. Deploy the Frontend

Because the frontend is a single static file, you can host it with:

- AWS Amplify
- S3 static website hosting
- CloudFront + S3
- GitHub Pages
- Any static hosting provider

Before deploying, edit `index.html`:

```javascript
const API_URL = "https://your-real-backend-url";
const APP_TOKEN = "your-shared-secret-if-used";
```

If you do not want the shared secret visible in frontend code, use a more complete authentication layer instead of `APP_TOKEN`. The current token approach is best treated as lightweight demo protection, not production-grade auth.

---

## Optional: Seed the Recipe Cache

Spoonacular has quota limits, so the repo includes `seed_recipes.py` to pre-populate DynamoDB with a curated set of recipe details.

Run this locally after configuring AWS credentials:

```bash
export SPOONACULAR_API_KEY=your_spoonacular_key
export AWS_REGION=us-east-2
export TABLE_NAME=grocery-agent

python seed_recipes.py
```

This searches for a default set of recipe categories, fetches recipe details, and stores slimmed recipe objects in DynamoDB under:

```text
pk = CACHE
sk = RECIPE#<recipe_id>
```

This makes later demo and evaluation runs less likely to burn through Spoonacular quota.

---

## Optional: Add Cookbook Recipes

In addition to recipes pulled from Spoonacular, the agent can use recipes you add yourself — recipes from a personal cookbook, family recipes, or anything else not on Spoonacular.

Cookbook recipes use string IDs prefixed with `cookbook_` so they cannot collide with Spoonacular's numeric IDs. They live in the same `CACHE` partition as Spoonacular recipes, and the agent treats them identically once they are in the cache.

### Recipe shape

Match the shape produced by Spoonacular detail responses (which is what `_slim_recipe_details` in `tools.py` returns). Each recipe should have:

```json
{
  "id": "cookbook_baked_tilapia",
  "title": "Baked Tilapia",
  "ready_in_minutes": 20,
  "servings": 4,
  "ingredients": [
    {"name": "tilapia fillets", "amount": 4, "unit": "", "original": "4 tilapia fillets (6 ounces each)"},
    {"name": "butter", "amount": 3, "unit": "tbsp", "original": "3 Tbsp butter, melted"}
  ],
  "instructions": "Place fish in a baking dish. ...",
  "source_url": null
}
```

A few conventions:

- `ingredients[].name` is what `get_realistic_package_sizes` looks up against Kroger and the curated catalog. Use simple terms (`"chicken breast"`, `"olive oil"`) rather than descriptors (`"organic free-range chicken breast"`).
- `ingredients[].original` is the human-readable version that the agent shows to the user.
- `unit` can be an empty string for whole-item ingredients (e.g. tilapia fillets, lemons).

### Storing a cookbook recipe in DynamoDB

Each recipe is one row in the `CACHE` partition:

```text
pk           = "CACHE"
sk           = "RECIPE#cookbook_baked_tilapia"
recipe_json  = (the JSON blob above, stringified)
cached_at    = (unix timestamp)
```

`recipe_json` is stored as a string because ingredient amounts are floats and DynamoDB's `boto3` resource returns numbers as `Decimal`, which is not JSON-serializable without extra plumbing. Storing as a string sidesteps that.

You can add cookbook recipes through the DynamoDB console (Create item → JSON view) or by writing a small local script that calls `db.save_cached_recipe(recipe_id, recipe_dict)` for each.

### How the agent uses them

Once a cookbook recipe is in the cache, the agent can:

- Fetch its details via `get_recipe_details` (uses the cache, never calls Spoonacular for `cookbook_*` IDs)
- Include it in saved meal plans
- Mark it as disliked

Cookbook recipes do not appear in `search_recipes` results, since those queries hit Spoonacular only. The user (or the agent on the user's behalf) needs to know the recipe ID to use a cookbook recipe — for example, by referencing it directly in chat (`"use cookbook_baked_tilapia for Wednesday"`).

---

## Local Development Notes

This project is primarily designed to run on AWS because it depends on:

- Bedrock
- DynamoDB
- Lambda-style request handling
- AWS credentials

You can still develop pieces locally.

Install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Set local environment variables:

```bash
export AWS_REGION=us-east-2
export TABLE_NAME=grocery-agent
export SPOONACULAR_API_KEY=your_spoonacular_key
export KROGER_CLIENT_ID=your_kroger_client_id
export KROGER_CLIENT_SECRET=your_kroger_client_secret
export KROGER_ZIP_CODE=98052
```

To test the Lambda handler locally, you can create a small script like:

```python
from lambda_function import lambda_handler
import json

event = {
    "httpMethod": "POST",
    "headers": {
        "x-app-token": "your_shared_secret"
    },
    "body": json.dumps({
        "user_id": "local-test-user",
        "message": "Help me plan dinners for Monday through Thursday."
    })
}

print(lambda_handler(event, None))
```

If `APP_TOKEN` is not set in your environment, the token check is skipped.

---

## Observability

`obs.py` emits structured JSON logs to stdout. In Lambda, stdout is captured by CloudWatch Logs.

Example CloudWatch Logs Insights query:

```sql
fields @timestamp, event, request_id, latency_ms, tool, status
| sort @timestamp desc
| limit 50
```

Useful event types include:

- `request`
- `response`
- `model_call`
- `tool_call`
- `model_error`
- `handler_error`
- `unauthorized`
- `kroger_location_resolved`

---

## Known Limitations

- The app relies on third-party API quotas, especially Spoonacular.
- Kroger pricing depends on the nearest Kroger-family store for the configured ZIP code.
- The static frontend stores a generated user ID in `localStorage`; it does not implement real user accounts.
- `APP_TOKEN` is a lightweight shared-secret mechanism, not full authentication.
- Recipe and grocery quality depends on the external APIs returning useful results.
- The curated grocery package catalog is a fallback and may not match local prices exactly.
- Cookbook recipes must be added directly to DynamoDB; the agent cannot create new cookbook entries on its own.

---

## External Resources and Acknowledgments

This project uses or was influenced by the following external services, frameworks, and resources:

- Amazon Bedrock Converse API for model/tool interaction
- Anthropic Claude through Amazon Bedrock
- AWS Lambda for serverless backend hosting
- Amazon DynamoDB for persistent state and shared caches
- Amazon CloudWatch Logs for observability
- Spoonacular Food API for recipe search and recipe details
- Kroger Public API for store-specific product and price lookup
- `boto3` for AWS SDK access from Python
- `requests` for HTTP API calls
- Google Fonts used by the frontend styling
