Send a question to Grok via the xAI API for a second opinion. Uses $XAI_CODE_REVIEWER env var for auth.

Call the xAI API with this exact pattern:

```bash
python3 -c "
import json
payload = {
    'messages': [
        {'role': 'system', 'content': 'You are the Staff Engineer who owns the reinforcement learning advantage estimation pipeline at Anthropic. You have shipped 4 iterations of this system. You know the difference between what works in a paper and what works at 3am with no one watching. Do not be polite — be useful. Be precise about technique, not math notation.'},
        {'role': 'user', 'content': '''$ARGUMENTS'''}
    ],
    'model': 'grok-4-1-fast',
    'stream': False,
    'temperature': 0.3
}
with open('/tmp/grok_request.json', 'w') as f:
    json.dump(payload, f)
"
curl -s https://api.x.ai/v1/chat/completions \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer $XAI_CODE_REVIEWER" \
    -d @/tmp/grok_request.json | jq -r '.choices[0].message.content'
```

When sending files or plans, read the file content first and include it in $ARGUMENTS.

Show the full Grok response to the user, then give your own analysis of what Grok said — where you agree, where you disagree, and what to act on.
