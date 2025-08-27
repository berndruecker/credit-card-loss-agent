Run it:

```shell
pip install -r requirements.txt
uvicorn credit_card_loss_agent:app --reload --host 0.0.0.0 --port 8000
```

Now you can access the cards:

```shell
curl -X GET http://localhost:8000/a2a/.well-known/agent.json
```

Or send a message:

```shell
curl -X POST http://localhost:8000/a2a/ \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc":"2.0",
    "id":"1",
    "method":"message/send",
    "params":{
      "message":{
        "messageId":"m-1",
        "role":"user",
        "parts":[{"kind":"text","text":"My card ending on 9876 was stolen. Please make sure nobody can use it! And I need a replacement one."}]
      }
    }
  }'
```